"""Stage J Bug 4 — time non-dimensionalisation contract.

The dynamic Ausas solver consumes time in shaft-rotation units
``τ = ω · t``. The previous adapter forwarded physical seconds
directly, which inflated the unsteady coefficient by ``ω`` (~199 at
1900 rpm) and made the squeeze term dominate the Couette term — the
exact failure mode that collapsed the orbit on step #1.

These tests pin the conversion at the public-API level so any
future refactor that drops ``omega_shaft`` from the call site
breaks here, not in production.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from models.diesel_ausas_adapter import (
    DieselAusasState,
    ausas_dt_from_physical,
    ausas_one_step_with_state,
    set_ausas_backend_for_tests,
)


def test_ausas_dt_is_omega_times_physical_dt():
    """The canonical case: at 1900 rpm with d_phi_crank = 4°, the
    physical dt is ``Δφ / ω`` and the Ausas dt must equal Δφ in
    radians."""
    omega = 2.0 * np.pi * 1900.0 / 60.0
    dt_s = np.deg2rad(4.0) / omega
    dt_ausas = ausas_dt_from_physical(dt_s, omega)
    assert dt_ausas == pytest.approx(np.deg2rad(4.0))


def test_ausas_dt_helper_is_linear():
    """Doubling either the physical timestep or the shaft speed
    must double ``dt_ausas`` — guards against accidental
    non-linear conversions."""
    base = ausas_dt_from_physical(1e-4, 200.0)
    assert ausas_dt_from_physical(2e-4, 200.0) == pytest.approx(2 * base)
    assert ausas_dt_from_physical(1e-4, 400.0) == pytest.approx(2 * base)


def test_omega_shaft_is_required_kwarg_in_one_step_signature():
    """``ausas_one_step_with_state`` must require ``omega_shaft``
    as a keyword argument — pin the public-API contract so a
    future refactor cannot quietly drop it and reintroduce the
    physical-seconds-as-tau regression."""
    sig = inspect.signature(ausas_one_step_with_state)
    assert "omega_shaft" in sig.parameters
    p = sig.parameters["omega_shaft"]
    # Must be keyword-only (after the bare ``*``).
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    # And must have no default — caller MUST supply it.
    assert p.default is inspect.Parameter.empty


def test_adapter_forwards_dt_ausas_not_physical_seconds():
    """Install a stub that records the ``dt`` it receives and
    confirm the adapter passes ``ω·dt_s``, not ``dt_s``."""
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        H = kwargs["H_curr"]
        # Trivial converged answer so the adapter accepts the step.
        return (np.zeros_like(H), np.ones_like(H), 0.0, 1)

    set_ausas_backend_for_tests(fake)
    try:
        omega = 200.0
        dt_phys = 5e-4
        H = np.ones((4, 8))
        state = DieselAusasState.from_initial_gap(H)
        ausas_one_step_with_state(
            state, H_curr=H, dt_s=dt_phys, omega_shaft=omega,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08, commit=True,
        )
    finally:
        set_ausas_backend_for_tests(None)
    # The adapter must have forwarded τ = ω·dt_s, NOT dt_phys.
    assert captured["dt"] == pytest.approx(omega * dt_phys)
    assert captured["dt"] != pytest.approx(dt_phys)
