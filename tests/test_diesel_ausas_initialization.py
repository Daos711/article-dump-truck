"""Stage J Bug 5 — initialisation contract for ``DieselAusasState``.

The legacy ``cold_start`` constructor allocated ``H_prev = ones``,
which injected a fictitious squeeze pulse on the first mechanical
step (the diesel runner's first H is generally non-unit for any
non-zero ``eps_x0`` / ``eps_y0``). The squeeze term ``∂(θH)/∂t``
then blew up the pressure on step #1 and collapsed the orbit.

The new ``from_initial_gap(H_initial)`` constructor stores the
**actual** first accepted gap as ``H_prev``, so step #1 sees a
zero-time-derivative consistent with rest. These tests pin that
contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_ausas_adapter import (
    DieselAusasState,
    ausas_one_step_with_state,
    set_ausas_backend_for_tests,
)


def _non_centred_gap(N_z: int = 4, N_phi: int = 8, eps: float = 0.4):
    """Smooth-bearing film at non-zero eccentricity — guarantees
    that H_initial != ones, so the regression test can distinguish
    the new constructor from the legacy ``cold_start``."""
    phi = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    Phi = np.broadcast_to(phi[None, :], (N_z, N_phi)).copy()
    return 1.0 - eps * np.cos(Phi)


def test_from_initial_gap_stores_actual_first_h():
    """``H_prev`` must equal the supplied ``H_initial`` byte-for-byte
    (NOT a unit gap), and ``P``/``theta`` default to identity."""
    H_initial = _non_centred_gap()
    state = DieselAusasState.from_initial_gap(H_initial)

    assert np.array_equal(state.H_prev, H_initial)
    # And explicitly NOT the legacy unit-gap default.
    assert not np.allclose(state.H_prev, np.ones_like(H_initial))
    # Identity defaults for P and theta.
    assert np.array_equal(state.P, np.zeros_like(H_initial))
    assert np.array_equal(state.theta, np.ones_like(H_initial))
    # All counters at the cold-start origin.
    assert state.step_index == 0
    assert state.time_s == 0.0
    assert state.dt_last_s == 0.0
    assert state.dt_ausas_last == 0.0


def test_from_initial_gap_accepts_warmup_overrides():
    """Optional warmup ``P_warm`` / ``theta_warm`` must replace the
    identity defaults so a steady-state pressure can seed step #1."""
    H_initial = _non_centred_gap()
    P_warm = 0.1 + np.zeros_like(H_initial)
    theta_warm = 0.5 + np.zeros_like(H_initial)
    state = DieselAusasState.from_initial_gap(
        H_initial, P_warm=P_warm, theta_warm=theta_warm)
    assert np.array_equal(state.P, P_warm)
    assert np.array_equal(state.theta, theta_warm)
    # H_prev must STILL track the supplied initial gap.
    assert np.array_equal(state.H_prev, H_initial)


def test_first_step_with_h_curr_equals_h_prev_has_no_squeeze_pulse():
    """When ``H_curr == H_prev`` the time-derivative ``∂(θH)/∂t``
    is identically zero and the steady solution should drop out of
    a single Ausas step. We use a linear stub that obeys this:
    ``P = c * (H_curr - H_prev)`` with ``c=10``. With the new
    ``from_initial_gap`` initialiser and ``H_curr = H_initial``
    the stub returns ``P = 0`` because there is no squeeze pulse.

    With the legacy ``cold_start`` (``H_prev = ones``) the same
    call would return ``P = 10 * (H_initial - 1) != 0``: the
    artificial squeeze that this Stage J fix eliminates.
    """
    H_initial = _non_centred_gap()
    pulse_coeff = 10.0

    def squeeze_sensitive_backend(**kwargs):
        H_curr = kwargs["H_curr"]
        H_prev = kwargs["H_prev"]
        # Linear "pseudo-pressure" proportional to ∂H/∂t (same
        # functional form the squeeze term has at first order).
        P = pulse_coeff * (H_curr - H_prev)
        theta = np.ones_like(H_curr)
        return (P, theta, 0.0, 1)

    set_ausas_backend_for_tests(squeeze_sensitive_backend)
    try:
        # New contract: H_prev = H_initial → no squeeze, P = 0.
        state_new = DieselAusasState.from_initial_gap(H_initial)
        out_new = ausas_one_step_with_state(
            state_new, H_curr=H_initial, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08, commit=False,
        )
        assert np.max(np.abs(out_new.P_nd)) < 1e-12

        # Legacy contract: H_prev = ones → fictitious pulse.
        state_legacy = DieselAusasState.cold_start(
            N_phi=H_initial.shape[1], N_z=H_initial.shape[0])
        out_legacy = ausas_one_step_with_state(
            state_legacy, H_curr=H_initial, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08, commit=False,
        )
        assert np.max(np.abs(out_legacy.P_nd)) > 0.1, (
            "Sanity: legacy cold_start should produce a nonzero "
            "fictitious squeeze response under the linear stub. "
            "If this assertion fails the test itself is broken.")
    finally:
        set_ausas_backend_for_tests(None)
