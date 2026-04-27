"""Stage J integration regression — real Ausas GPU backend test.

Integration safety net that the stub-based contract tests cannot
provide. Imports the real ``ausas_unsteady_one_step_gpu`` (skipping
cleanly when ``cupy`` or ``GPU_reynolds`` are absent) and runs the
adapter end-to-end on a tiny grid so the regressions cannot recur:

* Bug 1: passing ``eta`` / ``omega`` as kwargs would crash the real
  solver — the adapter must NOT forward them.
* Bug 2: the real solver returns ``(P, theta, residual, n_inner)``,
  not ``(P, theta, n_inner, converged)``.
* Bug 3: the solver does its own ``_pack_ghosts(...,
  periodic_phi=True)``; the adapter must pass the **unpadded**
  physical grid.
* Bug 4: physical seconds must NOT be forwarded as ``dt`` — the
  adapter converts to non-dim ``dt_ausas = ω·dt_s`` first.
* Bug 5: state initialisation from the actual first accepted gap
  (``from_initial_gap``), not ``cold_start`` with ``H_prev=ones``.

Stage J §3.5 — convergence is promoted from "no exception" to a
physical contract: residual finite & ≤ tol, n_inner < max_inner,
P_nd finite, theta in [0, 1].
"""
from __future__ import annotations

import numpy as np
import pytest


cupy = pytest.importorskip(
    "cupy",
    reason="real Ausas backend requires cupy — install GPU_reynolds")
solver_dynamic_gpu = pytest.importorskip(
    "reynolds_solver.cavitation.ausas.solver_dynamic_gpu",
    reason="real Ausas backend requires reynolds_solver")


from models.diesel_ausas_adapter import (  # noqa: E402
    DieselAusasState,
    DieselAusasStepResult,
    ausas_one_step_with_state,
    ausas_result_is_physical,
    set_ausas_backend_for_tests,
)


_OMEGA_DIESEL = 2.0 * np.pi * 1900.0 / 60.0   # rad/s at 1900 rpm
_TOL_REAL = 1e-3
_MAX_INNER_REAL = 500


def _restore_real_backend():
    """No leftover stub from another test file — the integration
    test wants the real GPU backend."""
    set_ausas_backend_for_tests(None)


def _build_smooth_film(N_z: int, N_phi: int, eps: float = 0.30):
    """Smooth-bearing film on the diesel physical grid."""
    phi = np.linspace(0.0, 2.0 * np.pi, int(N_phi), endpoint=False)
    Phi = np.broadcast_to(phi[None, :], (int(N_z), int(N_phi))).copy()
    return 1.0 - float(eps) * np.cos(Phi)


def _common_kwargs(N_z: int, N_phi: int):
    return dict(
        dt_s=np.deg2rad(4.0) / _OMEGA_DIESEL,
        omega_shaft=_OMEGA_DIESEL,
        d_phi=2.0 * np.pi / N_phi,
        d_Z=2.0 / max(N_z - 1, 1),
        R=0.1, L=0.08,
        extra_options={"tol": _TOL_REAL, "max_inner": _MAX_INNER_REAL},
    )


def test_real_ausas_backend_one_step_runs_clean():
    """End-to-end: real backend, real adapter, 16×8 grid. Step
    completes without raising; ``DieselAusasStepResult`` carries
    finite ``P_nd``/``theta`` shaped exactly like the unpadded
    physical grid; state advances by 1; ``dt_ausas`` is non-dim."""
    _restore_real_backend()
    N_phi, N_z = 16, 8
    H_initial = _build_smooth_film(N_z, N_phi)
    state = DieselAusasState.from_initial_gap(H_initial)

    # Bug 5 guard: H_prev must equal H_initial, NOT a unit gap.
    assert np.array_equal(state.H_prev, H_initial)
    assert not np.allclose(state.H_prev, np.ones_like(H_initial))

    out = ausas_one_step_with_state(
        state, H_curr=H_initial, commit=True,
        **_common_kwargs(N_z, N_phi),
    )

    assert isinstance(out, DieselAusasStepResult)
    # Bug 4 guard: dt_ausas is non-dimensional ω·dt_s, NOT dt_s.
    assert out.dt_phys_s == pytest.approx(_common_kwargs(N_z, N_phi)["dt_s"])
    assert out.dt_ausas == pytest.approx(np.deg2rad(4.0))
    # Bug 2 guard: residual is a finite scalar from the canonical
    # 4-tuple shape.
    assert np.isfinite(out.residual)
    assert out.n_inner >= 0
    # Convergence-as-physical-contract.
    assert ausas_result_is_physical(
        out, tol=_TOL_REAL, max_inner=_MAX_INNER_REAL), (
        f"physical-contract failed: residual={out.residual}, "
        f"n_inner={out.n_inner}, theta=[{out.theta_min}, "
        f"{out.theta_max}]")
    # Bug 3 guard: physical (unpadded) grid shape on output.
    assert out.P_nd.shape == (N_z, N_phi)
    assert out.theta.shape == (N_z, N_phi)
    assert state.step_index == 1
    assert np.array_equal(state.H_prev, H_initial)


def test_real_ausas_backend_two_steps_advance_state():
    """Two accepted steps must advance ``step_index`` by 2 and
    sync ``H_prev`` with the latest committed gap."""
    _restore_real_backend()
    N_phi, N_z = 16, 8
    H1 = _build_smooth_film(N_z, N_phi, eps=0.20)
    H2 = _build_smooth_film(N_z, N_phi, eps=0.30)
    state = DieselAusasState.from_initial_gap(H1)
    kw = _common_kwargs(N_z, N_phi)

    for H_step in (H1, H2):
        out = ausas_one_step_with_state(
            state, H_curr=H_step, commit=True, **kw)
        assert ausas_result_is_physical(
            out, tol=_TOL_REAL, max_inner=_MAX_INNER_REAL)

    assert state.step_index == 2
    assert np.array_equal(state.H_prev, H2)
    assert state.dt_last_s == pytest.approx(kw["dt_s"])
    assert state.time_s == pytest.approx(2 * kw["dt_s"])
    # Bug 4 guard: ``dt_ausas_last`` should be non-dim.
    assert state.dt_ausas_last == pytest.approx(np.deg2rad(4.0))


def test_real_ausas_backend_trial_does_not_mutate():
    """``commit=False`` against the real solver must leave every
    state field unchanged."""
    _restore_real_backend()
    N_phi, N_z = 16, 8
    H_initial = _build_smooth_film(N_z, N_phi)
    state = DieselAusasState.from_initial_gap(H_initial)
    P_before = state.P.copy()
    theta_before = state.theta.copy()
    H_prev_before = state.H_prev.copy()

    out = ausas_one_step_with_state(
        state, H_curr=H_initial, commit=False,
        **_common_kwargs(N_z, N_phi),
    )
    assert isinstance(out, DieselAusasStepResult)
    assert ausas_result_is_physical(
        out, tol=_TOL_REAL, max_inner=_MAX_INNER_REAL)
    assert np.array_equal(state.P, P_before)
    assert np.array_equal(state.theta, theta_before)
    assert np.array_equal(state.H_prev, H_prev_before)
    assert state.step_index == 0
