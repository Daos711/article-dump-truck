"""Stage J integration regression — real Ausas GPU backend test.

This test is the **integration safety net** that the stub-based
contract tests in ``test_diesel_ausas_adapter.py`` and
``test_diesel_transient_ausas_contract.py`` do **not** provide. It
imports the real ``ausas_unsteady_one_step_gpu`` (skipping cleanly
when ``cupy`` or ``GPU_reynolds`` are not installed) and runs the
adapter end-to-end on a tiny grid so the three regressions caught
in production cannot recur silently:

* Bug 1: passing ``eta`` / ``omega`` as kwargs would crash the real
  solver — the adapter must NOT forward them.
* Bug 2: the real solver returns ``(P, theta, residual, n_inner)``,
  not ``(P, theta, n_inner, converged)`` — the adapter's
  ``_unpack_ausas_return`` must consume the canonical shape.
* Bug 3: the real solver does its own ``_pack_ghosts(...,
  periodic_phi=True)`` — the adapter must pass the **unpadded**
  physical grid, otherwise every step is double-wrapped.

The test runs only when the GPU stack is fully available (no
mocking, no stubs). On CI hosts without ``cupy`` or without the
``reynolds_solver`` package the test reports a single skip, not a
failure — but if the package is present the test is guaranteed to
exercise the real call path.
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
    ausas_one_step_with_state,
    set_ausas_backend_for_tests,
)


def _restore_real_backend():
    """Make sure no leftover stub from another test file is still
    in place — the integration test wants the real GPU backend."""
    set_ausas_backend_for_tests(None)


def _build_smooth_film(N_z: int, N_phi: int, eps: float = 0.30):
    """Build a deterministic smooth-bearing film on the diesel
    physical grid (endpoint-free phi)."""
    phi = np.linspace(0.0, 2.0 * np.pi, int(N_phi), endpoint=False)
    Phi = np.broadcast_to(phi[None, :], (int(N_z), int(N_phi))).copy()
    H = 1.0 - float(eps) * np.cos(Phi)
    return H


def test_real_ausas_backend_one_step_runs_clean():
    """End-to-end integration: real backend, real adapter, tiny
    16×8 grid. The step must complete without raising, return
    finite ``P_phys`` / ``theta_phys`` shaped exactly like the
    physical grid (NOT padded to N_phi+2), and write the committed
    state back unchanged-shape."""
    _restore_real_backend()
    N_phi, N_z = 16, 8
    state = DieselAusasState.cold_start(N_phi=N_phi, N_z=N_z)
    # Precondition: state lives on the unpadded physical grid
    # (Bug 3 regression guard).
    assert state.P.shape == (N_z, N_phi)
    assert state.theta.shape == (N_z, N_phi)
    assert state.H_prev.shape == (N_z, N_phi)

    H_curr = _build_smooth_film(N_z, N_phi)
    out = ausas_one_step_with_state(
        state,
        H_curr_phys=H_curr,
        dt_s=1e-4,
        d_phi=2.0 * np.pi / N_phi,
        d_Z=2.0 / max(N_z - 1, 1),
        R=0.1, L=0.08,
        # No eta / omega — Bug 1 regression guard.
        commit=True,
    )

    # Bug 2 regression guard: ``residual`` must be present and a
    # finite scalar (the canonical real-solver 4-tuple is
    # ``(P, theta, residual, n_inner)``).
    assert "residual" in out
    assert np.isfinite(float(out["residual"]))
    assert out["n_inner"] >= 0

    # Output fields keep the physical (unpadded) grid shape.
    P = np.asarray(out["P_phys"])
    theta = np.asarray(out["theta_phys"])
    assert P.shape == (N_z, N_phi), (
        "Stage J Bug 3: real solver returned a padded array. The "
        "adapter must pass and receive the unpadded physical grid."
    )
    assert theta.shape == (N_z, N_phi)
    assert np.all(np.isfinite(P))
    assert np.all(np.isfinite(theta))
    # Theta must stay in the physical [0, 1] range.
    assert float(theta.min()) >= -1e-9
    assert float(theta.max()) <= 1.0 + 1e-9

    # State commit — shape preserved, step counter advanced.
    assert state.P.shape == (N_z, N_phi)
    assert state.theta.shape == (N_z, N_phi)
    assert state.H_prev.shape == (N_z, N_phi)
    assert state.step_index == 1
    assert np.array_equal(state.H_prev, H_curr)


def test_real_ausas_backend_two_steps_advance_state():
    """Two consecutive accepted steps must advance the state by
    exactly 2 and keep ``H_prev`` synchronised with the most-recent
    accepted gap. Catches any regression where the adapter forgets
    to commit or commits more than once."""
    _restore_real_backend()
    N_phi, N_z = 16, 8
    state = DieselAusasState.cold_start(N_phi=N_phi, N_z=N_z)

    H1 = _build_smooth_film(N_z, N_phi, eps=0.20)
    H2 = _build_smooth_film(N_z, N_phi, eps=0.30)

    for H_step in (H1, H2):
        out = ausas_one_step_with_state(
            state,
            H_curr_phys=H_step,
            dt_s=1e-4,
            d_phi=2.0 * np.pi / N_phi,
            d_Z=2.0 / max(N_z - 1, 1),
            R=0.1, L=0.08,
            commit=True,
        )
        assert np.all(np.isfinite(out["P_phys"]))

    assert state.step_index == 2
    assert np.array_equal(state.H_prev, H2)
    assert state.dt_last_s == pytest.approx(1e-4)
    # Time advances by Σ dt.
    assert state.time_s == pytest.approx(2e-4)


def test_real_ausas_backend_trial_does_not_mutate():
    """``commit=False`` against the real solver must leave every
    state field byte-for-byte unchanged. This is the contract the
    Verlet substep loop relies on for trial pressure evaluation."""
    _restore_real_backend()
    N_phi, N_z = 16, 8
    state = DieselAusasState.cold_start(N_phi=N_phi, N_z=N_z)
    P_before = state.P.copy()
    theta_before = state.theta.copy()
    H_prev_before = state.H_prev.copy()

    H_trial = _build_smooth_film(N_z, N_phi)
    out = ausas_one_step_with_state(
        state,
        H_curr_phys=H_trial,
        dt_s=1e-4,
        d_phi=2.0 * np.pi / N_phi,
        d_Z=2.0 / max(N_z - 1, 1),
        R=0.1, L=0.08,
        commit=False,
    )
    assert np.all(np.isfinite(out["P_phys"]))
    assert np.array_equal(state.P, P_before)
    assert np.array_equal(state.theta, theta_before)
    assert np.array_equal(state.H_prev, H_prev_before)
    assert state.step_index == 0
