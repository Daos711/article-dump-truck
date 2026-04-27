"""Stage J — diesel-side Ausas adapter contract tests.

These tests do **not** require the GPU solver. The wrapper
``ausas_one_step_with_state`` resolves its backend lazily via
``models.diesel_ausas_adapter._resolve_ausas_backend`` and exposes
``set_ausas_backend_for_tests`` so we can install a tiny Python stub
that exercises the commit/state semantics deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_ausas_adapter import (
    DieselAusasState,
    ausas_one_step_with_state,
    pad_phi_for_ausas,
    set_ausas_backend_for_tests,
    unpad_phi_from_ausas,
)


# ─── 1. Phi padding round-trip ─────────────────────────────────────


def test_phi_padding_roundtrip():
    """``unpad(pad(H)) == H`` for the diesel endpoint-free grid; the
    seam ghost columns must mirror the physical wrap exactly."""
    rng = np.random.default_rng(20240426)
    H = rng.normal(size=(8, 16))
    H_pad = pad_phi_for_ausas(H)
    assert H_pad.shape == (8, 16 + 2)
    # Left ghost column reflects column N_phi-1; right ghost
    # reflects column 0 (endpoint-free wrap convention).
    assert np.array_equal(H_pad[:, 0], H[:, -1])
    assert np.array_equal(H_pad[:, -1], H[:, 0])
    # Round-trip must match the input bit-for-bit.
    H_back = unpad_phi_from_ausas(H_pad)
    assert np.array_equal(H_back, H)


# ─── Helper: deterministic stub backend ────────────────────────────


def _make_backend(*, n_inner: int = 7,
                  residual: float = 1e-9,
                  raise_exc: bool = False):
    """Return a Python stub that mimics the real
    ``ausas_unsteady_one_step_gpu`` 4-tuple return shape
    ``(P, theta, residual, n_inner)`` — Stage J integration
    regression Bug 2 fix. The adapter trusts the solver: a finite
    return with no explicit non-convergence flag is an accepted
    step. To force the derived ``ok`` flag to ``False``, the caller
    must pass ``max_inner`` in ``extra_options`` such that
    ``n_inner >= max_inner`` — which is precisely the budget-
    exhausted condition that the live runner would also detect."""
    calls: list = []

    def fake(**kwargs):
        calls.append(dict(kwargs))
        if raise_exc:
            raise RuntimeError("stub: simulated solver failure")
        H = kwargs["H_curr"]
        # Synthetic but deterministic: P = 0.5*H, theta = clip(H, 0, 1).
        P = 0.5 * H
        theta = np.clip(H, 0.0, 1.0)
        return (P, theta, float(residual), int(n_inner))

    fake.calls = calls
    return fake


# ─── 2. State NOT mutated on trial solve (commit=False) ────────────


def test_state_not_mutated_on_trial_solve():
    """Calling the wrapper with ``commit=False`` must leave the
    state arrays unchanged regardless of the solver's output."""
    backend = _make_backend()
    set_ausas_backend_for_tests(backend)
    try:
        state = DieselAusasState.cold_start(N_phi=8, N_z=4)
        P_before = state.P.copy()
        theta_before = state.theta.copy()
        H_prev_before = state.H_prev.copy()
        step_index_before = state.step_index
        H_curr = np.ones((4, 8))
        result = ausas_one_step_with_state(
            state, H_curr=H_curr, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=False,
        )
        assert result.converged is True
        # State arrays must be untouched.
        assert np.array_equal(state.P, P_before)
        assert np.array_equal(state.theta, theta_before)
        assert np.array_equal(state.H_prev, H_prev_before)
        assert state.step_index == step_index_before
        # And the trial result must still report sensible diagnostics.
        assert result.P_nd is not None
        assert result.P_nd.shape == (4, 8)
        assert result.theta.shape == (4, 8)
    finally:
        set_ausas_backend_for_tests(None)


# ─── 3. State committed exactly once per accepted step ─────────────


def test_state_committed_once_per_step():
    """Two trial solves followed by one commit must increment
    ``step_index`` by exactly 1 — multiple Verlet candidates are
    explored, but only the accepted one writes to the state."""
    backend = _make_backend()
    set_ausas_backend_for_tests(backend)
    try:
        state = DieselAusasState.cold_start(N_phi=8, N_z=4)
        H_trial1 = np.ones((4, 8)) * 0.9
        H_trial2 = np.ones((4, 8)) * 1.1
        H_committed = np.ones((4, 8)) * 1.05
        # Two trial solves — neither must mutate the state.
        ausas_one_step_with_state(
            state, H_curr=H_trial1, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=False)
        ausas_one_step_with_state(
            state, H_curr=H_trial2, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=False)
        assert state.step_index == 0
        # Final commit — exactly one step advance.
        ausas_one_step_with_state(
            state, H_curr=H_committed, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=True)
        assert state.step_index == 1
        assert state.dt_last_s == pytest.approx(1e-4)
        # Stage J integration regression Bug 3 — state arrays now
        # live on the unpadded physical grid (the solver pads
        # internally), so ``H_prev`` must equal the committed H
        # bit-for-bit, NOT a pre-padded version of it.
        assert state.H_prev.shape == H_committed.shape
        assert np.array_equal(state.H_prev, H_committed)
    finally:
        set_ausas_backend_for_tests(None)


# ─── 4. Failed step does not poison the state ──────────────────────


def test_failed_step_does_not_poison_state():
    """When the backend raises, the state must remain valid (old
    arrays preserved, no NaN), and ``failed_step_count`` must
    increment so the runner sees the failure."""
    bad_backend = _make_backend(raise_exc=True)
    set_ausas_backend_for_tests(bad_backend)
    try:
        state = DieselAusasState.cold_start(N_phi=8, N_z=4)
        P_before = state.P.copy()
        theta_before = state.theta.copy()
        H_curr = np.ones((4, 8))
        result = ausas_one_step_with_state(
            state, H_curr=H_curr, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=True,
        )
        assert result.converged is False
        assert "ausas_one_step_failed" in result.reason
        # State preserved bit-for-bit.
        assert np.array_equal(state.P, P_before)
        assert np.array_equal(state.theta, theta_before)
        assert state.step_index == 0
        assert state.failed_step_count == 1
        assert not np.any(np.isnan(state.P))
        assert not np.any(np.isnan(state.theta))

        # Subsequent successful step (with a healthy backend) must
        # commit cleanly from the unchanged state.
        good_backend = _make_backend()
        set_ausas_backend_for_tests(good_backend)
        result2 = ausas_one_step_with_state(
            state, H_curr=H_curr, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=True,
        )
        assert result2.converged is True
        assert state.step_index == 1
    finally:
        set_ausas_backend_for_tests(None)


# ─── 5. Legacy half-Sommerfeld path does not require Ausas state ───


def test_half_sommerfeld_path_does_not_require_ausas_state():
    """The diesel transient runner constructs an Ausas state only
    when ``cavitation='ausas_dynamic'``. The legacy half-Sommerfeld
    path must remain importable and runnable without ever touching
    the adapter — this test pins that boundary by importing
    ``run_transient`` and checking that ``ausas_dynamic`` is not in
    its default cavitation."""
    import inspect
    from models.diesel_transient import run_transient
    sig = inspect.signature(run_transient)
    # Default must remain the legacy half-Sommerfeld closure (no
    # silent flip to Ausas).
    assert sig.parameters["cavitation"].default == "half_sommerfeld"
    # And the new Stage J kwargs are present with safe defaults.
    assert sig.parameters["texture_kind"].default == "dimple"
    assert sig.parameters["groove_preset"].default is None
    assert sig.parameters["fidelity"].default is None
    assert sig.parameters["ausas_options"].default is None


def test_commit_accepts_step_when_max_inner_not_overridden():
    """When the caller does NOT override ``max_inner`` in
    ``extra_options``, the adapter must trust the solver: a finite
    return value (no explicit non-convergence flag) means the step
    is accepted, ``step_index`` advances by 1, and
    ``rejected_commit_count`` stays 0.

    Regression guard for the live-GPU smoke that hit
    ``rejected_commit_count = 1, step_index = 0`` after every step
    because the adapter was comparing the solver's residual against
    a hard-coded local ``tol = 1e-6`` and demanding ``n_inner <
    200`` against a hard-coded local ``max_inner``. Both numbers
    were guesses unrelated to the solver's internal stopping
    criterion, and they silently rejected accepted steps.
    """
    # Backend reports n_inner=150 with a residual of 5e-3 — well
    # above the legacy local tol guess of 1e-6. The runner did not
    # pass ``max_inner`` overrides, so the adapter has no business
    # second-guessing the solver: the step is accepted.
    backend = _make_backend(n_inner=150, residual=5e-3)
    set_ausas_backend_for_tests(backend)
    try:
        state = DieselAusasState.cold_start(N_phi=8, N_z=4)
        H = np.ones((4, 8))
        out = ausas_one_step_with_state(
            state, H_curr=H, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            commit=True,
        )
        assert out.converged is True
        assert state.step_index == 1
        assert state.rejected_commit_count == 0
        assert out.residual == pytest.approx(5e-3)
        assert out.n_inner == 150
    finally:
        set_ausas_backend_for_tests(None)


def test_commit_rejects_step_when_n_inner_hits_user_max_inner():
    """When the caller explicitly passes ``max_inner`` in
    ``extra_options`` and the solver returns ``n_inner >=
    max_inner``, the adapter must mark the step as not converged
    (ok=False) and increment ``rejected_commit_count`` without
    advancing ``step_index``."""
    backend = _make_backend(n_inner=200, residual=1.0)
    set_ausas_backend_for_tests(backend)
    try:
        state = DieselAusasState.cold_start(N_phi=8, N_z=4)
        H = np.ones((4, 8))
        out = ausas_one_step_with_state(
            state, H_curr=H, dt_s=1e-4,
            omega_shaft=200.0,
            d_phi=0.1, d_Z=0.5, R=0.1, L=0.08,
            extra_options={"max_inner": 200},
            commit=True,
        )
        assert out.converged is False
        assert state.step_index == 0
        assert state.rejected_commit_count == 1
    finally:
        set_ausas_backend_for_tests(None)


def test_set_backend_none_clears_cache_for_re_probe():
    """``set_ausas_backend_for_tests(None)`` must clear the resolver
    cache so the next call re-probes for the real GPU backend.

    Stage J integration regression follow-up — the previous test
    hook set ``_AUSAS_BACKEND_PROBED = True`` even when ``fn is
    None``, which permanently locked the resolver onto ``None``.
    Contract tests that installed a stub and then cleaned up with
    ``set_ausas_backend_for_tests(None)`` left the real-backend
    integration tests unable to ever discover the GPU package.
    """
    from models import diesel_ausas_adapter as _m
    backend = _make_backend()
    set_ausas_backend_for_tests(backend)
    assert _m._AUSAS_BACKEND_PROBED is True
    assert _m._AUSAS_ONE_STEP_BACKEND is backend
    # Clearing must reset BOTH cache slots so the resolver re-probes.
    set_ausas_backend_for_tests(None)
    assert _m._AUSAS_BACKEND_PROBED is False, (
        "set_ausas_backend_for_tests(None) must clear the cache, "
        "not lock _AUSAS_ONE_STEP_BACKEND as None.")
    assert _m._AUSAS_ONE_STEP_BACKEND is None
    # And re-installing a different stub afterwards still works.
    backend2 = _make_backend()
    set_ausas_backend_for_tests(backend2)
    assert _m._AUSAS_ONE_STEP_BACKEND is backend2
    set_ausas_backend_for_tests(None)
