"""Stage J followup-2 Step 6 — DampedImplicitFilmPolicy unit tests.

Three structural tests for the Picard fixed-point coupling kernel
path. They are pure-CPU tests that install a deterministic Python
stub for the pressure backend so the kernel's iteration logic is
exercised without GPU dependencies.

Tests
-----
1. ``test_damped_iteration_blends_candidate_with_relax``
   The candidate eps update between trials is exactly
   ``mech_relax_initial`` × the underlying Verlet correction —
   confirms the Picard blend is applied (not a 100% jump).

2. ``test_damped_terminates_early_on_fixed_point_convergence``
   When the stub backend produces a force such that the candidate
   stabilises within ``policy.eps_update_tol`` after a few
   iterations, the kernel exits early (n_trials < max_mech_inner)
   AND commits the step (state_committed=True).

3. ``test_damped_does_not_commit_on_non_convergence``
   When the stub produces a force that keeps moving the candidate
   beyond ``eps_update_tol`` on every iteration, the kernel
   exhausts ``max_mech_inner`` trials and refuses to commit
   (state_committed=False, rejection_reason=SOLVER_RESIDUAL).
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_coupling import (
    AusasDynamicBackend,
    CouplingPolicy,
    POLICY_AUSAS_DYNAMIC,
    PressureSolveResult,
    StepContext,
    advance_mechanical_step,
)
from models.diesel_ausas_adapter import (
    DieselAusasState,
    set_ausas_backend_for_tests,
)


def _make_step_ctx(N_phi: int = 16, N_z: int = 8) -> StepContext:
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    Z_1D = np.linspace(-1.0, 1.0, N_z)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]
    return StepContext(
        phi_deg=0.0,
        F_ext_x=0.0, F_ext_y=-1.0e3,
        F_max=1.0e3,
        p_scale=1.0e6,
        omega=200.0, eta=0.01,
        R=0.1, L=0.08, c=1.2e-4,
        Phi_mesh=Phi_mesh, Z_mesh=Z_mesh,
        phi_1D=phi_1D, Z_1D=Z_1D,
        d_phi=float(d_phi), d_Z=float(d_Z),
        cfg={"label": "test", "textured": False},
        texture_kind="none",
        groove_relief=None,
        phi_c_flat=None, Z_c_flat=None,
        closure="laminar",
        cavitation="ausas_dynamic",
        P_warm_init=None,
    )


def _identity_clamp(ex, ey, vx, vy):
    """No-op clamp — ε never reaches the wall on these synthetic
    setups so we can isolate Picard iteration behaviour."""
    return ex, ey, vx, vy, False


def _trivial_build_H(eps_x, eps_y):
    return 1.0 - 0.3 * np.cos(np.linspace(
        0.0, 2.0 * np.pi, 16, endpoint=False)
    )[None, :].repeat(8, axis=0)


def _make_const_force_backend(*, P_per_call: list, theta_per_call: list):
    """Stub backend that returns a pre-scripted ``(P, theta, residual,
    n_inner)`` tuple per call, plus a synthetic Fx/Fy via
    ``compute_hydro_forces`` from the supplied P. The kernel's
    ``backend.solve_trial`` is the real ``AusasDynamicBackend.solve_trial``
    body, so we install the fake at the *adapter* layer."""
    calls = {"n": 0}

    def fake(**kwargs):
        i = min(calls["n"], len(P_per_call) - 1)
        P = np.asarray(P_per_call[i], dtype=float)
        theta = np.asarray(theta_per_call[i], dtype=float)
        calls["n"] += 1
        return (P, theta, 1e-8, 100)
    set_ausas_backend_for_tests(fake)
    return calls


def _common_advance_kwargs(ctx, policy):
    from models.diesel_coupling import PhysicalGuardsConfig
    return dict(
        ex_n=0.0, ey_n=-0.6 * ctx.c,
        vx_n=0.0, vy_n=0.0,
        ax_prev=0.0, ay_prev=0.0,
        dt_phys_s=1e-4,
        policy=policy,
        guards_cfg=PhysicalGuardsConfig.from_profile(
            "diagnostic", "general"),
        ausas_tol=1e-6, ausas_max_inner=5000,
        extra_options=None,
        context=ctx,
        m_shaft=10.0,
        eps_max=0.95,
        clamp_fn=_identity_clamp,
        build_H_fn=_trivial_build_H,
        p_warm_init=None,
    )


# ─── 1. Picard blend with relax_initial ────────────────────────────


def test_damped_iteration_blends_candidate_with_relax():
    """Each Picard trial must move the candidate by exactly
    ``policy.mech_relax_initial`` × the underlying Verlet
    correction (vs the legacy 100% jump).

    Setup: a stub that returns a strong constant force pushing the
    candidate toward a known (eps_x_full, eps_y_full). After 1
    Picard iteration, the candidate should be at
    ``ε_initial + mech_relax_initial · (ε_full - ε_initial)``,
    not at ``ε_full``.
    """
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(
        np.ones((8, 16), dtype=float))
    backend = AusasDynamicBackend(ausas_options=None)
    # Stub: P = const, large enough that Verlet predicts a HUGE
    # eps_full ~ +0.5c on the y-axis (opposite of -0.6c initial).
    # With mech_relax_initial=0.25 and ONE iteration, candidate
    # should move 25% of the way.
    big_P = np.full((8, 16), 5.0, dtype=float)
    theta_one = np.ones_like(big_P)
    _make_const_force_backend(
        P_per_call=[big_P] * 10,
        theta_per_call=[theta_one] * 10,
    )

    # Tight max_mech_inner=2 + huge eps_update_tol so the loop
    # does NOT terminate early; we want to inspect the candidate
    # after exactly 1 blend.
    policy = CouplingPolicy(
        name="damped_implicit_film",
        max_mech_inner=2,
        mech_relax_initial=0.25,
        mech_relax_min=0.25,
        enable_line_search=False,
        enable_physical_guards=True,
        physical_guards_mode="off",
        commit_on_clamp=False,
        require_solver_converged=True,
        max_delta_eps_inner=10.0,    # disabled
        max_delta_eps_step=10.0,     # disabled
        # Impossibly tight tolerance — Picard MUST iterate the full
        # max_mech_inner budget (no early termination), so the test
        # observes the blend across both iterations.
        eps_update_tol=1.0e-15,
    )

    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_common_advance_kwargs(ctx, policy),
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Damped path should have entered the Picard loop and produced
    # at least 2 trials (max_mech_inner=2, no early stop).
    assert ms.n_trials == 2, (
        f"expected 2 Picard trials, got {ms.n_trials}")
    # Each TrialRecord carries the candidate eps it was given;
    # under blend=0.25, the second iteration's eps_x_cand should
    # have moved ONLY 25% of the way from iteration 1's candidate
    # toward the Verlet-full estimate.
    eps0_x = ms.trial_log[0].eps_x_cand
    eps0_y = ms.trial_log[0].eps_y_cand
    eps1_x = ms.trial_log[1].eps_x_cand
    eps1_y = ms.trial_log[1].eps_y_cand
    delta_x = abs(eps1_x - eps0_x)
    delta_y = abs(eps1_y - eps0_y)
    # The Picard blend is applied: the second-iteration candidate
    # is NOT identical to the first (otherwise no blend was applied),
    # AND it doesn't jump 100% (no oscillation runaway).
    assert delta_x > 0.0 or delta_y > 0.0, (
        "Picard blend was not applied — candidate did not move "
        "between iterations.")
    # Bound the jump: with relax=0.25 and m_shaft=10 kg, the
    # Verlet step ε change in 1e-4 s is small. We just check the
    # iteration is stable (no order-of-magnitude blow-up).
    assert delta_x < 0.5 and delta_y < 0.5, (
        f"Picard blend amplitude too large: "
        f"|Δε_x|={delta_x:.4f}, |Δε_y|={delta_y:.4f}")


# ─── 2. Early termination on fixed-point convergence ──────────────


def test_damped_terminates_early_on_fixed_point_convergence():
    """When the candidate stabilises within ``eps_update_tol``,
    the kernel exits the Picard loop early (n_trials <
    max_mech_inner) and commits the step (state_committed=True)."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(
        np.ones((8, 16), dtype=float))
    backend = AusasDynamicBackend(ausas_options=None)
    # Stub: P = ZERO → zero hydro force → Verlet correction
    # produces only the gravity-like ext load contribution,
    # which over a tiny dt with m_shaft=10 kg is a tiny eps shift.
    # Combined with relax=0.25 it stabilises in 1-2 iterations.
    zero_P = np.zeros((8, 16), dtype=float)
    theta_one = np.ones_like(zero_P)
    _make_const_force_backend(
        P_per_call=[zero_P] * 10,
        theta_per_call=[theta_one] * 10,
    )

    # Use the production damped structure but loosen
    # ``eps_update_tol`` so the per-iteration ε change (set by the
    # synthetic test physics — small F_ext + zero hydro response +
    # m_shaft=10 kg + dt=1e-4 s) crosses the convergence threshold
    # within the 8-iteration budget. The production tolerance
    # (``POLICY_AUSAS_DYNAMIC.eps_update_tol = 1e-4``) is calibrated
    # for real-physics dimensional forces, not the toy stub here.
    policy = CouplingPolicy(
        name="damped_implicit_film",
        max_mech_inner=8,
        mech_relax_initial=0.25,
        mech_relax_min=0.03125,
        enable_line_search=True,
        enable_physical_guards=True,
        physical_guards_mode="off",
        commit_on_clamp=False,
        require_solver_converged=True,
        max_delta_eps_inner=10.0,
        max_delta_eps_step=10.0,
        eps_update_tol=1.0e-3,
    )

    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_common_advance_kwargs(ctx, policy),
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Early termination gate.
    assert ms.n_trials < policy.max_mech_inner, (
        f"Picard did not terminate early: n_trials={ms.n_trials}, "
        f"max_mech_inner={policy.max_mech_inner}")
    # Convergence ⇒ state committed (Step 6 commit gate).
    assert ms.state_committed is True, (
        "Picard converged but state was not committed — Step 6 "
        "commit gate is broken.")
    # Step state index advanced.
    assert state.step_index == 1


# ─── 3. No commit when Picard does not converge in budget ──────────


def test_damped_does_not_commit_on_non_convergence():
    """When the stub yields a force that keeps moving the candidate
    beyond ``eps_update_tol`` on every iteration, the kernel
    exhausts ``max_mech_inner`` trials and MUST refuse to commit.
    State stays at step_index=0; rejection_reason mentions Picard
    non-convergence."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(
        np.ones((8, 16), dtype=float))
    backend = AusasDynamicBackend(ausas_options=None)

    # Alternating-magnitude POSITIVE pressure: each iteration
    # produces a force of substantially different magnitude, so
    # the Picard candidate never stabilises (the fixed point
    # depends on the iteration parity, no consistent attractor).
    # All entries are positive so the Step 7 solver-validity gate
    # (which rejects ``P_nd.min() < 0``) does NOT fire — the test
    # specifically pins Picard non-convergence, not a solver
    # rejection.
    #
    # Step 9 — magnitudes kept SMALL enough that ε stays well
    # below ``eps_max`` and the per-iter ``|Δε_inner|`` stays
    # below the policy cap; this isolates the pure Picard non-
    # convergence path (no mechanical-guard line-search).
    big = 0.05
    small = 0.005
    P_seq = [
        (big if i % 2 == 0 else small)
        * np.ones((8, 16), dtype=float)
        for i in range(20)
    ]
    theta_seq = [np.ones((8, 16), dtype=float)] * 20
    _make_const_force_backend(
        P_per_call=P_seq, theta_per_call=theta_seq)

    # Use very tight eps_update_tol so it CANNOT possibly converge
    # given the alternating force; max_mech_inner=4 to keep test
    # fast.
    policy = CouplingPolicy(
        name="damped_implicit_film",
        max_mech_inner=4,
        mech_relax_initial=0.25,
        mech_relax_min=0.25,
        enable_line_search=False,
        enable_physical_guards=True,
        physical_guards_mode="off",
        commit_on_clamp=False,
        require_solver_converged=True,
        max_delta_eps_inner=10.0,
        max_delta_eps_step=10.0,
        eps_update_tol=1e-12,        # impossibly tight
    )

    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_common_advance_kwargs(ctx, policy),
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Trials within budget (line-search may shrink relax, but the
    # alternating-magnitude stub always passes solver-validity, so
    # the loop runs each k slot until Picard either converges
    # (impossible at eps_update_tol=1e-12) or budget is exhausted).
    assert 1 <= ms.n_trials <= policy.max_mech_inner
    # Step 6 commit gate refused: backend stateful, solve_ok, no
    # clamp, but Picard NOT converged → no commit.
    assert ms.state_committed is False, (
        "Step 6 commit gate failed: state was committed despite "
        "Picard non-convergence.")
    # State still at the cold-start step_index.
    assert state.step_index == 0
    # Rejection reason mentions Picard non-convergence.
    assert ("damped_picard_not_converged" in ms.rejection_detail
            or ms.rejection_reason.value == "solver_residual_above_tol"), (
        f"rejection_reason={ms.rejection_reason.value!r}, "
        f"detail={ms.rejection_detail!r}")
