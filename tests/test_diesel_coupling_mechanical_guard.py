"""Stage J followup-2 Step 9 — mechanical-candidate guard tests.

Direct unit tests for ``check_mechanical_candidate`` (doc 1 §3.3 /
§4.3) plus kernel-integration tests for the line-search shrink loop
inside ``_run_damped_implicit_film``.

Compositional rejection priority (doc 1 §4.3):
    1. ``|Δε_inner|`` cap → MECHANICAL_DELTA_EPS_INNER
    2. ``|Δε_step|``  cap → MECHANICAL_DELTA_EPS_STEP
    3. ``|ε| < eps_max``  → MECHANICAL_EPS_MAX

Kernel-side contract (per user followup-2 §3.4):
    * mechanical guard runs BEFORE the GPU call; a rejected
      candidate is never pushed to the backend (saves wasted GPU);
    * on reject, ``mech_relax`` halves and the candidate retreats
      toward ``anchor`` (last accepted candidate);
    * line-search exhausts when ``mech_relax < policy.mech_relax_min``;
    * the surfaced ``rejection_reason`` is the LAST guard that
      rejected — NOT a generic ``MECHANICAL_RELAX_EXHAUSTED``
      bucket (preserves diagnostic value for the Stage 10 summary).
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_coupling import (
    AusasDynamicBackend,
    CouplingPolicy,
    POLICY_AUSAS_DYNAMIC,
    POLICY_LEGACY_HS,
    GuardOutcome,
    PhysicalGuardsConfig,
    PressureSolveResult,
    RejectionReason,
    StepContext,
    advance_mechanical_step,
    check_mechanical_candidate,
)
from models.diesel_ausas_adapter import (
    DieselAusasState,
    set_ausas_backend_for_tests,
)


# ─── 1-7. Direct unit tests on check_mechanical_candidate ──────────


def test_mech_guard_accepts_in_range_candidate():
    """Small Δε, well below eps_max → accept."""
    out = check_mechanical_candidate(
        eps_x_curr=0.10, eps_y_curr=0.20,
        eps_x_cand=0.12, eps_y_cand=0.22,        # Δε_inner ~ 0.028
        eps_x_step_start=0.05, eps_y_step_start=0.15,  # Δε_step ~ 0.099
        eps_max=0.95,
        max_delta_eps_inner=0.10,                # 0.028 < 0.10 — OK
        max_delta_eps_step=0.25,                 # 0.099 < 0.25 — OK
    )
    assert out.accept is True
    assert out.reason is RejectionReason.NONE


def test_mech_guard_rejects_delta_eps_inner_exceeded():
    """``|Δε_inner| > max_delta_eps_inner`` → INNER reason."""
    out = check_mechanical_candidate(
        eps_x_curr=0.10, eps_y_curr=0.10,
        eps_x_cand=0.30, eps_y_cand=0.10,        # Δε_inner = 0.20
        eps_x_step_start=0.10, eps_y_step_start=0.10,
        eps_max=0.95,
        max_delta_eps_inner=0.10,                # 0.20 > 0.10
        max_delta_eps_step=0.25,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.MECHANICAL_DELTA_EPS_INNER
    assert "Δε_inner" in out.detail or "delta_eps_inner" in out.detail.lower() \
        or "0.20" in out.detail or "0.2000" in out.detail


def test_mech_guard_rejects_delta_eps_step_exceeded():
    """``|Δε_step| > max_delta_eps_step`` (but inner OK) → STEP reason."""
    # Δε_inner small (0.05) — passes inner check.
    # Δε_step = hypot(0.30, 0.0) = 0.30 — fails step check (0.25).
    out = check_mechanical_candidate(
        eps_x_curr=0.55, eps_y_curr=0.10,
        eps_x_cand=0.60, eps_y_cand=0.10,        # Δε_inner = 0.05
        eps_x_step_start=0.30, eps_y_step_start=0.10,  # Δε_step = 0.30
        eps_max=0.95,
        max_delta_eps_inner=0.10,
        max_delta_eps_step=0.25,                 # 0.30 > 0.25
    )
    assert out.accept is False
    assert out.reason is RejectionReason.MECHANICAL_DELTA_EPS_STEP


def test_mech_guard_rejects_eps_at_clamp():
    """``|ε| >= eps_max`` (even with both Δ caps OK) → EPS_MAX."""
    # ε_cand = (0.95, 0.0) → |ε| = 0.95 = eps_max → reject.
    out = check_mechanical_candidate(
        eps_x_curr=0.94, eps_y_curr=0.00,
        eps_x_cand=0.95, eps_y_cand=0.00,        # |ε| = eps_max
        eps_x_step_start=0.94, eps_y_step_start=0.00,
        eps_max=0.95,
        max_delta_eps_inner=0.10,
        max_delta_eps_step=0.25,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.MECHANICAL_EPS_MAX


def test_mech_guard_priority_inner_beats_step_and_eps_max():
    """All three checks would fail; the first (inner) wins."""
    out = check_mechanical_candidate(
        eps_x_curr=0.10, eps_y_curr=0.10,
        eps_x_cand=0.99, eps_y_cand=0.10,        # Δε_inner=0.89,
                                                  # Δε_step=0.89,
                                                  # |ε|=0.99
        eps_x_step_start=0.10, eps_y_step_start=0.10,
        eps_max=0.95,
        max_delta_eps_inner=0.10,
        max_delta_eps_step=0.25,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.MECHANICAL_DELTA_EPS_INNER


def test_mech_guard_priority_step_beats_eps_max():
    """When inner is None or passes, step wins over eps_max."""
    # Δε_inner=0.08 (would-pass), Δε_step=0.30 (fail), |ε|=0.99 (also fail).
    out = check_mechanical_candidate(
        eps_x_curr=0.91, eps_y_curr=0.00,
        eps_x_cand=0.99, eps_y_cand=0.00,
        eps_x_step_start=0.69, eps_y_step_start=0.00,  # Δε_step=0.30
        eps_max=0.95,
        max_delta_eps_inner=0.10,
        max_delta_eps_step=0.25,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.MECHANICAL_DELTA_EPS_STEP


def test_mech_guard_skips_delta_bounds_when_none():
    """Both Δε bounds None → only ``eps_max`` is enforced (this is
    the LEGACY_HS configuration: no per-trial / per-step caps)."""
    # Huge Δ in both directions but |ε| stays well under eps_max.
    out = check_mechanical_candidate(
        eps_x_curr=0.00, eps_y_curr=0.00,
        eps_x_cand=0.50, eps_y_cand=0.50,        # Δε_inner ≈ 0.71
        eps_x_step_start=0.00, eps_y_step_start=0.00,
        eps_max=0.95,
        max_delta_eps_inner=None,
        max_delta_eps_step=None,
    )
    assert out.accept is True
    assert out.reason is RejectionReason.NONE
    # But eps_max is still enforced even without Δ bounds.
    out_clamp = check_mechanical_candidate(
        eps_x_curr=0.00, eps_y_curr=0.00,
        eps_x_cand=0.95, eps_y_cand=0.00,        # |ε| = eps_max
        eps_x_step_start=0.00, eps_y_step_start=0.00,
        eps_max=0.95,
        max_delta_eps_inner=None,
        max_delta_eps_step=None,
    )
    assert out_clamp.accept is False
    assert out_clamp.reason is RejectionReason.MECHANICAL_EPS_MAX


def test_mech_guard_step_bound_alone_when_inner_is_none():
    """Inner=None still enforces the step bound."""
    out = check_mechanical_candidate(
        eps_x_curr=0.50, eps_y_curr=0.00,
        eps_x_cand=0.55, eps_y_cand=0.00,        # Δε_inner=0.05
        eps_x_step_start=0.20, eps_y_step_start=0.00,  # Δε_step=0.35
        eps_max=0.95,
        max_delta_eps_inner=None,                # skipped
        max_delta_eps_step=0.25,                 # 0.35 > 0.25 → reject
    )
    assert out.accept is False
    assert out.reason is RejectionReason.MECHANICAL_DELTA_EPS_STEP


# ─── 9-12. Kernel integration with line-search loop ───────────────


def _make_step_ctx(N_phi: int = 16, N_z: int = 8) -> StepContext:
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    Z_1D = np.linspace(-1.0, 1.0, N_z)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    return StepContext(
        phi_deg=10.0, F_ext_x=0.0, F_ext_y=-1.0e3, F_max=1.0e3,
        p_scale=8.3e6, omega=200.0, eta=0.01,
        R=0.1, L=0.08, c=1.2e-4,
        Phi_mesh=Phi_mesh, Z_mesh=Z_mesh,
        phi_1D=phi_1D, Z_1D=Z_1D,
        d_phi=float(phi_1D[1] - phi_1D[0]),
        d_Z=float(Z_1D[1] - Z_1D[0]),
        cfg={"label": "test", "textured": False},
        texture_kind="none",
        groove_relief=None,
        phi_c_flat=None, Z_c_flat=None,
        closure="laminar", cavitation="ausas_dynamic",
        P_warm_init=None,
    )


def _kernel_kwargs(ctx, policy, *, ex_n=0.0, ey_n=None,
                    vx_n=0.0, vy_n=0.0, mode="hard",
                    profile="general"):
    if ey_n is None:
        ey_n = -0.6 * ctx.c
    return dict(
        ex_n=ex_n, ey_n=ey_n, vx_n=vx_n, vy_n=vy_n,
        ax_prev=0.0, ay_prev=0.0, dt_phys_s=1e-4,
        policy=policy,
        guards_cfg=PhysicalGuardsConfig.from_profile(mode, profile),
        ausas_tol=1e-6, ausas_max_inner=5000,
        extra_options=None,
        context=ctx, m_shaft=10.0, eps_max=0.95,
        clamp_fn=lambda ex, ey, vx, vy: (ex, ey, vx, vy, False),
        build_H_fn=lambda ex, ey: np.ones((8, 16)),
        p_warm_init=None,
    )


def test_damped_kernel_mech_guard_skips_gpu_on_bad_candidate():
    """A candidate that violates the mechanical guard must NOT
    reach the GPU backend. Patch the backend's ``solve_trial`` to
    count calls; assert it is called STRICTLY fewer times than the
    Picard inner budget when the guard rejects every candidate."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()

    # Policy with very tight Δε_inner cap so the predictor's first
    # candidate is rejected on iteration 0.
    policy = CouplingPolicy(
        name="damped_implicit_film",
        max_mech_inner=8,
        mech_relax_initial=1.0,
        mech_relax_min=0.5,                       # exhausts after 1 shrink
        enable_line_search=True,
        enable_physical_guards=True,
        physical_guards_mode="hard",
        commit_on_clamp=False,
        require_solver_converged=True,
        max_delta_eps_inner=1e-9,                 # any motion fails
        max_delta_eps_step=1e-9,
        eps_update_tol=1e-4,
    )

    # Set initial state so the predictor's Verlet step lands at a
    # different point than ex_n / ey_n (predictor_clamped=False but
    # |Δε| large enough to fail the guard).
    state = DieselAusasState.from_initial_gap(np.ones((8, 16)))
    backend = AusasDynamicBackend(ausas_options=None)

    # Sanity stub: would converge if called, returns small finite P
    # on the padded ``(N_z, N_phi+2)`` boundary that the adapter
    # now uses (Stage J fu-2 ghost-grid migration).
    n_calls = {"k": 0}

    def fake(**kwargs):
        H_pad = np.asarray(kwargs["H_curr"])
        n_calls["k"] += 1
        P_const = 0.01 * np.ones(H_pad.shape, dtype=float)
        theta_const = np.ones(H_pad.shape, dtype=float)
        return (P_const, theta_const, 1e-9, 100)

    set_ausas_backend_for_tests(fake)
    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_kernel_kwargs(
                ctx, policy,
                # Pick a v large enough to make the predictor a
                # finite jump from ex_n: vx*dt_step = 1e-4*1e-4=1e-8
                # — that's nowhere near 1e-9 cap, so v=10 gives
                # vx*dt = 1e-3 m → ε change = 1e-3/c = 8.3, far
                # exceeds the 1e-9 cap.
                vx_n=10.0, vy_n=10.0,
            ),
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Mechanical guard rejected on iter 0 → no GPU call.
    assert n_calls["k"] == 0, (
        f"GPU backend was called {n_calls['k']} times despite the "
        "mechanical guard rejecting the candidate. Step 9 contract: "
        "mech guard runs BEFORE the backend call.")
    assert ms.state_committed is False
    # Rejection reason MUST be the mechanical bucket (not generic
    # SOLVER_RESIDUAL) — tight inner cap means INNER fires.
    assert ms.rejection_reason is RejectionReason.MECHANICAL_DELTA_EPS_INNER
    # Trial 0 records the skipped trial with no real pressure result.
    assert len(ms.trial_log) >= 1
    tr0 = ms.trial_log[0]
    assert tr0.outcome_mechanical.accept is False
    assert (tr0.outcome_mechanical.reason
            is RejectionReason.MECHANICAL_DELTA_EPS_INNER)
    assert tr0.accepted is False


def test_damped_kernel_line_search_shrinks_relax_on_physical_reject():
    """A converged-but-physically-bad solver result drives the
    line-search loop: ``relax`` halves on each rejection, and the
    last-rejection reason matches the physical-guard verdict.

    To exercise the SHRINK path (rather than the Step 9 fixup's
    reject-at-anchor abort), the test sets non-zero ``vx_n``,
    ``vy_n`` so the Verlet predictor produces ``ex_pred != ex_n``;
    that puts the candidate strictly off the anchor, and a
    rejecting trial then drives the relax shrink retreat toward
    anchor on each retry.
    """
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(np.ones((8, 16)))
    backend = AusasDynamicBackend(ausas_options=None)

    # Converged-looking solve with P_dim = 2 GPa (above 1 GPa cap).
    # Output shape matches the padded ``(N_z, N_phi+2)`` boundary
    # the adapter uses post Stage J ghost-grid migration.
    p_nd_huge = 2.0e9 / ctx.p_scale

    def fake(**kwargs):
        H_pad = np.asarray(kwargs["H_curr"])
        P_const = np.full(H_pad.shape, p_nd_huge, dtype=float)
        theta_const = np.ones(H_pad.shape, dtype=float)
        return (P_const, theta_const, 1e-9, 100)

    set_ausas_backend_for_tests(fake)
    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_kernel_kwargs(ctx, POLICY_AUSAS_DYNAMIC,
                              # Non-zero starting velocity so the
                              # Verlet predictor moves the candidate
                              # OFF the anchor by ~vx*dt = 1e-7 m
                              # ≈ 8.3e-4 in ε units (above the 1e-12
                              # reject-at-anchor floor; well below
                              # the 0.10 max_delta_eps_inner cap).
                              ex_n=0.0, ey_n=-0.6 * ctx.c,
                              vx_n=1e-3, vy_n=1e-3,
                              mode="hard", profile="general"),
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Line-search shrunk relax at least once (cand was off-anchor).
    assert ms.mech_relax_min_seen < float(
        POLICY_AUSAS_DYNAMIC.mech_relax_initial), (
        f"line-search did not shrink: min_seen="
        f"{ms.mech_relax_min_seen}, initial="
        f"{POLICY_AUSAS_DYNAMIC.mech_relax_initial}")
    # Final rejection_reason = the last gate that rejected. Since
    # solver-validity passes and physical-guards rejects every
    # retry with PHYSICAL_PRESSURE_GPA, that's the surfaced reason.
    assert ms.state_committed is False
    assert ms.rejection_reason is RejectionReason.PHYSICAL_PRESSURE_GPA
    # The detail string includes the line-search-exhausted tag
    # (or, on borderline retreat, reject-at-anchor). Either path
    # surfaces the physical-guard reason.
    assert (
        "line_search_exhausted" in ms.rejection_detail
        or "reject_at_anchor" in ms.rejection_detail), (
        f"unexpected detail: {ms.rejection_detail!r}")
    assert "physical_pressure_above_dim_max" in ms.rejection_detail


def test_damped_kernel_no_shrink_when_all_trials_accept():
    """When every guard accepts, the line-search never shrinks
    ``mech_relax``. Verifies the (Step 9) contract that a
    successful trial RESETS ``relax`` to ``mech_relax_initial`` for
    the next iteration — there is no compounding shrink across
    accepts. We don't pin Picard convergence here (orthogonal:
    a non-static state may take many Picard iterations to settle);
    we pin only that ``mech_relax_min_seen == mech_relax_initial``
    and that no trial recorded a guard rejection."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(np.ones((8, 16)))
    backend = AusasDynamicBackend(ausas_options=None)

    # Tiny p_nd → all gates pass on every trial. Padded shape per
    # Stage J fu-2 ghost-grid migration.
    def fake(**kwargs):
        H_pad = np.asarray(kwargs["H_curr"])
        p_const = 1e-3 * np.ones(H_pad.shape, dtype=float)
        theta_const = np.ones(H_pad.shape, dtype=float)
        return (p_const, theta_const, 1e-9, 100)

    set_ausas_backend_for_tests(fake)
    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_kernel_kwargs(ctx, POLICY_AUSAS_DYNAMIC,
                              mode="hard", profile="general"),
        )
    finally:
        set_ausas_backend_for_tests(None)

    # No guard ever rejected → relax never shrunk.
    assert ms.mech_relax_min_seen == float(
        POLICY_AUSAS_DYNAMIC.mech_relax_initial), (
        f"relax shrunk despite no guard rejections: "
        f"min_seen={ms.mech_relax_min_seen}, initial="
        f"{POLICY_AUSAS_DYNAMIC.mech_relax_initial}")
    for tr in ms.trial_log:
        assert tr.outcome_solver.accept is True
        assert tr.outcome_physical.accept is True
        assert tr.outcome_mechanical.accept is True
        assert tr.accepted is True


def test_legacy_path_does_not_invoke_mechanical_guard():
    """Step 9 wires the mechanical-candidate guard into the damped
    path ONLY. Patch ``models.diesel_coupling.kernel.check_mechanical_candidate``
    to count calls; assert zero invocations on POLICY_LEGACY_HS.
    Same patch-counter pattern as the Step 7 / Step 8 legacy-path-not-
    touched tests — pins the Gate 1 invariance."""
    from models.diesel_coupling import kernel as _k
    calls = {"n": 0}
    real = _k.check_mechanical_candidate

    def watch(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _k.check_mechanical_candidate = watch
    set_ausas_backend_for_tests(None)

    class _StubHS:
        name = "half_sommerfeld"
        stateful = False
        requires_implicit_mech_coupling = False

        def solve_trial(self, **kw):
            from models.diesel_transient import compute_hydro_forces
            H = kw["H_curr"]
            P = 0.5 * np.ones_like(H)
            ctx = kw["context"]
            Fx, Fy = compute_hydro_forces(
                P, ctx.p_scale, ctx.Phi_mesh,
                ctx.phi_1D, ctx.Z_1D, ctx.R, ctx.L)
            return PressureSolveResult(
                P_nd=P, theta=None,
                Fx_hyd=float(Fx), Fy_hyd=float(Fy),
                H_used=H, residual=float("nan"),
                n_inner=10, converged=True,
                backend_name=self.name, reason="ok",
            )

    ctx = _make_step_ctx()
    try:
        advance_mechanical_step(
            backend=_StubHS(), backend_state=None,
            **_kernel_kwargs(ctx, POLICY_LEGACY_HS,
                              mode="diagnostic"),
        )
    finally:
        _k.check_mechanical_candidate = real
        set_ausas_backend_for_tests(None)

    assert calls["n"] == 0, (
        f"check_mechanical_candidate was called {calls['n']} times "
        "on the legacy_verlet path — Gate 1 invariance regression. "
        "Step 9 gate must be wired into _run_damped_implicit_film "
        "ONLY.")
