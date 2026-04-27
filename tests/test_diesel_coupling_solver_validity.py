"""Stage J followup-2 Step 7 — solver-validity hard gate tests.

Direct unit tests for ``check_solver_validity`` (doc 1 §3.1) plus a
kernel-integration test that pins the gate's behaviour inside
``_run_damped_implicit_film``: bad solver outputs trigger an early
break of the Picard loop, with the SPECIFIC rejection reason
threaded into ``MechanicalStepResult.rejection_reason`` for the
Stage 10 summary writer.

The legacy_verlet path is NOT touched by Step 7 — the Gate 1
invariance test in
``tests/test_diesel_legacy_invariance_post_coupling_refactor.py``
(real-GPU box) is the contract that proves it. CPU-side, this
file's structural tests are sufficient.
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
    RejectionReason,
    StepContext,
    advance_mechanical_step,
    check_solver_validity,
)
from models.diesel_ausas_adapter import (
    DieselAusasState,
    set_ausas_backend_for_tests,
)


# ─── 1-7. Direct unit tests on check_solver_validity ───────────────


def _good():
    """Helper: a realistic valid (P, theta) pair."""
    P = 0.5 * np.ones((4, 8), dtype=float)
    theta = np.ones((4, 8), dtype=float)
    return P, theta


def test_solver_validity_accepts_finite_in_range_inputs():
    P, theta = _good()
    out = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta,
        residual=1e-9, n_inner=42, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is True
    assert out.reason is RejectionReason.NONE


def test_solver_validity_rejects_n_inner_at_max_with_BUDGET():
    """``n_inner == max_inner`` MUST never be accepted, even with
    a tiny residual (doc 1 §1.2)."""
    P, theta = _good()
    out = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta,
        residual=1e-12, n_inner=5000, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.SOLVER_BUDGET
    assert "n_inner=5000" in out.detail


def test_solver_validity_rejects_residual_above_tol_with_RESIDUAL():
    P, theta = _good()
    out = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta,
        residual=1e-3, n_inner=200, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.SOLVER_RESIDUAL
    assert "residual=" in out.detail
    assert "tol=" in out.detail


def test_solver_validity_rejects_negative_pressure():
    P = 0.5 * np.ones((4, 8))
    P[2, 3] = -1e-3        # well below P_NEGATIVE_TOL_ND = 1e-12
    theta = np.ones((4, 8))
    out = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta,
        residual=1e-9, n_inner=42, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.SOLVER_NEG_PRESSURE


def test_solver_validity_rejects_theta_out_of_range():
    P, _ = _good()
    theta_bad = np.ones((4, 8))
    theta_bad[1, 1] = 1.5      # > THETA_CEIL = 1 + 1e-10
    out = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta_bad,
        residual=1e-9, n_inner=42, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.SOLVER_THETA_OUT_OF_RANGE
    # Negative side too.
    theta_neg = np.ones((4, 8))
    theta_neg[2, 3] = -0.1     # < THETA_FLOOR = -1e-10
    out2 = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta_neg,
        residual=1e-9, n_inner=42, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out2.accept is False
    assert out2.reason is RejectionReason.SOLVER_THETA_OUT_OF_RANGE


def test_solver_validity_rejects_nonfinite_pressure():
    P, theta = _good()
    P[0, 0] = np.nan
    out = check_solver_validity(
        backend_name="ausas_dynamic",
        P_nd=P, theta=theta,
        residual=1e-9, n_inner=42, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.SOLVER_NONFINITE


def test_solver_validity_skips_theta_for_half_sommerfeld():
    """``theta=None`` (half-Sommerfeld) skips theta checks. The
    NaN residual + converged=True case must be accepted (legacy
    SOR doesn't surface a residual)."""
    P, _ = _good()
    out = check_solver_validity(
        backend_name="half_sommerfeld",
        P_nd=P, theta=None,
        residual=float("nan"), n_inner=120, converged=True,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out.accept is True
    assert out.reason is RejectionReason.NONE
    # NaN residual + converged=False MUST still reject.
    out2 = check_solver_validity(
        backend_name="half_sommerfeld",
        P_nd=P, theta=None,
        residual=float("nan"), n_inner=120, converged=False,
        ausas_tol=1e-6, ausas_max_inner=5000,
    )
    assert out2.accept is False
    assert out2.reason is RejectionReason.SOLVER_NONFINITE


# ─── 8. Kernel-level integration: gate breaks Picard loop ─────────


def _make_step_ctx(N_phi: int = 16, N_z: int = 8) -> StepContext:
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    Z_1D = np.linspace(-1.0, 1.0, N_z)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    return StepContext(
        phi_deg=0.0, F_ext_x=0.0, F_ext_y=-1.0e3, F_max=1.0e3,
        p_scale=1.0e6, omega=200.0, eta=0.01,
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


def test_damped_kernel_breaks_on_n_inner_at_max():
    """Damped Picard: a stub that ALWAYS returns
    ``n_inner == max_inner`` must be rejected by Step 7's gate on
    the first trial; the kernel breaks immediately,
    ``state_committed=False``, ``rejection_reason=SOLVER_BUDGET``,
    and the only TrialRecord carries the right outcome_solver."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(
        np.ones((8, 16), dtype=float))
    backend = AusasDynamicBackend(ausas_options=None)
    P_const = 0.1 * np.ones((8, 16), dtype=float)
    theta_const = np.ones((8, 16), dtype=float)

    def fake(**kwargs):
        # n_inner intentionally HITS max_inner=5000 so
        # check_solver_validity flags SOLVER_BUDGET.
        return (P_const, theta_const, 1e-12, 5000)

    set_ausas_backend_for_tests(fake)
    try:
        ms = advance_mechanical_step(
            ex_n=0.0, ey_n=-0.6 * ctx.c, vx_n=0.0, vy_n=0.0,
            ax_prev=0.0, ay_prev=0.0, dt_phys_s=1e-4,
            backend=backend, backend_state=state,
            policy=POLICY_AUSAS_DYNAMIC,
            guards_cfg=PhysicalGuardsConfig.from_profile(
                "diagnostic", "general"),
            ausas_tol=1e-6, ausas_max_inner=5000,
            extra_options=None,
            context=ctx, m_shaft=10.0, eps_max=0.95,
            clamp_fn=lambda ex, ey, vx, vy: (ex, ey, vx, vy, False),
            build_H_fn=lambda ex, ey: np.ones((8, 16)),
            p_warm_init=None,
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Stage J fu-2 Step 9 — with the test setup (vx_n=vy_n=0,
    # ax_prev=ay_prev=0) the Verlet predictor produces ex_pred==ex_n,
    # ey_pred==ey_n so the FIRST candidate equals the anchor. The
    # Step 9 fixup detects ``|cand - anchor| < 1e-12`` and aborts
    # the line-search immediately (shrinking can never recover when
    # the anchor itself fails the gate); n_trials=1, the surfaced
    # rejection reason is the SAME SOLVER_BUDGET enum.
    assert ms.state_committed is False
    assert ms.rejection_reason is RejectionReason.SOLVER_BUDGET, (
        f"expected SOLVER_BUDGET, got {ms.rejection_reason}")
    assert "n_inner=5000" in ms.rejection_detail
    # solve_reason carries the abort path tag for the summary writer.
    assert ("reject_at_anchor" in ms.solve_reason
            or "line_search_exhausted" in ms.solve_reason), (
        f"solve_reason={ms.solve_reason!r}")
    assert 1 <= ms.n_trials <= POLICY_AUSAS_DYNAMIC.max_mech_inner
    # Every TrialRecord carries the solver outcome verbatim.
    assert len(ms.trial_log) == ms.n_trials
    for tr in ms.trial_log:
        assert tr.accepted is False
        assert tr.outcome_solver.accept is False
        assert tr.outcome_solver.reason is RejectionReason.SOLVER_BUDGET
    # Ausas state never advances on a rejected step.
    assert state.step_index == 0


def test_damped_kernel_breaks_on_negative_pressure():
    """Stub returning a negative P_nd → SOLVER_NEG_PRESSURE
    rejection on first trial."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(
        np.ones((8, 16), dtype=float))
    backend = AusasDynamicBackend(ausas_options=None)
    P_neg = -1.0 * np.ones((8, 16), dtype=float)
    theta_const = np.ones((8, 16), dtype=float)

    def fake(**kwargs):
        return (P_neg, theta_const, 1e-12, 100)

    set_ausas_backend_for_tests(fake)
    try:
        ms = advance_mechanical_step(
            ex_n=0.0, ey_n=-0.6 * ctx.c, vx_n=0.0, vy_n=0.0,
            ax_prev=0.0, ay_prev=0.0, dt_phys_s=1e-4,
            backend=backend, backend_state=state,
            policy=POLICY_AUSAS_DYNAMIC,
            guards_cfg=PhysicalGuardsConfig.from_profile(
                "diagnostic", "general"),
            ausas_tol=1e-6, ausas_max_inner=5000,
            extra_options=None,
            context=ctx, m_shaft=10.0, eps_max=0.95,
            clamp_fn=lambda ex, ey, vx, vy: (ex, ey, vx, vy, False),
            build_H_fn=lambda ex, ey: np.ones((8, 16)),
            p_warm_init=None,
        )
    finally:
        set_ausas_backend_for_tests(None)

    # Step 9 — same invariants as the BUDGET test: with cand=anchor
    # at k=0 the kernel aborts immediately (no shrinking can
    # recover). state not committed, rejection reason matches.
    assert ms.state_committed is False
    assert ms.rejection_reason is RejectionReason.SOLVER_NEG_PRESSURE
    assert state.step_index == 0


def test_legacy_path_does_not_invoke_solver_validity_gate():
    """The Step 7 solver-validity gate is wired into the damped
    path ONLY. The legacy path (HS / POLICY_LEGACY_HS) must never
    call ``check_solver_validity`` — Gate 1 invariance depends on
    this. This test verifies it by patching
    ``models.diesel_coupling.kernel.check_solver_validity`` and
    asserting it is NOT called when policy=POLICY_LEGACY_HS."""
    from models.diesel_coupling import kernel as _k
    calls = {"n": 0}
    real = _k.check_solver_validity

    def watch(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _k.check_solver_validity = watch
    set_ausas_backend_for_tests(None)

    # Build a minimal HS-shaped backend stub that returns a finite
    # pressure (no theta) so the legacy path runs cleanly.
    class _StubHS:
        name = "half_sommerfeld"
        stateful = False
        requires_implicit_mech_coupling = False

        def solve_trial(self, **kw):
            from models.diesel_coupling import PressureSolveResult
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
            ex_n=0.0, ey_n=-0.6 * ctx.c, vx_n=0.0, vy_n=0.0,
            ax_prev=0.0, ay_prev=0.0, dt_phys_s=1e-4,
            backend=_StubHS(), backend_state=None,
            policy=POLICY_LEGACY_HS,
            guards_cfg=PhysicalGuardsConfig.from_profile(
                "diagnostic", "general"),
            ausas_tol=1e-6, ausas_max_inner=5000,
            extra_options=None,
            context=ctx, m_shaft=10.0, eps_max=0.95,
            clamp_fn=lambda ex, ey, vx, vy: (ex, ey, vx, vy, False),
            build_H_fn=lambda ex, ey: np.ones((8, 16)),
            p_warm_init=None,
        )
    finally:
        _k.check_solver_validity = real
        set_ausas_backend_for_tests(None)

    assert calls["n"] == 0, (
        f"check_solver_validity was called {calls['n']} times on "
        "the legacy_verlet path — Gate 1 invariance regression. "
        "Step 7 gate must be wired into _run_damped_implicit_film "
        "ONLY.")
