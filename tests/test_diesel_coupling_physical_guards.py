"""Stage J followup-2 Step 8 — physical guards tests.

Direct unit tests for ``check_physical_guards`` (doc 1 §3.2 / §4.2)
plus kernel-integration tests for the three modes (off / diagnostic
/ hard) inside ``_run_damped_implicit_film``.

Compositional rejection priority (doc 1 §4.2):
1. Pressure cap          → PHYSICAL_PRESSURE_GPA
2. Force-ratio cap       → PHYSICAL_FORCE_RATIO
3. Same-direction        → PHYSICAL_SAME_DIR_RUNAWAY
4. Cavitation runaway    → PHYSICAL_CAV_RUNAWAY
"""
from __future__ import annotations

import warnings

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
    check_physical_guards,
)
from models.diesel_coupling.guards import (
    P_DIM_HARD_MAX_PA,
    P_DIM_SMOKE_MAX_PA,
    F_RATIO_HARD_MAX,
    F_RATIO_SMOKE_MAX,
)
from models.diesel_ausas_adapter import (
    DieselAusasState,
    set_ausas_backend_for_tests,
)


def _good_inputs(*, p_nd_max=1e-2, F_hyd_x=1e3, F_hyd_y=-2e3,
                  F_ext_x=0.0, F_ext_y=-200e3, theta=None):
    """A nominally-physical pressure / force pair for a smooth-
    bearing static-like trial."""
    P = np.full((4, 8), p_nd_max, dtype=float)
    th = (np.full((4, 8), 1.0, dtype=float)
          if theta is None else theta)
    return dict(
        P_nd=P, p_scale=8.3e6,
        Fx_hyd=F_hyd_x, Fy_hyd=F_hyd_y,
        Fx_ext=F_ext_x, Fy_ext=F_ext_y,
        theta=th,
        phi_deg=10.0,
    )


# ─── 1-8. Direct unit tests on check_physical_guards ──────────────


def test_physical_guards_off_always_accepts():
    cfg = PhysicalGuardsConfig.from_profile("off", "general")
    out = check_physical_guards(
        cfg=cfg,
        # Pass values that WOULD trigger rejection in hard mode.
        P_nd=np.full((4, 8), 1e6, dtype=float),  # huge P_nd
        p_scale=1e6,
        Fx_hyd=1e10, Fy_hyd=0.0,                 # huge force
        Fx_ext=0.0, Fy_ext=-1e3,
        theta=np.zeros((4, 8), dtype=float),     # full cavitation
        phi_deg=10.0,
    )
    assert out.accept is True
    assert out.reason is RejectionReason.NONE
    assert out.detail == "off"


def test_physical_guards_accepts_in_range_inputs():
    cfg = PhysicalGuardsConfig.from_profile("hard", "general")
    out = check_physical_guards(cfg=cfg, **_good_inputs())
    assert out.accept is True


def test_physical_guards_rejects_p_dim_above_general_cap():
    cfg = PhysicalGuardsConfig.from_profile("hard", "general")
    # General cap = 1 GPa. Set p_dim_max = 2 GPa.
    P = np.full((4, 8), 2.0e9 / 8.3e6, dtype=float)
    out = check_physical_guards(
        cfg=cfg, P_nd=P, p_scale=8.3e6,
        Fx_hyd=1e3, Fy_hyd=0.0,
        Fx_ext=0.0, Fy_ext=-1e3,
        theta=np.ones((4, 8)), phi_deg=10.0,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.PHYSICAL_PRESSURE_GPA
    assert "2.000e+09" in out.detail or "2.0e+09" in out.detail.lower() \
        or "max(P_dim)" in out.detail


def test_physical_guards_smoke_profile_uses_tighter_cap():
    """Smoke profile caps at 200 MPa on the early-cycle window;
    a 500 MPa pressure is rejected in smoke but accepted in
    general."""
    P = np.full((4, 8), 5.0e8 / 8.3e6, dtype=float)
    base = dict(
        P_nd=P, p_scale=8.3e6,
        Fx_hyd=1e3, Fy_hyd=0.0,
        Fx_ext=0.0, Fy_ext=-1e3,
        theta=np.ones((4, 8)),
        phi_deg=10.0,    # < early_phi_deg=120
    )
    cfg_smoke = PhysicalGuardsConfig.from_profile("hard", "smoke")
    out_smoke = check_physical_guards(cfg=cfg_smoke, **base)
    assert out_smoke.accept is False
    assert out_smoke.reason is RejectionReason.PHYSICAL_PRESSURE_GPA
    cfg_general = PhysicalGuardsConfig.from_profile("hard", "general")
    out_general = check_physical_guards(cfg=cfg_general, **base)
    assert out_general.accept is True


def test_physical_guards_force_ratio_above_cap():
    cfg = PhysicalGuardsConfig.from_profile("hard", "smoke")
    # F_ratio cap = 25 in smoke. Set |F_hyd|=300 kN, |F_ext|=10 kN
    # → ratio=30.
    out = check_physical_guards(
        cfg=cfg,
        P_nd=np.full((4, 8), 1e-2),   # P-cap inactive
        p_scale=8.3e6,
        Fx_hyd=300e3, Fy_hyd=0.0,
        Fx_ext=10e3, Fy_ext=0.0,
        theta=np.ones((4, 8)),
        phi_deg=10.0,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.PHYSICAL_FORCE_RATIO


def test_physical_guards_same_direction_runaway():
    """``dot_norm > 0.9`` AND ``F_ratio > 10`` simultaneously
    triggers SAME_DIR_RUNAWAY (the "film blew up in the SAME
    direction as the load" pathology). Either alone does not."""
    cfg = PhysicalGuardsConfig.from_profile("hard", "general")
    # F_hyd points along F_ext (both +x), ratio = 15 → reject.
    out = check_physical_guards(
        cfg=cfg,
        P_nd=np.full((4, 8), 1e-2),
        p_scale=8.3e6,
        Fx_hyd=15e3, Fy_hyd=0.1e3,    # nearly aligned with F_ext
        Fx_ext=1e3, Fy_ext=0.0,
        theta=np.ones((4, 8)),
        phi_deg=10.0,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.PHYSICAL_SAME_DIR_RUNAWAY


def test_physical_guards_same_direction_safe_when_low_ratio():
    """Same direction but small ratio (~1) — should NOT trigger
    runaway (ε likely small enough that hydro just balances load)."""
    cfg = PhysicalGuardsConfig.from_profile("hard", "general")
    out = check_physical_guards(
        cfg=cfg,
        P_nd=np.full((4, 8), 1e-2),
        p_scale=8.3e6,
        Fx_hyd=1.2e3, Fy_hyd=0.0,    # ratio = 1.2, dot_norm = +1
        Fx_ext=1e3, Fy_ext=0.0,
        theta=np.ones((4, 8)),
        phi_deg=10.0,
    )
    # dot_norm=+1 alone (high) without F_ratio>10 must NOT reject;
    # ratio=1.2 is within F_ratio_max=100. Resisting-force check
    # is a separate diagnostic (Step 9+ may add).
    assert out.accept is True


def test_physical_guards_cav_frac_runaway():
    cfg = PhysicalGuardsConfig.from_profile("hard", "general")
    # 99% of cells cavitating → reject (cav_frac_runaway = 0.98).
    theta = np.ones((10, 10), dtype=float)
    theta[:, :].fill(0.5)        # all cells cavitating
    theta[0, 0] = 1.0           # one cell still full-film, frac = 0.99
    out = check_physical_guards(
        cfg=cfg,
        P_nd=np.full((10, 10), 1e-2),
        p_scale=8.3e6,
        Fx_hyd=1e3, Fy_hyd=0.0,
        Fx_ext=0.0, Fy_ext=-1e3,
        theta=theta,
        phi_deg=10.0,
    )
    assert out.accept is False
    assert out.reason is RejectionReason.PHYSICAL_CAV_RUNAWAY


# ─── 9-11. Kernel integration with mode dispatch ──────────────────


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


def _huge_pressure_stub(cap_pa: float = 2.0e9):
    """Stub that returns a converged-looking solve with P_dim above
    the supplied cap (default 2 GPa)."""
    p_nd = cap_pa / 8.3e6
    P_const = np.full((8, 16), p_nd, dtype=float)
    theta_const = np.ones((8, 16), dtype=float)

    def fake(**kwargs):
        return (P_const, theta_const, 1e-9, 100)

    set_ausas_backend_for_tests(fake)


def _kernel_kwargs(ctx, policy, guards_profile="general", mode="hard"):
    return dict(
        ex_n=0.0, ey_n=-0.6 * ctx.c, vx_n=0.0, vy_n=0.0,
        ax_prev=0.0, ay_prev=0.0, dt_phys_s=1e-4,
        policy=policy,
        guards_cfg=PhysicalGuardsConfig.from_profile(
            mode, guards_profile),
        ausas_tol=1e-6, ausas_max_inner=5000,
        extra_options=None,
        context=ctx, m_shaft=10.0, eps_max=0.95,
        clamp_fn=lambda ex, ey, vx, vy: (ex, ey, vx, vy, False),
        build_H_fn=lambda ex, ey: np.ones((8, 16)),
        p_warm_init=None,
    )


def test_damped_kernel_hard_mode_breaks_on_pressure_cap():
    """Hard mode + p_dim above general 1 GPa cap → kernel breaks
    on first trial with PHYSICAL_PRESSURE_GPA."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(np.ones((8, 16)))
    backend = AusasDynamicBackend(ausas_options=None)
    _huge_pressure_stub(cap_pa=2.0e9)

    try:
        ms = advance_mechanical_step(
            backend=backend, backend_state=state,
            **_kernel_kwargs(ctx, POLICY_AUSAS_DYNAMIC,
                              guards_profile="general", mode="hard"),
        )
    finally:
        set_ausas_backend_for_tests(None)

    assert ms.n_trials == 1
    assert ms.state_committed is False
    assert ms.rejection_reason is RejectionReason.PHYSICAL_PRESSURE_GPA
    # Solver-validity passed (n_inner=100 < 5000, residual=1e-9 < 1e-6),
    # so the rejection came from physical guards.
    tr = ms.trial_log[0]
    assert tr.outcome_solver.accept is True
    assert tr.outcome_physical.accept is False
    assert tr.outcome_physical.reason is RejectionReason.PHYSICAL_PRESSURE_GPA


def test_damped_kernel_diagnostic_mode_warns_but_does_not_break():
    """Diagnostic mode + p_dim above cap → warning emitted, trial
    accepted, Picard loop continues (until other gate or
    convergence ends it)."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(np.ones((8, 16)))
    backend = AusasDynamicBackend(ausas_options=None)
    _huge_pressure_stub(cap_pa=2.0e9)

    with warnings.catch_warnings(record=True) as w_record:
        warnings.simplefilter("always")
        try:
            ms = advance_mechanical_step(
                backend=backend, backend_state=state,
                **_kernel_kwargs(ctx, POLICY_AUSAS_DYNAMIC,
                                  guards_profile="general",
                                  mode="diagnostic"),
            )
        finally:
            set_ausas_backend_for_tests(None)
    # At least one trial must have run; physical_guards verdict
    # is logged on the trial but does NOT force a break.
    assert ms.n_trials >= 1
    # The first trial's physical outcome was a rejection verdict,
    # but ``accepted=True`` because diagnostic mode doesn't gate.
    tr0 = ms.trial_log[0]
    assert tr0.outcome_physical.accept is False
    assert tr0.outcome_physical.reason is RejectionReason.PHYSICAL_PRESSURE_GPA
    assert tr0.accepted is True
    # And a RuntimeWarning was emitted with the right detail.
    msgs = [str(x.message) for x in w_record]
    assert any("physical guard (diagnostic mode)" in m
               and "physical_pressure_above_dim_max" in m
               for m in msgs), (
        f"expected diagnostic warning, got: {msgs}")


def test_damped_kernel_off_mode_silent():
    """Off mode → no checks, no rejection from physical guards;
    the trial is accepted on solver-validity alone."""
    set_ausas_backend_for_tests(None)
    ctx = _make_step_ctx()
    state = DieselAusasState.from_initial_gap(np.ones((8, 16)))
    backend = AusasDynamicBackend(ausas_options=None)
    _huge_pressure_stub(cap_pa=2.0e9)

    # Build a custom policy with enable_physical_guards=False AND
    # mode="off" to be doubly explicit.
    policy = CouplingPolicy(
        name="damped_implicit_film",
        max_mech_inner=1,                  # halt after one step
        mech_relax_initial=0.25,
        mech_relax_min=0.25,
        enable_line_search=False,
        enable_physical_guards=False,
        physical_guards_mode="off",
        commit_on_clamp=False,
        require_solver_converged=True,
        max_delta_eps_inner=10.0,
        max_delta_eps_step=10.0,
        eps_update_tol=10.0,
    )

    with warnings.catch_warnings(record=True) as w_record:
        warnings.simplefilter("always")
        try:
            ms = advance_mechanical_step(
                backend=backend, backend_state=state,
                **_kernel_kwargs(ctx, policy,
                                  guards_profile="general",
                                  mode="off"),
            )
        finally:
            set_ausas_backend_for_tests(None)
    # No physical-guard warning emitted in off mode.
    msgs = [str(x.message) for x in w_record]
    assert not any("physical guard" in m for m in msgs), (
        f"off mode should be silent, got: {msgs}")
    # First trial's outcome_physical reports "off" / accept=True.
    tr = ms.trial_log[0]
    assert tr.outcome_physical.accept is True
    # Either committed (Picard converged trivially with eps_tol=10)
    # or not committed (max_mech_inner=1 didn't satisfy fixed-point);
    # either way no rejection.
    assert ms.rejection_reason in (
        RejectionReason.NONE,
        RejectionReason.SOLVER_RESIDUAL,   # damped_picard_not_converged
    )


def test_legacy_path_does_not_invoke_physical_guards():
    """Step 8 wires physical guards into the damped path ONLY.
    Patch ``models.diesel_coupling.kernel.check_physical_guards``
    to count calls; assert zero invocations on POLICY_LEGACY_HS.
    Same pattern as the Step 7 legacy-path-not-touched test."""
    from models.diesel_coupling import kernel as _k
    calls = {"n": 0}
    real = _k.check_physical_guards

    def watch(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _k.check_physical_guards = watch
    set_ausas_backend_for_tests(None)

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
            backend=_StubHS(), backend_state=None,
            **_kernel_kwargs(ctx, POLICY_LEGACY_HS,
                              guards_profile="general",
                              mode="diagnostic"),
        )
    finally:
        _k.check_physical_guards = real
        set_ausas_backend_for_tests(None)

    assert calls["n"] == 0, (
        f"check_physical_guards was called {calls['n']} times on "
        "the legacy_verlet path — Gate 1 invariance regression. "
        "Step 8 gate must be wired into _run_damped_implicit_film "
        "ONLY.")
