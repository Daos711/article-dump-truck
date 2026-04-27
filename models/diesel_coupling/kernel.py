"""Stage J followup-2 — backend-agnostic mechanical-step kernel.

One mechanical step (one accepted Verlet update) lives entirely
inside :func:`advance_mechanical_step`. The Verlet OUTER loop
remains in ``run_transient`` — only the body of one step (currently
the ``for k in range(N_SUB)`` corrector + the post-clamp commit)
moves here.

Two policies are dispatched on the same kernel call:

* ``POLICY_LEGACY_HS`` — bit-for-bit equivalent to the existing
  3-pass corrector. No line search, no relax shrinking, guards in
  diagnostic mode (warn-only). Locked in by Gate 1.
* ``POLICY_AUSAS_DYNAMIC`` — damped implicit film coupling. Up
  to ``policy.max_mech_inner`` candidate evaluations; relax shrinks
  on rejection; guards in hard mode; no commit on clamped step.

Selection is by the policy ``name`` field, not by backend name —
the same kernel handles both via different relax / guard /
line-search paths.

Step 2 — skeleton only. Step 3 implements ``LegacyVerletPolicy``
path (bit-for-bit move). Step 6 adds ``DampedImplicitFilmPolicy``
skeleton. Step 7-9 wire the guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from .backends import PressureBackend, PressureSolveResult, StepContext
from .guards import (
    GuardOutcome,
    PhysicalGuardsConfig,
    RejectionReason,
)
from .policies import CouplingPolicy


# ─── Per-trial record (debug + summary) ────────────────────────────


@dataclass
class TrialRecord:
    """One pressure-solve attempt within a single mechanical step.

    Captured for the debug printer (Stage J followup-1
    ``--debug-first-steps N``) and for the per-step npz array
    ``trials_per_step`` (Gate 3 schema).
    """
    inner: int
    relax_used: float
    eps_x_cand: float
    eps_y_cand: float
    pressure_result: PressureSolveResult
    outcome_solver: GuardOutcome
    outcome_physical: GuardOutcome
    outcome_mechanical: GuardOutcome
    accepted: bool


# ─── Kernel return value ───────────────────────────────────────────


@dataclass
class MechanicalStepResult:
    """Outcome of one mechanical step.

    On ``accepted=True``:
        new shaft state ``(eps_x_new, eps_y_new, vx_new, vy_new,
        ax_new, ay_new)`` and committed pressure / theta / film
        thickness for downstream diagnostics.
    On ``accepted=False``:
        the runner treats the step as solver-failed for envelope
        statistics; the mechanical state advances anyway (matches
        the legacy "Accept (even if solver failed — keep mechanical
        state progressing without poisoned pressure)" comment in
        the pre-refactor runner).

    Legacy-accounting passthrough fields preserve runner-side
    bookkeeping bit-for-bit:
        ``n_contact_events``, ``predictor_clamped``,
        ``final_clamped`` feed ``contact_clamp_event_count_all``
        and ``contact_clamp_all`` arrays;
        ``retry_recovered`` / ``omega_recovery`` populate the
        retry-policy summary;
        ``solve_reason`` is the reason string the runner uses for
        solver-failed first-phi tracking;
        ``p_warm_out`` is the SOR warm-start hint to thread into
        the next mechanical step.
    """
    accepted: bool
    eps_x_new: float
    eps_y_new: float
    vx_new: float
    vy_new: float
    ax_new: float
    ay_new: float

    H_committed: Optional[np.ndarray]
    P_nd_committed: Optional[np.ndarray]
    theta_committed: Optional[np.ndarray]
    Fx_hyd_committed: float
    Fy_hyd_committed: float

    residual: float
    n_inner: int
    n_trials: int
    mech_relax_min_seen: float

    rejection_reason: RejectionReason
    rejection_detail: str
    trial_log: List[TrialRecord]

    state_committed: bool
    step_clamped: bool

    # Legacy accounting passthrough — preserves runner bit-for-bit.
    n_contact_events: int = 0
    predictor_clamped: bool = False
    final_clamped: bool = False
    retry_recovered: bool = False
    omega_recovery: Optional[float] = None
    solve_reason: str = "no_attempt"
    p_warm_out: Optional[np.ndarray] = None
    # Stateful-backend (Ausas) per-step diagnostics. Zero / 1.0
    # default values are used by the half-Sommerfeld path so the
    # runner's per-step diagnostic arrays for both backends stay on
    # a single dataclass schema.
    cav_frac_committed: float = 0.0
    theta_min_committed: float = 1.0
    theta_max_committed: float = 1.0
    dt_ausas_committed: float = 0.0


# ─── Public kernel entry point ─────────────────────────────────────


def advance_mechanical_step(
    *,
    # Mechanical state at start of step
    ex_n: float,
    ey_n: float,
    vx_n: float,
    vy_n: float,
    ax_prev: float,
    ay_prev: float,
    # Step driving
    dt_phys_s: float,
    # Backend / policy / guards
    backend: PressureBackend,
    backend_state: Optional[Any],
    policy: CouplingPolicy,
    guards_cfg: PhysicalGuardsConfig,
    # Solver knobs (forwarded to the backend's ``extra_options``)
    ausas_tol: float,
    ausas_max_inner: int,
    extra_options: Optional[dict],
    # Geometric / textural / kinematic context
    context: StepContext,
    # Mechanical scaffolding owned by the runner
    m_shaft: float,
    eps_max: float,
    clamp_fn: Callable[
        [float, float, float, float],
        Any,
    ],
    build_H_fn: Callable[
        [float, float],
        np.ndarray,
    ],
    # Legacy SOR warm-start hint threaded across steps by the runner.
    p_warm_init: Optional[np.ndarray] = None,
) -> MechanicalStepResult:
    """One mechanical step.

    Dispatches on ``policy.name``:

    ``legacy_verlet``
        Bit-for-bit move of the pre-refactor
        ``for k in range(N_SUB):`` corrector body in
        ``run_transient`` plus the post-clamp commit. Op order is
        preserved (predictor clamp → for-k {build H → backend
        trial → forces → Verlet update → substep clamp} →
        accept → final clamp). Locked by Gate 1.

    ``damped_implicit_film``
        Damped implicit film coupling (Step 6+ — body still
        ``...`` here, raises NotImplementedError until Step 6).
    """
    if policy.name == "legacy_verlet":
        return _run_legacy_verlet(
            ex_n=ex_n, ey_n=ey_n, vx_n=vx_n, vy_n=vy_n,
            ax_prev=ax_prev, ay_prev=ay_prev,
            dt_phys_s=dt_phys_s,
            backend=backend,
            backend_state=backend_state,
            policy=policy,
            extra_options=extra_options,
            context=context,
            m_shaft=m_shaft,
            eps_max=eps_max,
            clamp_fn=clamp_fn,
            build_H_fn=build_H_fn,
            p_warm_init=p_warm_init,
        )
    if policy.name == "damped_implicit_film":
        return _run_damped_implicit_film(
            ex_n=ex_n, ey_n=ey_n, vx_n=vx_n, vy_n=vy_n,
            ax_prev=ax_prev, ay_prev=ay_prev,
            dt_phys_s=dt_phys_s,
            backend=backend,
            backend_state=backend_state,
            policy=policy,
            extra_options=extra_options,
            context=context,
            m_shaft=m_shaft,
            eps_max=eps_max,
            clamp_fn=clamp_fn,
            build_H_fn=build_H_fn,
            p_warm_init=p_warm_init,
        )
    raise ValueError(
        f"advance_mechanical_step: unknown policy {policy.name!r}; "
        f"expected 'legacy_verlet' or 'damped_implicit_film'.")


def _run_legacy_verlet(
    *,
    ex_n: float, ey_n: float, vx_n: float, vy_n: float,
    ax_prev: float, ay_prev: float,
    dt_phys_s: float,
    backend: PressureBackend,
    backend_state: Optional[Any],
    policy: CouplingPolicy,
    extra_options: Optional[dict],
    context: StepContext,
    m_shaft: float,
    eps_max: float,
    clamp_fn,
    build_H_fn,
    p_warm_init: Optional[np.ndarray],
) -> MechanicalStepResult:
    """Bit-for-bit move of the legacy ``for k in range(N_SUB):``
    corrector body + post-clamp commit. No guards beyond what
    ``_solve_dynamic_with_retry`` itself does."""
    # Local imports avoid a top-level cycle through the runner.
    from models.diesel_quasistatic import _parse_retry_outcome

    n_sub = int(policy.max_mech_inner)
    dt_step = float(dt_phys_s)

    # ── Initial Verlet predict + predictor clamp ───────────────
    ex_pred = ex_n + vx_n * dt_step + 0.5 * ax_prev * dt_step ** 2
    ey_pred = ey_n + vy_n * dt_step + 0.5 * ay_prev * dt_step ** 2
    ex_pred, ey_pred, _, _, predictor_clamped = clamp_fn(
        ex_pred, ey_pred, vx_n, vy_n)

    n_contact_events = 0
    p_warm_local = p_warm_init
    if predictor_clamped:
        n_contact_events += 1
        p_warm_local = None

    vx_corr, vy_corr = vx_n, vy_n
    ax_new, ay_new = ax_prev, ay_prev
    P_last: Optional[np.ndarray] = None
    H_last: Optional[np.ndarray] = None
    Fx_hyd = float("nan")
    Fy_hyd = float("nan")
    solve_ok = False
    solve_reason = "no_attempt"
    retry_recovered_step = False
    omega_recovery: Optional[float] = None
    n_trials = 0
    last_n_inner = 0
    trial_log: List[TrialRecord] = []

    # ── for k in range(N_SUB) corrector ────────────────────────
    for k in range(n_sub):
        eps_x_ = ex_pred / context.c
        eps_y_ = ey_pred / context.c
        H_ = build_H_fn(eps_x_, eps_y_)
        n_trials += 1

        result: PressureSolveResult = backend.solve_trial(
            H_curr=H_,
            H_prev=None,
            dt_phys=dt_step,
            omega=context.omega,
            state=backend_state,
            commit=False,
            context=context,
            extra_options=extra_options,
            p_warm_init=p_warm_local,
            vx_squeeze=vx_corr,
            vy_squeeze=vy_corr,
        )
        ok_ = bool(result.converged)
        reason_ = str(result.reason)
        last_n_inner = int(result.n_inner)

        outcome = _parse_retry_outcome(reason_)
        if outcome["retry_recovered"]:
            retry_recovered_step = True
            omega_recovery = outcome["retry_omega"]

        trial_log.append(TrialRecord(
            inner=k,
            relax_used=1.0,
            eps_x_cand=eps_x_,
            eps_y_cand=eps_y_,
            pressure_result=result,
            outcome_solver=GuardOutcome(
                accept=ok_, reason=RejectionReason.NONE if ok_
                else RejectionReason.SOLVER_RESIDUAL,
                detail=reason_,
            ),
            outcome_physical=GuardOutcome(
                accept=True, reason=RejectionReason.NONE),
            outcome_mechanical=GuardOutcome(
                accept=True, reason=RejectionReason.NONE),
            accepted=ok_,
        ))

        if not ok_:
            p_warm_local = None
            P_last = result.P_nd
            H_last = H_
            Fx_hyd = float("nan")
            Fy_hyd = float("nan")
            solve_ok = False
            solve_reason = reason_
            break

        P_last = result.P_nd
        H_last = H_
        Fx_hyd = float(result.Fx_hyd)
        Fy_hyd = float(result.Fy_hyd)
        solve_ok = True
        solve_reason = reason_
        p_warm_local = result.P_nd

        # External load + Verlet correction (legacy formula).
        Fx_ext = float(context.F_ext_x)
        Fy_ext = float(context.F_ext_y)
        ax_new = (Fx_ext + Fx_hyd) / m_shaft
        ay_new = (Fy_ext + Fy_hyd) / m_shaft
        vx_corr = vx_n + 0.5 * (ax_prev + ax_new) * dt_step
        vy_corr = vy_n + 0.5 * (ay_prev + ay_new) * dt_step

        if k < n_sub - 1:
            ex_pred = (ex_n + vx_corr * dt_step
                       + 0.5 * ax_new * dt_step ** 2)
            ey_pred = (ey_n + vy_corr * dt_step
                       + 0.5 * ay_new * dt_step ** 2)
            ex_pred, ey_pred, vx_corr, vy_corr, cl = clamp_fn(
                ex_pred, ey_pred, vx_corr, vy_corr)
            if cl:
                n_contact_events += 1
                p_warm_local = None

    # ── Accept (even on solver fail — legacy contract) ────────
    ex_new, ey_new = ex_pred, ey_pred
    vx_new, vy_new = vx_corr, vy_corr

    ex_new, ey_new, vx_new, vy_new, final_clamped = clamp_fn(
        ex_new, ey_new, vx_new, vy_new)
    if final_clamped:
        n_contact_events += 1
        p_warm_local = None

    step_clamped = bool(predictor_clamped or final_clamped)

    # ── Stateful-backend commit (Stage J fu-2 Step 5) ─────────
    # Per Followup-2 §3.4: commit the accepted hydrodynamic step
    # only when the solver converged AND the step was NOT clamped
    # (boundary-limited gaps don't describe a hydrodynamic state).
    # No rebuild of H from post-clamp ε — that's the explicit
    # difference vs the original Stage J inline code.
    state_committed = False
    theta_for_diag: Optional[np.ndarray] = None
    residual_for_diag = float("nan")
    cav_frac_for_diag = 0.0
    theta_min_for_diag = 1.0
    theta_max_for_diag = 1.0
    dt_ausas_for_diag = 0.0
    if (backend.stateful and solve_ok and not step_clamped
            and H_last is not None):
        commit_result = backend.solve_trial(
            H_curr=H_last,
            H_prev=None,
            dt_phys=dt_step,
            omega=context.omega,
            state=backend_state,
            commit=True,
            context=context,
            extra_options=extra_options,
            p_warm_init=p_warm_local,
            vx_squeeze=vx_corr,
            vy_squeeze=vy_corr,
        )
        # Use the COMMITTED solve as the per-step diagnostic
        # values (mass-conservation invariant): this matches the
        # pre-refactor inline code's behaviour where the post-
        # clamp commit replaced the last-trial pressure.
        if (commit_result.converged
                and commit_result.P_nd is not None):
            P_last = commit_result.P_nd
            theta_for_diag = commit_result.theta
            Fx_hyd = float(commit_result.Fx_hyd)
            Fy_hyd = float(commit_result.Fy_hyd)
            residual_for_diag = float(commit_result.residual)
            cav_frac_for_diag = float(commit_result.cav_frac)
            theta_min_for_diag = float(commit_result.theta_min)
            theta_max_for_diag = float(commit_result.theta_max)
            dt_ausas_for_diag = float(commit_result.dt_ausas)
            state_committed = True
        else:
            # Commit returned non-converged → adapter increments
            # rejected_commit_count internally; runner sees
            # state.step_index unchanged. Do NOT poison diag arrays
            # with the non-converged residual.
            state_committed = False

    # Legacy reports "accepted" even when the SOR solver failed —
    # the mechanical state always advances; only force/pressure
    # arrays go to NaN. Match that contract.
    return MechanicalStepResult(
        accepted=True,
        eps_x_new=ex_new, eps_y_new=ey_new,
        vx_new=vx_new, vy_new=vy_new,
        ax_new=ax_new, ay_new=ay_new,
        H_committed=H_last,
        P_nd_committed=(P_last if solve_ok else None),
        theta_committed=theta_for_diag,
        Fx_hyd_committed=Fx_hyd,
        Fy_hyd_committed=Fy_hyd,
        residual=residual_for_diag,
        n_inner=last_n_inner,
        n_trials=n_trials,
        mech_relax_min_seen=1.0,
        rejection_reason=(
            RejectionReason.NONE if solve_ok
            else RejectionReason.SOLVER_RESIDUAL),
        rejection_detail=("" if solve_ok else solve_reason),
        trial_log=trial_log,
        state_committed=state_committed,
        step_clamped=step_clamped,
        n_contact_events=n_contact_events,
        predictor_clamped=bool(predictor_clamped),
        final_clamped=bool(final_clamped),
        retry_recovered=bool(retry_recovered_step),
        omega_recovery=omega_recovery,
        solve_reason=solve_reason,
        p_warm_out=p_warm_local,
        # Ausas-only diagnostics (zeros for HS).
        cav_frac_committed=cav_frac_for_diag,
        theta_min_committed=theta_min_for_diag,
        theta_max_committed=theta_max_for_diag,
        dt_ausas_committed=dt_ausas_for_diag,
    )


def _run_damped_implicit_film(
    *,
    ex_n: float, ey_n: float, vx_n: float, vy_n: float,
    ax_prev: float, ay_prev: float,
    dt_phys_s: float,
    backend: PressureBackend,
    backend_state: Optional[Any],
    policy: CouplingPolicy,
    extra_options: Optional[dict],
    context: StepContext,
    m_shaft: float,
    eps_max: float,
    clamp_fn,
    build_H_fn,
    p_warm_init: Optional[np.ndarray],
) -> MechanicalStepResult:
    """Damped implicit film coupling — Stage J fu-2 Step 6 skeleton.

    Differences from ``_run_legacy_verlet``:

    * up to ``policy.max_mech_inner`` candidate evaluations
      (default 8 for ``POLICY_AUSAS_DYNAMIC`` vs 3 for legacy);
    * Picard fixed-point blending: each candidate moves only
      ``policy.mech_relax_initial`` (default 0.25) of the way
      toward the new Verlet estimate, damping squeeze-driven
      oscillations between iterations;
    * early termination when the candidate stabilises to within
      ``policy.eps_update_tol`` (Picard convergence);
    * commit gate extended — beyond ``backend.stateful AND solve_ok
      AND not step_clamped`` from legacy, the damped policy ALSO
      requires the fixed-point iteration to have converged. A
      candidate that exhausted ``max_mech_inner`` without
      stabilising is treated as solver-failed for envelope
      statistics (no commit, no poisoned state).

    .. note::
       Step 7 (solver-validity hard gate) and Step 8 (physical
       guards) layer rejection logic on top — bad trials get
       rejected and the candidate-blend is repeated.
       **Step 9** introduces the line-search shrink: when a guard
       rejects, ``mech_relax`` halves and the candidate is
       re-blended; the loop aborts when ``mech_relax`` falls below
       ``policy.mech_relax_min`` (0.03125 by default). At Step 6,
       relax is held constant at ``mech_relax_initial`` for the
       full inner iteration count.

    .. note::
       The commit hook is unchanged from ``_run_legacy_verlet``
       except for the additional fixed-point convergence gate.
       Per Followup-2 §3.4 commit semantics:
       ``commit_on_clamp == False`` and (Step 6 extension)
       ``commit_on_non_convergence == False``.
    """
    from models.diesel_quasistatic import _parse_retry_outcome

    max_mech_inner = int(policy.max_mech_inner)
    relax = float(policy.mech_relax_initial)
    eps_tol = float(policy.eps_update_tol)
    dt_step = float(dt_phys_s)

    # ── Initial Verlet predict + predictor clamp ───────────────
    ex_pred = ex_n + vx_n * dt_step + 0.5 * ax_prev * dt_step ** 2
    ey_pred = ey_n + vy_n * dt_step + 0.5 * ay_prev * dt_step ** 2
    ex_pred, ey_pred, _, _, predictor_clamped = clamp_fn(
        ex_pred, ey_pred, vx_n, vy_n)

    n_contact_events = 0
    p_warm_local = p_warm_init
    if predictor_clamped:
        n_contact_events += 1
        p_warm_local = None

    vx_corr, vy_corr = vx_n, vy_n
    ax_new, ay_new = ax_prev, ay_prev
    P_last: Optional[np.ndarray] = None
    H_last: Optional[np.ndarray] = None
    Fx_hyd = float("nan")
    Fy_hyd = float("nan")
    solve_ok = False
    solve_reason = "no_attempt"
    retry_recovered_step = False
    omega_recovery: Optional[float] = None
    n_trials = 0
    last_n_inner = 0
    last_residual = float("nan")
    trial_log: List[TrialRecord] = []

    fixed_point_converged = False
    mech_relax_min_seen = relax

    # ── Picard fixed-point loop ────────────────────────────────
    for k in range(max_mech_inner):
        eps_x_old = ex_pred / context.c
        eps_y_old = ey_pred / context.c
        H_ = build_H_fn(eps_x_old, eps_y_old)
        n_trials += 1

        result: PressureSolveResult = backend.solve_trial(
            H_curr=H_,
            H_prev=None,
            dt_phys=dt_step,
            omega=context.omega,
            state=backend_state,
            commit=False,
            context=context,
            extra_options=extra_options,
            p_warm_init=p_warm_local,
            vx_squeeze=vx_corr,
            vy_squeeze=vy_corr,
        )
        ok_ = bool(result.converged)
        reason_ = str(result.reason)
        last_n_inner = int(result.n_inner)
        last_residual = float(result.residual)

        outcome = _parse_retry_outcome(reason_)
        if outcome["retry_recovered"]:
            retry_recovered_step = True
            omega_recovery = outcome["retry_omega"]

        trial_log.append(TrialRecord(
            inner=k,
            relax_used=float(relax),
            eps_x_cand=eps_x_old,
            eps_y_cand=eps_y_old,
            pressure_result=result,
            outcome_solver=GuardOutcome(
                accept=ok_,
                reason=(RejectionReason.NONE if ok_
                        else RejectionReason.SOLVER_RESIDUAL),
                detail=reason_,
            ),
            outcome_physical=GuardOutcome(
                accept=True, reason=RejectionReason.NONE),
            outcome_mechanical=GuardOutcome(
                accept=True, reason=RejectionReason.NONE),
            accepted=ok_,
        ))

        if not ok_:
            p_warm_local = None
            P_last = result.P_nd
            H_last = H_
            Fx_hyd = float("nan")
            Fy_hyd = float("nan")
            solve_ok = False
            solve_reason = reason_
            break

        P_last = result.P_nd
        H_last = H_
        Fx_hyd = float(result.Fx_hyd)
        Fy_hyd = float(result.Fy_hyd)
        solve_ok = True
        solve_reason = reason_
        p_warm_local = result.P_nd

        # External load + Verlet correction → new candidate.
        Fx_ext = float(context.F_ext_x)
        Fy_ext = float(context.F_ext_y)
        ax_new = (Fx_ext + Fx_hyd) / m_shaft
        ay_new = (Fy_ext + Fy_hyd) / m_shaft
        vx_full = vx_n + 0.5 * (ax_prev + ax_new) * dt_step
        vy_full = vy_n + 0.5 * (ay_prev + ay_new) * dt_step
        ex_pred_full = (ex_n + vx_full * dt_step
                         + 0.5 * ax_new * dt_step ** 2)
        ey_pred_full = (ey_n + vy_full * dt_step
                         + 0.5 * ay_new * dt_step ** 2)

        # Picard blend: move only ``relax`` toward the new estimate
        # (vs legacy 100% jump). This is what damps squeeze-driven
        # oscillations between Verlet iterations.
        ex_pred_new = ex_pred + relax * (ex_pred_full - ex_pred)
        ey_pred_new = ey_pred + relax * (ey_pred_full - ey_pred)
        vx_corr_new = vx_corr + relax * (vx_full - vx_corr)
        vy_corr_new = vy_corr + relax * (vy_full - vy_corr)

        # Mid-iteration clamp matches legacy_verlet: substep clamp
        # is recorded as a contact event but does NOT set
        # ``step_clamped`` (only predictor / final do).
        ex_blend, ey_blend, vx_blend, vy_blend, cl = clamp_fn(
            ex_pred_new, ey_pred_new, vx_corr_new, vy_corr_new)
        if cl:
            n_contact_events += 1
            p_warm_local = None

        # Picard convergence — magnitude of the candidate update
        # in non-dim ε units.
        delta_eps = float(np.hypot(
            (ex_blend - ex_pred) / context.c,
            (ey_blend - ey_pred) / context.c,
        ))
        ex_pred, ey_pred = ex_blend, ey_blend
        vx_corr, vy_corr = vx_blend, vy_blend
        mech_relax_min_seen = min(mech_relax_min_seen, float(relax))

        if delta_eps <= eps_tol:
            fixed_point_converged = True
            break

    # ── Accept post-iteration mechanical state ────────────────
    ex_new, ey_new = ex_pred, ey_pred
    vx_new, vy_new = vx_corr, vy_corr

    ex_new, ey_new, vx_new, vy_new, final_clamped = clamp_fn(
        ex_new, ey_new, vx_new, vy_new)
    if final_clamped:
        n_contact_events += 1
        p_warm_local = None

    step_clamped = bool(predictor_clamped or final_clamped)

    # ── Stateful commit gate (Followup-2 §3.4 + Step 6 extension) ─
    # Commit only when:
    #   * backend is stateful
    #   * solver converged on the last trial
    #   * step was NOT clamped (boundary-limited gap)
    #   * the Picard iteration converged (candidate stabilised
    #     within ``policy.eps_update_tol``)
    state_committed = False
    theta_for_diag: Optional[np.ndarray] = None
    residual_for_diag = last_residual
    cav_frac_for_diag = 0.0
    theta_min_for_diag = 1.0
    theta_max_for_diag = 1.0
    dt_ausas_for_diag = 0.0
    rejection_reason = (RejectionReason.NONE if solve_ok
                        else RejectionReason.SOLVER_RESIDUAL)
    rejection_detail = ("" if solve_ok else solve_reason)

    if (backend.stateful and solve_ok and not step_clamped
            and fixed_point_converged and H_last is not None):
        commit_result = backend.solve_trial(
            H_curr=H_last,
            H_prev=None,
            dt_phys=dt_step,
            omega=context.omega,
            state=backend_state,
            commit=True,
            context=context,
            extra_options=extra_options,
            p_warm_init=p_warm_local,
            vx_squeeze=vx_corr,
            vy_squeeze=vy_corr,
        )
        if (commit_result.converged
                and commit_result.P_nd is not None):
            P_last = commit_result.P_nd
            theta_for_diag = commit_result.theta
            Fx_hyd = float(commit_result.Fx_hyd)
            Fy_hyd = float(commit_result.Fy_hyd)
            residual_for_diag = float(commit_result.residual)
            cav_frac_for_diag = float(commit_result.cav_frac)
            theta_min_for_diag = float(commit_result.theta_min)
            theta_max_for_diag = float(commit_result.theta_max)
            dt_ausas_for_diag = float(commit_result.dt_ausas)
            state_committed = True
        else:
            state_committed = False
    elif backend.stateful and solve_ok and not fixed_point_converged:
        # Picard iteration exhausted ``max_mech_inner`` without
        # stabilising — treat the step as solver-failed (commit-on-
        # non-convergence=False extension of Followup-2 §3.4).
        rejection_reason = RejectionReason.SOLVER_RESIDUAL
        rejection_detail = (
            f"damped_picard_not_converged after "
            f"{n_trials}/{max_mech_inner} trials")

    return MechanicalStepResult(
        accepted=True,
        eps_x_new=ex_new, eps_y_new=ey_new,
        vx_new=vx_new, vy_new=vy_new,
        ax_new=ax_new, ay_new=ay_new,
        H_committed=H_last,
        P_nd_committed=(P_last if solve_ok else None),
        theta_committed=theta_for_diag,
        Fx_hyd_committed=Fx_hyd,
        Fy_hyd_committed=Fy_hyd,
        residual=residual_for_diag,
        n_inner=last_n_inner,
        n_trials=n_trials,
        mech_relax_min_seen=mech_relax_min_seen,
        rejection_reason=rejection_reason,
        rejection_detail=rejection_detail,
        trial_log=trial_log,
        state_committed=state_committed,
        step_clamped=step_clamped,
        n_contact_events=n_contact_events,
        predictor_clamped=bool(predictor_clamped),
        final_clamped=bool(final_clamped),
        retry_recovered=bool(retry_recovered_step),
        omega_recovery=omega_recovery,
        solve_reason=solve_reason,
        p_warm_out=p_warm_local,
        cav_frac_committed=cav_frac_for_diag,
        theta_min_committed=theta_min_for_diag,
        theta_max_committed=theta_max_for_diag,
        dt_ausas_committed=dt_ausas_for_diag,
    )
