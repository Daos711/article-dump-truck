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
    check_mechanical_candidate,
    check_physical_guards,
    check_solver_validity,
)

# ─── Stage J fu-2 Step 9 fixup-2 — Picard contractivity ────────────
# Constants for the damped-policy contractivity / oscillation / stall
# detector that drives the per-step relax shrink (expert patch).
#
# * ``PICARD_GROW_RATIO`` — δε grew by more than 5% vs the previous
#   iter; symptom: the Picard map is locally non-contractive.
# * ``PICARD_OSC_COS`` — step direction reversed (cos < -0.25) AND
#   ratio > 0.70; symptom: the candidate is bouncing between two
#   attractors of the squeeze film.
# * ``PICARD_STALL_RATIO`` — k≥3 and δε never improved beyond 85%
#   of best_step_norm; symptom: shrink budget is being wasted.
# * ``PICARD_ACTIVE_FLOOR_MULT`` — detector only activates when
#   ``δε > 10 × eps_update_tol``; near-converged tails (δε ~ tol)
#   should NOT shrink relax further (would just delay accept).
PICARD_STALL_RATIO: float = 0.85
PICARD_GROW_RATIO: float = 1.05
PICARD_OSC_COS: float = -0.25
PICARD_ACTIVE_FLOOR_MULT: float = 10.0
PICARD_OSC_RATIO_MIN: float = 0.70

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

    # Stage J fu-2 Step 10 — Picard contractivity diagnostics for
    # Gate 3 summary block + npz schema. Both default to inert
    # values so the legacy_verlet path (which has neither concept)
    # writes consistent per-step arrays.
    fixed_point_converged_flag: bool = False
    picard_shrinks_count: int = 0

    # Stage J fu-2 Task 32 — commit-semantics state machine.
    # Distinguishes "what the final trial did" (which may be a
    # nonfinite failure) from "what the runner actually committed
    # to the trajectory" (which must be finite by the pre-commit
    # gate). Without these fields, a final-trial failure with
    # F_hyd=NaN/res=NaN/converged=False used to print as
    # ``[ACCEPTED]`` because the kernel still committed the legacy
    # ``last_known`` state — misleading.
    #
    # ``final_trial_status``:
    #   "converged" / "solver_nonfinite" / "solver_invalid_input" /
    #   "solver_budget" / "solver_residual" / "picard_not_converged" /
    #   "mechanical_guard" / "physical_guard" / "no_attempt".
    # ``committed_state_status``:
    #   "committed_converged" — final trial was finite + converged.
    #   "committed_last_valid" — final trial failed; last finite
    #                            trial in trial_log was committed.
    #   "rolled_back_previous" — no finite trial existed; mechanics
    #                            rolled back to ex_n / ey_n.
    #   "rejected_no_commit"   — none of the above; trajectory has
    #                            NaN markers (legacy fallback path).
    # ``accepted_state_source``:
    #   "converged_trial" / "last_valid_trial" / "rollback_previous" /
    #   "none".
    final_trial_status: str = "no_attempt"
    committed_state_status: str = "rejected_no_commit"
    accepted_state_source: str = "none"
    committed_state_is_finite: bool = False
    final_trial_failure_kind: str = ""
    final_trial_residual: float = float("nan")
    final_trial_n_inner: int = 0


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
    # Stage J fu-2 Step 9 diagnostic — when True the damped-policy
    # body emits a per-iteration dump after each Picard accept
    # showing anchor / candidate / Verlet-full / blend / Δε /
    # mech_relax / fixed_point_converged. Gated to AVOID per-step
    # cost during normal runs; runner sets this only when the
    # caller's ``debug_first_steps`` window includes the current
    # step. NOT used by the legacy_verlet path.
    debug_dump: bool = False,
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
            guards_cfg=guards_cfg,
            ausas_tol=ausas_tol,
            ausas_max_inner=ausas_max_inner,
            extra_options=extra_options,
            context=context,
            m_shaft=m_shaft,
            eps_max=eps_max,
            clamp_fn=clamp_fn,
            build_H_fn=build_H_fn,
            p_warm_init=p_warm_init,
            debug_dump=debug_dump,
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
    # Stage J fu-2 Task 32 — derive commit-semantics fields from
    # the same ``solve_ok`` signal: converged → committed_converged;
    # SOR-failed → rejected_no_commit (legacy fallback for HS path
    # which has no last-valid-trial concept). The legacy mechanical
    # advance to ex_pred is preserved bit-for-bit (Gate 1).
    if solve_ok:
        _final_trial_status = "converged"
        _committed_state_status = "committed_converged"
        _accepted_state_source = "converged_trial"
        _committed_finite = bool(
            P_last is not None
            and np.all(np.isfinite(np.asarray(P_last)))
            and np.isfinite(Fx_hyd) and np.isfinite(Fy_hyd))
    else:
        _final_trial_status = "solver_residual"
        _committed_state_status = "rejected_no_commit"
        _accepted_state_source = "none"
        _committed_finite = False
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
        # Legacy_verlet has no Picard fixed-point concept and no
        # contractivity shrink — both fields stay at their inert
        # defaults so the per-step npz arrays remain on a single
        # schema for both backends.
        fixed_point_converged_flag=False,
        picard_shrinks_count=0,
        final_trial_status=_final_trial_status,
        committed_state_status=_committed_state_status,
        accepted_state_source=_accepted_state_source,
        committed_state_is_finite=bool(_committed_finite),
        final_trial_failure_kind=("" if solve_ok else solve_reason),
        final_trial_residual=residual_for_diag,
        final_trial_n_inner=last_n_inner,
    )


def _run_damped_implicit_film(
    *,
    ex_n: float, ey_n: float, vx_n: float, vy_n: float,
    ax_prev: float, ay_prev: float,
    dt_phys_s: float,
    backend: PressureBackend,
    backend_state: Optional[Any],
    policy: CouplingPolicy,
    guards_cfg: PhysicalGuardsConfig,
    ausas_tol: float,
    ausas_max_inner: int,
    extra_options: Optional[dict],
    context: StepContext,
    m_shaft: float,
    eps_max: float,
    clamp_fn,
    build_H_fn,
    p_warm_init: Optional[np.ndarray],
    debug_dump: bool = False,
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
    # Stage J fu-2 Step 9 — line-search anchor + last-rejection
    # tracking. ``last_rejection`` carries the GuardOutcome of the
    # most recent guard reject so that, on line-search exhaustion
    # (``relax < mech_relax_min``), the kernel reports the SPECIFIC
    # gate that drove the failure (per user's followup-2 §3.4
    # diagnostic-priority decision: a real rejection bucket like
    # PHYSICAL_PRESSURE_GPA is more informative in summary than a
    # generic MECHANICAL_RELAX_EXHAUSTED).
    last_rejection: Optional[GuardOutcome] = None

    # Stage J fu-2 Step 9 fixup-2 — Picard contractivity tracking.
    # Per expert patch, the relax shrink is driven by THREE signals
    # measured between consecutive accepted Picard iterations:
    #   * growth (δε_k > 1.05 × δε_{k-1});
    #   * oscillation (cos(step_k, step_{k-1}) < -0.25 AND ratio > 0.70);
    #   * stall (k≥3 AND δε_k > 0.85 × min_seen).
    # The detector is gated on δε > 10 × eps_update_tol so near-
    # converged tails don't trigger pointless shrinks.
    prev_step_vec: Optional[np.ndarray] = None
    prev_step_norm: Optional[float] = None
    best_step_norm: float = float("inf")
    # Stage J fu-2 Step 10 — count Picard-shrink events in this
    # mechanical step for the Gate 3 summary aggregator.
    picard_shrinks_count: int = 0

    # Anchor = last accepted candidate (initial = step start).
    ex_anchor, ey_anchor = float(ex_n), float(ey_n)
    vx_anchor, vy_anchor = float(vx_n), float(vy_n)
    ax_anchor, ay_anchor = float(ax_prev), float(ay_prev)
    # Candidate = current attempt (initial = post-predictor-clamp Verlet).
    ex_cand, ey_cand = float(ex_pred), float(ey_pred)
    vx_cand, vy_cand = float(vx_n), float(vy_n)

    # Empty-result placeholder for trials skipped by the mechanical
    # gate (no GPU call → no real PressureSolveResult).
    def _empty_pressure_result(reason: str) -> PressureSolveResult:
        return PressureSolveResult(
            P_nd=None, theta=None,
            Fx_hyd=float("nan"), Fy_hyd=float("nan"),
            H_used=np.zeros((1, 1)),
            residual=float("nan"), n_inner=0, converged=False,
            backend_name=backend.name, reason=reason,
        )

    # ── Picard + line-search loop ──────────────────────────────
    for k in range(max_mech_inner):
        n_trials += 1

        # Stage J fu-2 Step 9 — mechanical candidate guard runs
        # BEFORE the GPU call. A candidate that violates Δε bounds
        # or sits at the eps_max wall is rejected immediately
        # (saves a GPU trial that would only produce a useless
        # GPa pressure). Skipped when the policy declares no
        # bounds (``max_delta_eps_inner is None``).
        mech_outcome = check_mechanical_candidate(
            eps_x_curr=ex_anchor / context.c,
            eps_y_curr=ey_anchor / context.c,
            eps_x_cand=ex_cand / context.c,
            eps_y_cand=ey_cand / context.c,
            eps_x_step_start=ex_n / context.c,
            eps_y_step_start=ey_n / context.c,
            eps_max=eps_max,
            max_delta_eps_inner=policy.max_delta_eps_inner,
            max_delta_eps_step=policy.max_delta_eps_step,
        )

        if debug_dump:
            _dx_inner = float(np.hypot(
                (ex_cand - ex_anchor) / context.c,
                (ey_cand - ey_anchor) / context.c))
            _dx_step = float(np.hypot(
                (ex_cand - ex_n) / context.c,
                (ey_cand - ey_n) / context.c))
            _eps_mag = float(np.hypot(
                ex_cand / context.c, ey_cand / context.c))
            print(
                f"    [J9-dump k={k} ENTER] "
                f"anchor=({ex_anchor/context.c:+.6f},"
                f"{ey_anchor/context.c:+.6f}) "
                f"cand=({ex_cand/context.c:+.6f},"
                f"{ey_cand/context.c:+.6f}) "
                f"|Δε_inner|={_dx_inner:.6f} "
                f"|Δε_step|={_dx_step:.6f} |ε|={_eps_mag:.6f} "
                f"relax={relax:.5f} "
                f"mech_outcome={mech_outcome.reason.value}",
                flush=True)

        if not mech_outcome.accept:
            # Mechanical reject — log skipped trial, shrink relax,
            # re-blend toward anchor; NO GPU call.
            trial_log.append(TrialRecord(
                inner=k, relax_used=float(relax),
                eps_x_cand=ex_cand / context.c,
                eps_y_cand=ey_cand / context.c,
                pressure_result=_empty_pressure_result(
                    "skipped_by_mechanical_guard"),
                outcome_solver=GuardOutcome(
                    accept=True,
                    reason=RejectionReason.NONE,
                    detail="skipped"),
                outcome_physical=GuardOutcome(
                    accept=True,
                    reason=RejectionReason.NONE,
                    detail="skipped"),
                outcome_mechanical=mech_outcome,
                accepted=False,
            ))
            last_rejection = mech_outcome
            relax *= 0.5
            mech_relax_min_seen = min(mech_relax_min_seen, relax)
            if relax < float(policy.mech_relax_min):
                solve_ok = False
                solve_reason = (
                    f"line_search_exhausted_"
                    f"{mech_outcome.reason.value}: "
                    f"{mech_outcome.detail}")
                break
            ex_cand = ex_anchor + relax * (ex_cand - ex_anchor)
            ey_cand = ey_anchor + relax * (ey_cand - ey_anchor)
            vx_cand = vx_anchor + relax * (vx_cand - vx_anchor)
            vy_cand = vy_anchor + relax * (vy_cand - vy_anchor)
            ex_cand, ey_cand, vx_cand, vy_cand, cl = clamp_fn(
                ex_cand, ey_cand, vx_cand, vy_cand)
            if cl:
                n_contact_events += 1
                p_warm_local = None
            continue

        # ── GPU trial ──
        H_ = build_H_fn(ex_cand / context.c, ey_cand / context.c)

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
            vx_squeeze=vx_cand,
            vy_squeeze=vy_cand,
        )
        ok_ = bool(result.converged)
        reason_ = str(result.reason)
        last_n_inner = int(result.n_inner)
        last_residual = float(result.residual)

        outcome = _parse_retry_outcome(reason_)
        if outcome["retry_recovered"]:
            retry_recovered_step = True
            omega_recovery = outcome["retry_omega"]

        # Solver-validity (Step 7).
        solver_outcome = check_solver_validity(
            backend_name=backend.name,
            P_nd=result.P_nd, theta=result.theta,
            residual=result.residual, n_inner=result.n_inner,
            converged=result.converged,
            ausas_tol=ausas_tol,
            ausas_max_inner=ausas_max_inner,
        )

        # Physical guards (Step 8).
        phys_outcome: GuardOutcome
        guards_mode = guards_cfg.mode
        if (policy.enable_physical_guards
                and guards_mode != "off"
                and solver_outcome.accept):
            phys_outcome = check_physical_guards(
                P_nd=result.P_nd,
                p_scale=context.p_scale,
                Fx_hyd=result.Fx_hyd, Fy_hyd=result.Fy_hyd,
                Fx_ext=context.F_ext_x,
                Fy_ext=context.F_ext_y,
                theta=result.theta,
                phi_deg=context.phi_deg,
                cfg=guards_cfg,
            )
            if (not phys_outcome.accept
                    and guards_mode == "diagnostic"):
                import warnings as _warnings
                _warnings.warn(
                    f"Stage-J physical guard (diagnostic mode): "
                    f"{phys_outcome.reason.value} — "
                    f"{phys_outcome.detail}",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            phys_outcome = GuardOutcome(
                accept=True, reason=RejectionReason.NONE)

        accepted_trial = bool(
            ok_
            and solver_outcome.accept
            and (phys_outcome.accept or guards_mode != "hard"))

        trial_log.append(TrialRecord(
            inner=k, relax_used=float(relax),
            eps_x_cand=ex_cand / context.c,
            eps_y_cand=ey_cand / context.c,
            pressure_result=result,
            outcome_solver=solver_outcome,
            outcome_physical=phys_outcome,
            outcome_mechanical=mech_outcome,
            accepted=accepted_trial,
        ))

        if not accepted_trial:
            # Track which gate rejected for line-search-exhausted
            # ``rejection_reason``. Solver-validity has priority,
            # then physical guards (only in hard mode), else fall
            # through to a generic SOLVER_RESIDUAL bucket.
            if not solver_outcome.accept:
                last_rejection = solver_outcome
            elif (not phys_outcome.accept
                    and guards_mode == "hard"):
                last_rejection = phys_outcome
            else:
                last_rejection = GuardOutcome(
                    accept=False,
                    reason=RejectionReason.SOLVER_RESIDUAL,
                    detail=reason_,
                )

            # Stage J fu-2 Step 9 fixup — if the rejected candidate
            # is already AT the anchor (e.g. on k=0 the predictor
            # equals ex_n / ey_n because vx_n=vy_n=ax_prev=ay_prev=0),
            # then ``cand = anchor + relax*(cand - anchor) = anchor``
            # for any relax — the line-search would issue (n)
            # IDENTICAL GPU trials before exhausting. That's a real
            # solver failure on the anchor itself; abort
            # immediately with the SAME rejection reason rather
            # than spinning the budget.
            _retreat_dx = float(np.hypot(
                (ex_cand - ex_anchor) / context.c,
                (ey_cand - ey_anchor) / context.c))
            if _retreat_dx <= 1e-12:
                if debug_dump:
                    print(
                        f"    [J9-dump k={k} REJECT-AT-ANCHOR] "
                        f"solver={solver_outcome.reason.value} "
                        f"phys={phys_outcome.reason.value} "
                        f"converged={ok_} reason={reason_!r} "
                        f"retreat_Δε={_retreat_dx:.3e} ≤ 1e-12 "
                        f"— abort line-search (anchor itself "
                        f"failed; no shrink can recover)",
                        flush=True)
                solve_ok = False
                P_last = result.P_nd
                H_last = H_
                Fx_hyd = float("nan")
                Fy_hyd = float("nan")
                solve_reason = (
                    f"reject_at_anchor_"
                    f"{last_rejection.reason.value}: "
                    f"{last_rejection.detail}")
                break

            if debug_dump:
                print(
                    f"    [J9-dump k={k} REJECT] "
                    f"solver={solver_outcome.reason.value} "
                    f"phys={phys_outcome.reason.value} "
                    f"converged={ok_} reason={reason_!r} "
                    f"shrink relax {relax:.5f} → {relax*0.5:.5f}",
                    flush=True)

            relax *= 0.5
            mech_relax_min_seen = min(mech_relax_min_seen, relax)
            if relax < float(policy.mech_relax_min):
                solve_ok = False
                P_last = result.P_nd
                H_last = H_
                Fx_hyd = float("nan")
                Fy_hyd = float("nan")
                solve_reason = (
                    f"line_search_exhausted_"
                    f"{last_rejection.reason.value}: "
                    f"{last_rejection.detail}")
                break

            # Retreat candidate toward anchor with smaller relax.
            ex_cand = ex_anchor + relax * (ex_cand - ex_anchor)
            ey_cand = ey_anchor + relax * (ey_cand - ey_anchor)
            vx_cand = vx_anchor + relax * (vx_cand - vx_anchor)
            vy_cand = vy_anchor + relax * (vy_cand - vy_anchor)
            ex_cand, ey_cand, vx_cand, vy_cand, cl = clamp_fn(
                ex_cand, ey_cand, vx_cand, vy_cand)
            if cl:
                n_contact_events += 1
                p_warm_local = None
            continue

        # ── Trial ACCEPTED ─────────────────────────────────────
        P_last = result.P_nd
        H_last = H_
        Fx_hyd = float(result.Fx_hyd)
        Fy_hyd = float(result.Fy_hyd)
        solve_ok = True
        solve_reason = reason_
        p_warm_local = result.P_nd

        # New anchor = current accepted candidate.
        ex_anchor = ex_cand
        ey_anchor = ey_cand
        vx_anchor = vx_cand
        vy_anchor = vy_cand

        # Compute Verlet-full target with the new force.
        Fx_ext = float(context.F_ext_x)
        Fy_ext = float(context.F_ext_y)
        ax_new = (Fx_ext + Fx_hyd) / m_shaft
        ay_new = (Fy_ext + Fy_hyd) / m_shaft
        ax_anchor = ax_new
        ay_anchor = ay_new
        vx_full = vx_n + 0.5 * (ax_prev + ax_new) * dt_step
        vy_full = vy_n + 0.5 * (ay_prev + ay_new) * dt_step
        ex_full = (ex_n + vx_full * dt_step
                    + 0.5 * ax_new * dt_step ** 2)
        ey_full = (ey_n + vy_full * dt_step
                    + 0.5 * ay_new * dt_step ** 2)

        # Stage J fu-2 Step 9 fixup-2 — DO NOT reset relax here.
        # The expert patch keeps the relax shrunk by the
        # contractivity detector (below) across the entire
        # mechanical step. Resetting to ``mech_relax_initial`` after
        # every accept defeats the shrink: shrunk_relax → calmer
        # candidate → next accept → reset to 0.25 → oscillation
        # restarts. Auto-grow back is intentionally NOT done in
        # the smoke profile.

        # Picard residual BEFORE relaxation (the un-damped Verlet
        # step the solver wants to take). Used to (a) drive the
        # convergence-after-shrink check and (b) compare to the
        # relaxed step for the contractivity ratio.
        picard_vec = np.array([
            (ex_full - ex_anchor) / context.c,
            (ey_full - ey_anchor) / context.c,
        ], dtype=float)
        picard_res_unrelaxed = float(np.linalg.norm(picard_vec))

        # Picard blend: next-iteration candidate.
        next_ex = ex_anchor + relax * (ex_full - ex_anchor)
        next_ey = ey_anchor + relax * (ey_full - ey_anchor)
        next_vx = vx_anchor + relax * (vx_full - vx_anchor)
        next_vy = vy_anchor + relax * (vy_full - vy_anchor)
        next_ex, next_ey, next_vx, next_vy, cl = clamp_fn(
            next_ex, next_ey, next_vx, next_vy)
        if cl:
            n_contact_events += 1
            p_warm_local = None

        step_vec = np.array([
            (next_ex - ex_anchor) / context.c,
            (next_ey - ey_anchor) / context.c,
        ], dtype=float)
        delta_eps = float(np.linalg.norm(step_vec))

        # Convergence check — the relaxed step AND one of two
        # secondary conditions must be small.
        #
        # Primary condition (always required): δε_blend ≤ eps_tol —
        # the actual orbit step the runner is about to apply is
        # within tolerance.
        #
        # Either secondary condition:
        # (a) picard_res_unrelaxed ≤ picard_res_tol — the Picard
        #     map's un-relaxed residual is small. ``picard_res_tol
        #     = eps_tol / mech_relax_initial`` scales the un-
        #     relaxed tolerance back so the expected equivalent
        #     damped tolerance is recovered. This is the
        #     "Picard-clean" path — works at the default relax.
        # (b) k ≥ 3 — fixup-3 softening: when contractivity-driven
        #     shrink has driven relax to a small value (e.g.
        #     0.0625), the unrelaxed residual stays intrinsically
        #     16× the actual step even after the orbit has
        #     stabilised. δε_blend is the ground truth: if the
        #     real orbit step is within tolerance and we've done
        #     enough iterations to be sure (k ≥ 3), declare
        #     convergence regardless of unrelaxed-residual size.
        #     The k ≥ 3 floor avoids declaring convergence on
        #     k=0/k=1 with very-small starting relax — those
        #     iterations haven't seen enough of the Picard map.
        picard_res_tol = (
            float(eps_tol) / float(policy.mech_relax_initial))
        if delta_eps <= eps_tol and (
                picard_res_unrelaxed <= picard_res_tol or k >= 3):
            if debug_dump:
                print(
                    f"    [J9-dump k={k} ACCEPT-CONVERGED] "
                    f"Δε_blend={delta_eps:.6e} ≤ tol={eps_tol:.1e} "
                    f"AND (picard_res={picard_res_unrelaxed:.6e} "
                    f"≤ {picard_res_tol:.3e} OR k={k} ≥ 3)",
                    flush=True)
            fixed_point_converged = True
            break

        # Contractivity / oscillation / stall detector. Only active
        # when δε significantly exceeds the convergence tolerance.
        too_large = False
        picard_ratio = float("nan")
        picard_cos = float("nan")
        if (prev_step_norm is not None
                and prev_step_vec is not None
                and delta_eps > PICARD_ACTIVE_FLOOR_MULT
                * float(eps_tol)):
            picard_ratio = (delta_eps
                            / max(float(prev_step_norm), 1e-30))
            picard_cos = float(
                np.dot(step_vec, prev_step_vec)
                / max(delta_eps * float(prev_step_norm), 1e-30))
            growth_bad = picard_ratio > PICARD_GROW_RATIO
            osc_bad = (picard_cos < PICARD_OSC_COS
                       and picard_ratio > PICARD_OSC_RATIO_MIN)
            stall_bad = (
                k >= 3
                and delta_eps > PICARD_STALL_RATIO * best_step_norm)
            too_large = growth_bad or osc_bad or stall_bad

        # Apply Picard-shrink — halve relax (down to mech_relax_min)
        # and re-blend the candidate from the SAME accepted anchor /
        # Verlet-full target. Distinct from the line-search shrink
        # (which fires on guard reject and retreats the candidate
        # toward anchor on a rejected GPU trial).
        if too_large and relax > float(policy.mech_relax_min):
            picard_shrinks_count += 1
            old_relax = relax
            relax = max(float(policy.mech_relax_min), 0.5 * relax)
            mech_relax_min_seen = min(mech_relax_min_seen, relax)
            next_ex = ex_anchor + relax * (ex_full - ex_anchor)
            next_ey = ey_anchor + relax * (ey_full - ey_anchor)
            next_vx = vx_anchor + relax * (vx_full - vx_anchor)
            next_vy = vy_anchor + relax * (vy_full - vy_anchor)
            next_ex, next_ey, next_vx, next_vy, cl = clamp_fn(
                next_ex, next_ey, next_vx, next_vy)
            if cl:
                n_contact_events += 1
                p_warm_local = None
            step_vec = np.array([
                (next_ex - ex_anchor) / context.c,
                (next_ey - ey_anchor) / context.c,
            ], dtype=float)
            delta_eps = float(np.linalg.norm(step_vec))
            if debug_dump:
                print(
                    f"    [J9-dump k={k} PICARD-SHRINK] "
                    f"old_relax={old_relax:.5f} → "
                    f"{relax:.5f} "
                    f"Δε_blend={delta_eps:.6f} "
                    f"picard_res={picard_res_unrelaxed:.6f} "
                    f"ratio={picard_ratio:.3f} "
                    f"cos={picard_cos:+.3f} "
                    f"best={best_step_norm:.6f}",
                    flush=True)

        if debug_dump:
            print(
                f"    [J9-dump k={k} ACCEPT] "
                f"|F_hyd|=({Fx_hyd/1e3:+.4f},{Fy_hyd/1e3:+.4f})kN "
                f"|F_ext|=({Fx_ext/1e3:+.4f},{Fy_ext/1e3:+.4f})kN "
                f"a_new=({ax_new:+.4e},{ay_new:+.4e}) "
                f"v_full=({vx_full:+.4e},{vy_full:+.4e}) "
                f"ε_anchor=({ex_anchor/context.c:+.6f},"
                f"{ey_anchor/context.c:+.6f}) "
                f"ε_full=({ex_full/context.c:+.6f},"
                f"{ey_full/context.c:+.6f}) "
                f"ε_blend=({next_ex/context.c:+.6f},"
                f"{next_ey/context.c:+.6f}) "
                f"relax={relax:.5f} "
                f"picard_res_unrelaxed={picard_res_unrelaxed:.6f} "
                f"Δε_blend={delta_eps:.6f} "
                f"ratio={picard_ratio:.3f} "
                f"cos={picard_cos:+.3f} "
                f"eps_tol={eps_tol:.1e}",
                flush=True)

        # Store contractivity diagnostics AFTER any shrink — the
        # next iteration compares against the FINAL accepted step,
        # not the pre-shrink one.
        prev_step_vec = step_vec
        prev_step_norm = delta_eps
        best_step_norm = min(best_step_norm, delta_eps)

        ex_cand, ey_cand = next_ex, next_ey
        vx_cand, vy_cand = next_vx, next_vy

    # ── End of Picard + line-search loop ──────────────────────
    if debug_dump:
        print(
            f"    [J9-dump END] n_trials={n_trials} "
            f"solve_ok={solve_ok} "
            f"fixed_point_converged={fixed_point_converged} "
            f"mech_relax_min_seen={mech_relax_min_seen:.5f} "
            f"ε_anchor=({ex_anchor/context.c:+.6f},"
            f"{ey_anchor/context.c:+.6f}) "
            f"last_rejection="
            f"{last_rejection.reason.value if last_rejection else 'none'}",
            flush=True)
    # Anchor holds the last accepted state (or the step-start
    # state if no trial ever succeeded).
    ex_pred = ex_anchor
    ey_pred = ey_anchor
    vx_corr = vx_anchor
    vy_corr = vy_anchor
    ax_new = ax_anchor
    ay_new = ay_anchor

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
    # Stage J fu-2 Step 7 — surface the SPECIFIC solver-validity
    # rejection reason (SOLVER_BUDGET / SOLVER_RESIDUAL /
    # SOLVER_NONFINITE / SOLVER_NEG_PRESSURE /
    # SOLVER_THETA_OUT_OF_RANGE) from the last trial's
    # ``outcome_solver`` rather than the bucket default
    # ``SOLVER_RESIDUAL``. This is what lets the Stage 10 summary
    # writer report ``rejected_by_solver`` with the correct
    # subdivision per Gate 3.
    if solve_ok:
        rejection_reason = RejectionReason.NONE
        rejection_detail = ""
    else:
        # Step 9 — surface ``last_rejection.reason`` (the SPECIFIC
        # gate that drove every retry until the line search
        # exhausted ``mech_relax_min``). Per user followup-2 §3.4:
        # NO new MECHANICAL_RELAX_EXHAUSTED bucket — a real
        # rejection like SOLVER_BUDGET / PHYSICAL_PRESSURE_GPA /
        # MECHANICAL_DELTA_EPS_INNER preserves diagnostic value
        # for the Stage 10 summary. ``last_rejection`` is set on
        # every guard-reject branch (solver, physical-hard,
        # mechanical) so this covers all paths to ``solve_ok=False``.
        if last_rejection is not None:
            rejection_reason = last_rejection.reason
        else:
            # Defensive — solve_ok=False without a recorded
            # last_rejection should not happen after Step 9, but
            # keep the bucket fallback for robustness.
            rejection_reason = RejectionReason.SOLVER_RESIDUAL
        rejection_detail = solve_reason

    # Stage J fu-2 fixup-3 — Picard no-advance flag.
    # For a stateful Ausas backend, pressure / theta state and the
    # mechanical state form ONE coupled object. Advancing mechanics
    # while the Ausas commit is blocked (Picard didn't reach a fixed
    # point) leaves ``state.H_prev`` lagging behind the orbit; the
    # next step's predictor then drives an artificially huge
    # ∂h/∂t = (H_curr - H_prev)/dt squeeze impulse and the solver
    # honestly returns 250+ MPa from a desync, not a real peak. The
    # fix: when fixed_point_converged=False on a stateful backend,
    # do a FULL no-advance — no pressure commit AND mechanical state
    # rolls back to step start. Set in the elif branch below.
    picard_no_advance = False
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
        # Step 9 fixup-2: enrich the detail string with the relax
        # floor and the last accepted step δε so the Stage 10
        # summary writer can flag iterations that stalled at the
        # min relax (pathology) vs. ran out of budget on a
        # productively contracting δε (just need bigger budget).
        # Stage J fu-2 Task 27 — use the dedicated
        # ``PICARD_NOT_CONVERGED`` enum so failure_classifier
        # buckets this as ``picard_noncontractive`` rather than
        # mislabelling it as ``solver_residual``. The Ausas trials
        # all returned converged + finite; the failure is on the
        # coupling Picard side, not the solver.
        rejection_reason = RejectionReason.PICARD_NOT_CONVERGED
        rejection_detail = (
            f"damped_picard_not_converged after "
            f"{n_trials}/{max_mech_inner} trials; "
            f"min_relax={mech_relax_min_seen:.5f}; "
            f"last_delta_eps="
            f"{prev_step_norm if prev_step_norm is not None else 0.0:.3e}"
        )
        # Stage J fu-2 fixup-3 — full no-advance.
        # Diagnosis: smoke 190-209 showed steps 190-196 with
        # fixed_point_converged=False and mech_relax_min_seen=0.0625
        # (relax floor). Mechanics still advanced to ex_anchor;
        # Ausas commit was blocked. After 6 such steps state.H_prev
        # lagged the orbit by ~7 dt, so step 197's ∂h/∂t had a
        # 7-dt numerator over a 1-dt denominator → artificial
        # squeeze impulse → 250+ MPa peak that the smoke pressure
        # cap then rejected on k=0. The PDE was correct, the
        # H_curr/H_prev pair was desynced. Fix: roll mechanics back
        # to step start and zero out trial outputs so nothing leaks
        # downstream as warm-start or a fake hydrodynamic state.
        picard_no_advance = True
        solve_ok = False
        P_last = None
        H_last = None
        theta_for_diag = None
        Fx_hyd = float("nan")
        Fy_hyd = float("nan")
        residual_for_diag = float("nan")
        p_warm_local = None
        # Roll mechanical state back to the input of this kernel
        # call (last accepted state from the previous mechanical
        # step). Overwrites the ex_new/vx_new/ax_new currently
        # holding the post-clamp anchor values from the Picard loop.
        ex_new, ey_new = float(ex_n), float(ey_n)
        vx_new, vy_new = float(vx_n), float(vy_n)
        ax_new, ay_new = float(ax_prev), float(ay_prev)

    # ─── Stage J fu-2 Task 32 — commit-semantics finalisation ──
    # Compute the four state-machine labels from local variables
    # already set above. No additional GPU calls — we use what
    # ``state_committed`` / ``picard_no_advance`` / the local
    # P/F/theta finite flags tell us. The pre-commit finite gate
    # is a SOFT check: if the committed state is non-finite, we
    # mark it as ``rolled_back_previous`` and reset eps locals
    # to the kernel's start-of-step inputs so the runner sees a
    # safe rollback. This catches the pathological case where
    # the runner used to print ``[ACCEPTED] |F_hyd|=nankN ...``
    # because a non-converged commit silently kept last-trial
    # NaN values.
    def _finite_arr(a) -> bool:
        if a is None:
            return False
        return bool(np.all(np.isfinite(np.asarray(a))))

    _commit_state_finite = (
        _finite_arr(P_last)
        and (theta_for_diag is None
              or _finite_arr(theta_for_diag))
        and np.isfinite(Fx_hyd)
        and np.isfinite(Fy_hyd)
    )

    if picard_no_advance:
        # Already rolled back above. The kernel earlier set
        # ex_new=ex_n / vx_new=vx_n / ax_new=ax_prev so mechanics
        # are at the kernel-input state — finite by construction.
        final_trial_status = "picard_not_converged"
        committed_state_status = "rolled_back_previous"
        accepted_state_source = "rollback_previous"
        committed_state_is_finite = True
        final_trial_failure_kind = "picard_not_converged"
        final_trial_residual = (prev_step_norm
                                  if prev_step_norm is not None
                                  else float("nan"))
    elif state_committed and _commit_state_finite:
        final_trial_status = "converged"
        committed_state_status = "committed_converged"
        accepted_state_source = "converged_trial"
        committed_state_is_finite = True
        final_trial_failure_kind = ""
        final_trial_residual = residual_for_diag
    elif solve_ok and _commit_state_finite:
        # Last accepted Picard trial's pressure / F_hyd are finite,
        # but ``state_committed=False`` means the FINAL commit-time
        # GPU call returned non-converged (e.g. squeeze near
        # ε_max). Trajectory continuity uses the last-valid-trial
        # values that the existing code already keeps in P_last /
        # Fx_hyd / Fy_hyd.
        final_trial_status = (
            "solver_nonfinite"
            if (not np.isfinite(residual_for_diag))
            else "solver_residual")
        committed_state_status = "committed_last_valid"
        accepted_state_source = "last_valid_trial"
        committed_state_is_finite = True
        final_trial_failure_kind = (
            "commit_call_not_converged"
            if not np.isfinite(residual_for_diag)
            else "")
        final_trial_residual = residual_for_diag
    else:
        # ``solve_ok=False`` (line-search exhausted) OR
        # committed state non-finite. Rollback mechanics to the
        # kernel-input state and zero out trial outputs so the
        # runner doesn't write NaN-tainted P / F / theta to the
        # trajectory and doesn't print ``[ACCEPTED]`` for it.
        ex_new, ey_new = float(ex_n), float(ey_n)
        vx_new, vy_new = float(vx_n), float(vy_n)
        ax_new, ay_new = float(ax_prev), float(ay_prev)
        P_last = None
        theta_for_diag = None
        Fx_hyd = float("nan")
        Fy_hyd = float("nan")
        residual_for_diag = float("nan")
        p_warm_local = None
        # Classify failure-kind for the dump trigger / classifier.
        if not _commit_state_finite and state_committed:
            # Pathological: commit returned converged but the
            # state itself was non-finite — sentinel for the
            # gpu-reynolds Task 12 nonfinite signal.
            final_trial_status = "solver_nonfinite"
            final_trial_failure_kind = "nonfinite_state"
        elif rejection_reason == RejectionReason.PICARD_NOT_CONVERGED:
            final_trial_status = "picard_not_converged"
            final_trial_failure_kind = "picard_not_converged"
        elif rejection_reason == RejectionReason.SOLVER_BUDGET:
            final_trial_status = "solver_budget"
            final_trial_failure_kind = "solver_budget"
        elif rejection_reason == RejectionReason.SOLVER_NONFINITE:
            final_trial_status = "solver_nonfinite"
            final_trial_failure_kind = "nonfinite_state"
        elif rejection_reason == RejectionReason.SOLVER_RESIDUAL:
            final_trial_status = "solver_residual"
            final_trial_failure_kind = "solver_residual"
        elif rejection_reason in (
                RejectionReason.MECHANICAL_DELTA_EPS_INNER,
                RejectionReason.MECHANICAL_DELTA_EPS_STEP,
                RejectionReason.MECHANICAL_EPS_MAX):
            final_trial_status = "mechanical_guard"
            final_trial_failure_kind = "mechanical_guard"
        elif rejection_reason in (
                RejectionReason.PHYSICAL_PRESSURE_GPA,
                RejectionReason.PHYSICAL_FORCE_RATIO,
                RejectionReason.PHYSICAL_SAME_DIR_RUNAWAY,
                RejectionReason.PHYSICAL_CAV_RUNAWAY):
            final_trial_status = "physical_guard"
            final_trial_failure_kind = "physical_guard"
        else:
            final_trial_status = "no_attempt"
            final_trial_failure_kind = ""
        committed_state_status = "rolled_back_previous"
        accepted_state_source = "rollback_previous"
        committed_state_is_finite = True  # kernel-input ε is by-construction finite
        final_trial_residual = float("nan")
        state_committed = False
        # accepted reflects whether the runner should treat this as
        # a real advance for trajectory continuity. Rollback is NOT
        # a real advance — set False so the runner-side
        # ``_ms.accepted=False`` branch fires.
        picard_no_advance = True

    return MechanicalStepResult(
        accepted=not bool(picard_no_advance),
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
        fixed_point_converged_flag=bool(fixed_point_converged),
        picard_shrinks_count=int(picard_shrinks_count),
        final_trial_status=final_trial_status,
        committed_state_status=committed_state_status,
        accepted_state_source=accepted_state_source,
        committed_state_is_finite=bool(committed_state_is_finite),
        final_trial_failure_kind=final_trial_failure_kind,
        final_trial_residual=float(final_trial_residual)
            if np.isfinite(final_trial_residual) else float("nan"),
        final_trial_n_inner=int(last_n_inner),
    )
