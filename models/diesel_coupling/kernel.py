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
        raise NotImplementedError(
            "damped_implicit_film policy lands at Step 6/7/8/9 "
            "of Stage J followup-2; the kernel currently only "
            "wires the legacy_verlet path.")
    raise ValueError(
        f"advance_mechanical_step: unknown policy {policy.name!r}; "
        f"expected 'legacy_verlet' or 'damped_implicit_film'.")


def _run_legacy_verlet(
    *,
    ex_n: float, ey_n: float, vx_n: float, vy_n: float,
    ax_prev: float, ay_prev: float,
    dt_phys_s: float,
    backend: PressureBackend,
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
            state=None,
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
        theta_committed=None,
        Fx_hyd_committed=Fx_hyd,
        Fy_hyd_committed=Fy_hyd,
        residual=float("nan"),
        n_inner=last_n_inner,
        n_trials=n_trials,
        mech_relax_min_seen=1.0,
        rejection_reason=(
            RejectionReason.NONE if solve_ok
            else RejectionReason.SOLVER_RESIDUAL),
        rejection_detail=("" if solve_ok else solve_reason),
        trial_log=trial_log,
        state_committed=False,         # HS is stateless
        step_clamped=step_clamped,
        n_contact_events=n_contact_events,
        predictor_clamped=bool(predictor_clamped),
        final_clamped=bool(final_clamped),
        retry_recovered=bool(retry_recovered_step),
        omega_recovery=omega_recovery,
        solve_reason=solve_reason,
        p_warm_out=p_warm_local,
    )
