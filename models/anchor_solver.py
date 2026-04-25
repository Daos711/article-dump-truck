"""Stage I-A anchor solver.

Anchor solve is NOT angle continuation. It is a separate entry procedure
that lands on the branch at a chosen, well-conditioned crank angle. Only
after an anchor is accepted, the regular continuation_phi runner takes over.

Two-stage procedure:
  1. solve_anchor_smooth(): mild load homotopy F_h(X,Y) + lambda*Wa = 0,
     ramp lambda over a short schedule with warm-start between stages.
  2. solve_anchor_textured(): seed direct textured solve from accepted
     smooth anchor state at the same phi_a; if direct solve fails, use a
     short geometry continuation in alpha_tex (texture amplitude).

Anchor selection policy: explicit / from_legacy_matched_sector / scout_best.
For the current fallback surrogate_heavyduty_v1 case the pragmatic default
is phi_anchor_deg=500.0 (medium-load, well-conditioned sector).

Interface contract with the solver is unchanged: this module only consumes
eval_factory(mode) -> eval_fn(X, Y, g_init) closures already prepared by
the pipeline runner.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── PS budget modes ───────────────────────────────────────────────
#
# Pipeline-side adaptive PS budgets. The pipeline `eval_factory(mode)` is
# expected to honour these mode names. Trial evaluations are cheaper than
# accepted evaluations on purpose (Section 5.3 of the patch spec).

PS_BUDGETS: Dict[str, Dict[str, int]] = {
    "scout":               dict(ps_max_iter=12_000, hs_warmup_iter=1_000),
    "trial":               dict(ps_max_iter=15_000, hs_warmup_iter=2_000),
    "anchor_stage_first":  dict(ps_max_iter=100_000, hs_warmup_iter=25_000),
    "anchor_stage_later":  dict(ps_max_iter=80_000,  hs_warmup_iter=5_000),
    "accepted_node":       dict(ps_max_iter=25_000, hs_warmup_iter=2_000),
    "midpoint_rescue":     dict(ps_max_iter=50_000, hs_warmup_iter=10_000),
}


# ─── Anchor result types ───────────────────────────────────────────

@dataclass
class AnchorState:
    """Final accepted anchor state, ready for continuation runner."""
    phi_deg: float
    X: float
    Y: float
    eps: float
    attitude_deg: float
    h_min: float
    p_max: float
    cav_frac: float
    friction: float
    Ploss: float
    Qout: float
    rel_residual: float
    g: Optional[np.ndarray] = None  # warm-start pressure state (P/theta packed)
    status: str = "hard_converged"


@dataclass
class AnchorReport:
    """Diagnostic record for a single anchor solve."""
    mode: str                # "explicit" / "scout_best" / "from_legacy_matched_sector"
    phi_anchor_deg: float
    candidates: List[float] = field(default_factory=list)
    selection_reason: str = ""
    smooth_lambda_log: List[Dict[str, Any]] = field(default_factory=list)
    textured_log: List[Dict[str, Any]] = field(default_factory=list)
    smooth_state: Optional[AnchorState] = None
    textured_state: Optional[AnchorState] = None
    smooth_ok: bool = False
    textured_ok: bool = False
    wall_total_sec: float = 0.0


# ─── Anchor selection policy ───────────────────────────────────────

# Pragmatic defaults for the current fallback surrogate_heavyduty_v1 case.
DEFAULT_EXPLICIT_PHI_ANCHOR_DEG: float = 500.0
DEFAULT_BACKUP_PHI_LIST: Tuple[float, ...] = (490.0, 510.0, 530.0, 540.0, 450.0)
DEFAULT_SCOUT_CANDIDATES: Tuple[float, ...] = (450.0, 490.0, 500.0, 510.0,
                                               530.0, 540.0)


def pick_anchor_phi(
    phi_targets_deg: Sequence[float],
    load_fn: Callable[[float], Tuple[float, float]],
    *,
    mode: str = "explicit",
    explicit_phi_deg: float = DEFAULT_EXPLICIT_PHI_ANCHOR_DEG,
    legacy_matched_phis: Optional[Sequence[float]] = None,
    scout_candidates: Optional[Sequence[float]] = None,
    scout_eval_fn: Optional[Callable] = None,
) -> Tuple[float, str, List[float]]:
    """Choose an anchor crank angle.

    Hard rule: never pick global min-load or global max-load (Section 1.4(b)).
    Returns (phi_anchor_deg, reason_str, candidate_list_deg).
    """
    phi_arr = np.asarray(phi_targets_deg, dtype=float)
    loads = np.array([float(np.hypot(*load_fn(float(p)))) for p in phi_arr])
    idx_min = int(np.argmin(loads))
    idx_max = int(np.argmax(loads))
    base_step = float(np.median(np.diff(np.sort(phi_arr)))) if len(phi_arr) > 1 else 10.0

    def _snap_to_grid(p: float) -> float:
        return float(phi_arr[int(np.argmin(np.abs(phi_arr - (float(p) % 720.0))))])

    def _is_safe(p: float) -> bool:
        # Reject candidates within ±1 base step of global min or max load.
        return (abs((p - phi_arr[idx_min] + 360.0) % 720.0 - 360.0) > base_step and
                abs((p - phi_arr[idx_max] + 360.0) % 720.0 - 360.0) > base_step)

    if mode == "explicit":
        phi_a = _snap_to_grid(explicit_phi_deg)
        return phi_a, f"explicit override at {explicit_phi_deg:.1f} deg", [phi_a]

    if mode == "from_legacy_matched_sector":
        if legacy_matched_phis is not None and len(legacy_matched_phis) > 0:
            ordered = sorted(float(p) % 720.0 for p in legacy_matched_phis)
            # Find longest contiguous run on a wrapped 720 grid.
            best_run: List[float] = []
            cur: List[float] = []
            tol = 2.0 * base_step
            for p in ordered:
                if not cur or abs(p - cur[-1]) <= tol:
                    cur.append(p)
                else:
                    if len(cur) > len(best_run):
                        best_run = cur
                    cur = [p]
            if len(cur) > len(best_run):
                best_run = cur
            if best_run:
                mid = best_run[len(best_run) // 2]
                phi_a = _snap_to_grid(mid)
                if _is_safe(phi_a):
                    return (phi_a,
                            f"midpoint of legacy-matched sector "
                            f"({best_run[0]:.1f}-{best_run[-1]:.1f} deg, "
                            f"n={len(best_run)})",
                            [float(p) for p in best_run])
        # Fall through to pragmatic default if legacy info is absent / unsafe.
        phi_a = _snap_to_grid(DEFAULT_EXPLICIT_PHI_ANCHOR_DEG)
        backups = [_snap_to_grid(p) for p in DEFAULT_BACKUP_PHI_LIST]
        return (phi_a,
                "legacy info unavailable; pragmatic default phi=500 deg",
                [phi_a, *backups])

    if mode == "scout_best":
        cands = list(scout_candidates) if scout_candidates else list(DEFAULT_SCOUT_CANDIDATES)
        cands = [_snap_to_grid(p) for p in cands if _is_safe(_snap_to_grid(p))]
        # Medium-load band filter (~25-60 percentile).
        lo = float(np.percentile(loads, 25))
        hi = float(np.percentile(loads, 60))
        filt: List[float] = []
        for p in cands:
            W = float(np.hypot(*load_fn(p)))
            if lo <= W <= hi:
                filt.append(p)
        if not filt:
            filt = cands
        # Optional cheap scout: pick candidate with lowest control residual.
        if scout_eval_fn is not None and len(filt) > 1:
            best_p, best_r = filt[0], float("inf")
            for p in filt:
                Wx, Wy = load_fn(p)
                Wn = float(np.hypot(Wx, Wy))
                eps_s = min(0.7, 0.4 * (Wn / 1000.0) ** 0.25)
                try:
                    m, _, _ = scout_eval_fn(0.0, -eps_s, None)
                    r = math.hypot(m["Fx"] - Wx, m["Fy"] - Wy) / max(Wn, 1e-20)
                except Exception:
                    r = float("inf")
                if r < best_r:
                    best_r, best_p = r, p
            return (best_p,
                    f"scout_best: lowest control residual among "
                    f"{len(filt)} medium-load candidates (r~{best_r:.2e})",
                    filt)
        return (filt[0],
                f"scout_best: first medium-load candidate from {len(filt)} "
                f"(no cheap scout eval supplied)",
                filt)

    raise ValueError(f"unknown anchor mode: {mode!r}")


# ─── Local corrector for anchor stages ─────────────────────────────

def _local_corrector(
    Wa: np.ndarray,
    eval_full: Callable,
    eval_trial: Callable,
    X0: float, Y0: float,
    g_init: Optional[np.ndarray],
    *,
    max_iter: int = 8,
    hard_cap: int = 8,
    tol: float = 5e-3,
    soft_tol: float = 2e-2,
    eps_max: float = 0.92,
    step_cap: float = 0.15,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], int]:
    """Short-budget damped Newton with LM safeguard.

    Trial evaluations (Jacobian probes, line-search points) use eval_trial
    (cheap PS budget). The accepted-step evaluation that updates state uses
    eval_full (anchor-stage PS budget).

    Hard cap: ``hard_cap`` = 8 NL iterations (Section 6.1 of the patch spec).
    Default ``step_cap`` is intentionally larger here (0.15) than in the
    in-branch continuation runner (0.05): anchor solve has to *land on the
    branch*, often from a moderate eps_seed, in ≤ 8 iterations.
    """
    Wn = float(np.linalg.norm(Wa))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)
    g_cur = g_init
    lm_lam = 0.0
    corrector_type = "newton"

    m, P, theta = eval_full(X, Y, g_cur)
    g_cur = _pack_g(P, theta)
    Fx, Fy = m["Fx"], m["Fy"]
    rR = math.hypot(Fx - Wa[0], Fy - Wa[1]) / max(Wn, 1e-20)
    snap = dict(m)
    n_it = 0

    for it in range(min(max_iter, hard_cap)):
        if rR < tol:
            break
        # Jacobian via cheap trial evals.
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            mp, _, _ = eval_trial(X + dX_, Y + dY_, g_cur)
            mn, _, _ = eval_trial(X - dX_, Y - dY_, g_cur)
            J[0, col] = (mp["Fx"] - mn["Fx"]) / (2 * dXY)
            J[1, col] = (mp["Fy"] - mn["Fy"]) / (2 * dXY)
        Rx = Fx - Wa[0]
        Ry = Fy - Wa[1]
        JtJ = J.T @ J
        JtR = J.T @ np.array([Rx, Ry])
        if lm_lam > 0:
            JtJ[0, 0] += lm_lam
            JtJ[1, 1] += lm_lam
            corrector_type = "lm"
        det = JtJ[0, 0] * JtJ[1, 1] - JtJ[0, 1] * JtJ[1, 0]
        if abs(det) < 1e-30:
            lm_lam = max(lm_lam * 10, 1e-3)
            if lm_lam > 1.0:
                break
            continue
        dq = -np.linalg.solve(JtJ, JtR)
        ddX, ddY = float(dq[0]), float(dq[1])
        cap = step_cap / max(abs(ddX), abs(ddY), 1e-20)
        if cap < 1:
            ddX *= cap
            ddY *= cap
        ok = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Xt = X + alpha * ddX
            Yt = Y + alpha * ddY
            if math.hypot(Xt, Yt) >= eps_max:
                continue
            mt, _, _ = eval_trial(Xt, Yt, g_cur)
            rt = math.hypot(mt["Fx"] - Wa[0], mt["Fy"] - Wa[1]) / max(Wn, 1e-20)
            if rt < rR:
                # Accept the step using the heavy-budget evaluator so the
                # state we keep has fully converged PS at this anchor stage.
                mF, PF, thF = eval_full(Xt, Yt, g_cur)
                X, Y = Xt, Yt
                Fx, Fy = mF["Fx"], mF["Fy"]
                rR = math.hypot(Fx - Wa[0], Fy - Wa[1]) / max(Wn, 1e-20)
                g_cur = _pack_g(PF, thF)
                snap = dict(mF)
                ok = True
                n_it += 1
                lm_lam = 0.0
                break
        if not ok:
            lm_lam = max(lm_lam * 10, 1e-3)
            if lm_lam > 1.0:
                break

    eps = math.hypot(X, Y)
    att = math.degrees(math.atan2(Y, X))
    if rR <= tol:
        st = "hard_converged"
    elif rR <= soft_tol:
        st = "soft_converged"
    else:
        st = "failed"
    out = dict(
        X=X, Y=Y, eps=eps, attitude_deg=att,
        h_min=snap["h_min"], p_max=snap["p_max"],
        cav_frac=snap["cav_frac"], friction=snap["friction"],
        Ploss=snap["Ploss"], Qout=snap["Qout"],
        rel_residual=rR, status=st,
        corrector_type=corrector_type,
    )
    return out, g_cur, n_it


def _pack_g(P, theta):
    g = np.where(P > 1e-14, P, theta - 1.0)
    return np.ascontiguousarray(g, dtype=np.float64)


# ─── Smooth anchor: mild load homotopy ─────────────────────────────

def solve_anchor_smooth(
    phi_a_deg: float,
    Wa: np.ndarray,
    eval_factory_smooth: Callable[[str], Callable],
    *,
    X_seed: float = 0.0,
    Y_seed: Optional[float] = None,
    g_seed: Optional[np.ndarray] = None,
    lambda_schedule: Sequence[float] = (0.4, 0.6, 0.8, 1.0),
    fallback_schedule: Sequence[float] = (0.5, 0.75, 1.0),
    eps_max: float = 0.92,
    step_cap: float = 0.20,
    tol: float = 5e-3,
    soft_tol: float = 5e-2,
    log: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[AnchorState], List[Dict[str, Any]]]:
    """Solve smooth anchor by ramping external load magnitude through lambda.

    F_h(X, Y) + lambda * Wa = 0, at fixed phi_a, for lambda in schedule.
    Lambda DOES NOT start from 0 (Section 3.2). Warm-start (X, Y, g) carried
    between stages (Section 3.4).

    Defaults reflect anchor's role as a *landing entry point* (not as a
    production residual): step_cap=0.20 (vs 0.05 for in-branch continuation)
    so 8 NL iters can cover the seed→target distance, soft_tol=5e-2 so a
    landing within ~5% of the load magnitude is acceptable. Continuation
    will then tighten the residual along φ.

    Returns (anchor_state_or_None, lambda_log).
    """
    if log is None:
        log = []
    Wn = float(np.linalg.norm(Wa))
    if Y_seed is None:
        # Section 3.6: cap epsilon_seed to moderate range; lambda=0.4 does
        # not need a high-eps seed. Empirically ~0.20-0.30 lands the first
        # lambda-stage in 3-5 NL iters at moderate loads (1-5 kN).
        eps_s = min(0.30, 0.20 * (Wn / 1000.0) ** 0.25)
        Y_seed = -eps_s
    Wa = np.asarray(Wa, dtype=float)

    schedules = [tuple(lambda_schedule), tuple(fallback_schedule)]
    eval_full_first = eval_factory_smooth("anchor_stage_first")
    eval_full_later = eval_factory_smooth("anchor_stage_later")
    eval_trial = eval_factory_smooth("trial")

    for sched_idx, schedule in enumerate(schedules):
        X, Y = float(X_seed), float(Y_seed)
        g_cur = g_seed
        last_state: Optional[Dict[str, Any]] = None
        ok_all = True
        for stage_idx, lam in enumerate(schedule):
            eval_full = eval_full_first if stage_idx == 0 else eval_full_later
            t0 = time.time()
            d, g_out, nit = _local_corrector(
                lam * Wa, eval_full, eval_trial, X, Y, g_cur,
                max_iter=8, hard_cap=8,
                tol=tol, soft_tol=soft_tol,
                eps_max=eps_max, step_cap=step_cap,
            )
            dt = time.time() - t0
            entry = dict(
                schedule="primary" if sched_idx == 0 else "fallback",
                stage_idx=stage_idx, lambda_=float(lam),
                X=d["X"], Y=d["Y"], eps=d["eps"],
                attitude_deg=d["attitude_deg"],
                rel_residual=d["rel_residual"],
                status=d["status"], nr_iters=nit,
                ps_mode=("anchor_stage_first" if stage_idx == 0
                         else "anchor_stage_later"),
                wall_sec=dt,
            )
            log.append(entry)
            if d["status"] == "failed":
                ok_all = False
                break
            X, Y = d["X"], d["Y"]
            g_cur = g_out
            last_state = d

        if ok_all and last_state is not None:
            return (
                AnchorState(
                    phi_deg=float(phi_a_deg),
                    X=last_state["X"], Y=last_state["Y"],
                    eps=last_state["eps"],
                    attitude_deg=last_state["attitude_deg"],
                    h_min=last_state["h_min"], p_max=last_state["p_max"],
                    cav_frac=last_state["cav_frac"],
                    friction=last_state["friction"],
                    Ploss=last_state["Ploss"], Qout=last_state["Qout"],
                    rel_residual=last_state["rel_residual"],
                    g=g_cur,
                    status=last_state["status"],
                ),
                log,
            )
    return None, log


# ─── Textured anchor: smooth-seeded + optional geometry continuation ─

def solve_anchor_textured(
    phi_a_deg: float,
    Wa: np.ndarray,
    eval_factory_textured: Callable[[str], Callable],
    smooth_anchor: AnchorState,
    *,
    geometry_factory: Optional[Callable[[float, str], Callable]] = None,
    alpha_tex_schedule: Sequence[float] = (0.33, 0.66, 1.0),
    eps_max: float = 0.92,
    step_cap: float = 0.20,
    tol: float = 5e-3,
    soft_tol: float = 5e-2,
    log: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[AnchorState], List[Dict[str, Any]]]:
    """Solve textured anchor seeded by smooth anchor at the same phi_a.

    First attempt: direct textured solve at alpha_tex=1.0 from smooth state.
    Rescue: short geometry continuation alpha_tex_schedule, warm-starting
    (X, Y, g) between stages (Section 4.3). Each stage uses
    geometry_factory(alpha_tex, mode) -> eval_fn.
    """
    if log is None:
        log = []
    Wa = np.asarray(Wa, dtype=float)

    eval_full = eval_factory_textured("anchor_stage_first")
    eval_trial = eval_factory_textured("trial")

    # 1) Direct textured solve from smooth state.
    t0 = time.time()
    d, g_out, nit = _local_corrector(
        Wa, eval_full, eval_trial,
        smooth_anchor.X, smooth_anchor.Y, smooth_anchor.g,
        max_iter=6, hard_cap=8,
        tol=tol, soft_tol=soft_tol,
        eps_max=eps_max, step_cap=step_cap,
    )
    dt = time.time() - t0
    log.append(dict(
        path="direct_from_smooth", alpha_tex=1.0,
        X=d["X"], Y=d["Y"], eps=d["eps"],
        attitude_deg=d["attitude_deg"],
        rel_residual=d["rel_residual"], status=d["status"],
        nr_iters=nit, wall_sec=dt,
    ))
    if d["status"] != "failed":
        return (
            AnchorState(
                phi_deg=float(phi_a_deg),
                X=d["X"], Y=d["Y"], eps=d["eps"],
                attitude_deg=d["attitude_deg"],
                h_min=d["h_min"], p_max=d["p_max"],
                cav_frac=d["cav_frac"], friction=d["friction"],
                Ploss=d["Ploss"], Qout=d["Qout"],
                rel_residual=d["rel_residual"],
                g=g_out, status=d["status"],
            ),
            log,
        )

    # 2) Rescue via geometry continuation (texture amplitude).
    if geometry_factory is None:
        return None, log

    X, Y = smooth_anchor.X, smooth_anchor.Y
    g_cur = smooth_anchor.g
    last: Optional[Dict[str, Any]] = None
    for k, alpha in enumerate(alpha_tex_schedule):
        eval_full_a = geometry_factory(float(alpha), "anchor_stage_later"
                                        if k > 0 else "anchor_stage_first")
        eval_trial_a = geometry_factory(float(alpha), "trial")
        t0 = time.time()
        d, g_out, nit = _local_corrector(
            Wa, eval_full_a, eval_trial_a, X, Y, g_cur,
            max_iter=6, hard_cap=8,
            tol=tol, soft_tol=soft_tol,
            eps_max=eps_max, step_cap=step_cap,
        )
        dt = time.time() - t0
        log.append(dict(
            path="geometry_continuation", alpha_tex=float(alpha),
            stage_idx=k,
            X=d["X"], Y=d["Y"], eps=d["eps"],
            attitude_deg=d["attitude_deg"],
            rel_residual=d["rel_residual"], status=d["status"],
            nr_iters=nit, wall_sec=dt,
        ))
        if d["status"] == "failed":
            return None, log
        X, Y = d["X"], d["Y"]
        g_cur = g_out
        last = d

    if last is None:
        return None, log
    return (
        AnchorState(
            phi_deg=float(phi_a_deg),
            X=last["X"], Y=last["Y"], eps=last["eps"],
            attitude_deg=last["attitude_deg"],
            h_min=last["h_min"], p_max=last["p_max"],
            cav_frac=last["cav_frac"], friction=last["friction"],
            Ploss=last["Ploss"], Qout=last["Qout"],
            rel_residual=last["rel_residual"],
            g=g_cur, status=last["status"],
        ),
        log,
    )


__all__ = [
    "PS_BUDGETS",
    "AnchorState",
    "AnchorReport",
    "DEFAULT_EXPLICIT_PHI_ANCHOR_DEG",
    "DEFAULT_BACKUP_PHI_LIST",
    "DEFAULT_SCOUT_CANDIDATES",
    "pick_anchor_phi",
    "solve_anchor_smooth",
    "solve_anchor_textured",
]
