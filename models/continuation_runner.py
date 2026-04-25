"""Continuation cycle runner for Stage I-A.

Natural-parameter continuation in phi_crank with:
  * Secant predictor from 2 previous accepted states
  * Local corrector (damped NR, short budget, LM safeguard)
  * Wrap-safe phi deltas across the 720° boundary
  * Adaptive subdivision on failure (midpoint stepping stone),
    short by default (max_subdiv_depth=2 — Section 7.4 of basin patch)
  * Midpoint persistence: a successful midpoint is stored as a real
    solved node and added to history (Section 2.3 / 7.2)
  * Bridge nodes: residuals in (2e-2, 5e-2] kept for warm-start /
    predictor but excluded from headline metrics (Section 8)
  * Plateau / wrong-basin / cap-lock detectors (Section 3) and early
    multi-start reseed (Section 4-5) before max_subdiv_depth burns out
  * Per-node and per-segment wall-clock caps (Section 10)
  * Branch-jump detection
  * Rotated-cycle start from anchor; multi-shoot segments orchestrated
    by ``run_continuation_segment``
  * Pressure warm-start chain — never reuse g from a failed/plateau node
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.basin_policy import (
    DROP_STATUSES,
    HMIN_GUARD_DEFAULT,
    METRIC_STATUSES,
    NodeAttemptRecord,
    SeedCandidate,
    STATUS_BRIDGE,
    STATUS_CAPACITY_LIMITED,
    STATUS_FAILED,
    STATUS_METRIC_HARD,
    STATUS_METRIC_SOFT,
    STATUS_TIMEOUT_FAILED,
    TOL_BRIDGE,
    USABLE_FOR_HISTORY,
    WallClockCaps,
    angle_diff_deg,
    build_multistart_seeds,
    classify_node_status,
    detect_capacity_limited,
    detect_fast_wrong_basin,
    detect_plateau_lock,
    eps_expected,
    scout_seeds,
)


@dataclass
class ContinuationConfig:
    corrector_max_iter: int = 10
    corrector_tol: float = 5e-3
    corrector_soft_tol: float = 2e-2
    step_cap: float = 0.05
    eps_max: float = 0.92
    # Subdivision (Section 7.4: lower default 4 → 2).
    max_subdiv_depth: int = 2
    min_dphi_deg: float = 1.5
    # Branch-jump thresholds
    max_eps_jump: float = 0.15
    max_att_jump_deg: float = 25.0
    # LM regularization
    lm_lambda_init: float = 1e-3
    lm_lambda_max: float = 1.0


@dataclass
class SolvedNode:
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
    status: str
    nr_iters: int
    is_midpoint: bool = False
    subdiv_depth: int = 0
    predictor_type: str = "none"
    corrector_type: str = "newton"
    # basin-patch provenance / diagnostics
    detector_triggered: str = ""           # which detector fired (if any)
    reseed_used: bool = False              # multi-start reseed was invoked
    reseed_candidate_rank: int = -1        # 0,1,2,... after scout sort
    g_source: str = "previous"             # nearest_accepted / anchor / none
    ps_budget_mode: str = "accepted_node"
    node_elapsed_sec: float = 0.0
    reason_for_stop: str = ""


def _wrap_dphi(target: float, ref: float, period: float = 720.0) -> float:
    """Shortest signed angular delta target-ref on a wrapped period.

    Without wrap-safety, marching past phi=720° onto phi=0° yields a
    nominal delta of -710° instead of +10°, which makes the secant
    predictor extrapolate ~71 steps ahead and breaks the first wrap
    node every time.
    """
    return ((float(target) - float(ref) + period / 2) % period) - period / 2


def _secant_predict(q1: Tuple[float, float], q2: Tuple[float, float],
                     dphi_prev: float, dphi_target: float
                     ) -> Tuple[float, float]:
    """Secant predictor in (X, Y) space."""
    if abs(dphi_prev) < 1e-12:
        return q2
    s = dphi_target / dphi_prev
    X_pred = q2[0] + s * (q2[0] - q1[0])
    Y_pred = q2[1] + s * (q2[1] - q1[1])
    return X_pred, Y_pred


def _check_branch_jump(X_pred, Y_pred, X_sol, Y_sol, cfg: ContinuationConfig
                        ) -> bool:
    """Return True if solution jumped to a suspicious branch."""
    eps_pred = math.sqrt(X_pred ** 2 + Y_pred ** 2)
    eps_sol = math.sqrt(X_sol ** 2 + Y_sol ** 2)
    if abs(eps_sol - eps_pred) > cfg.max_eps_jump:
        return True
    att_pred = math.degrees(math.atan2(Y_pred, X_pred))
    att_sol = math.degrees(math.atan2(Y_sol, X_sol))
    datt = abs((att_sol - att_pred + 180) % 360 - 180)
    if datt > cfg.max_att_jump_deg:
        return True
    return False


def corrector_solve(
        Wa: np.ndarray,
        eval_fn: Callable,
        X0: float, Y0: float,
        g_init: Optional[np.ndarray],
        cfg: ContinuationConfig,
        eval_fn_trial: Optional[Callable] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], int]:
    """Short-budget damped NR with LM safeguard.

    Section 5.3 of the patch spec: trial evaluations (Jacobian probes,
    line-search trial points) MUST be cheaper than accepted evaluations.
    When ``eval_fn_trial`` is supplied, it is used for those probes; the
    accepted-step evaluation that updates state still uses ``eval_fn``.
    When ``eval_fn_trial`` is None we fall back to the legacy single-budget
    behaviour.

    Returns (result_dict, g_out, n_iters).
    """
    if eval_fn_trial is None:
        eval_fn_trial = eval_fn
    Wn = float(np.linalg.norm(Wa))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)
    g_cur = g_init
    lm_lam = 0.0
    corrector_type = "newton"

    m, P, theta = eval_fn(X, Y, g_cur)
    g_cur_out = _pack_g(P, theta)
    Fx, Fy = m["Fx"], m["Fy"]
    hm, pm, cv, fr, pl, qo = (m["h_min"], m["p_max"], m["cav_frac"],
                                m["friction"], m["Ploss"], m["Qout"])
    rR = math.sqrt((Fx - Wa[0])**2 + (Fy - Wa[1])**2) / max(Wn, 1e-20)
    n_it = 0

    for it in range(cfg.corrector_max_iter):
        if rR < cfg.corrector_tol:
            break
        # Jacobian — trial budget.
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            mp, _, _ = eval_fn_trial(X + dX_, Y + dY_, g_cur)
            mn, _, _ = eval_fn_trial(X - dX_, Y - dY_, g_cur)
            J[0, col] = (mp["Fx"] - mn["Fx"]) / (2 * dXY)
            J[1, col] = (mp["Fy"] - mn["Fy"]) / (2 * dXY)
        Rx = Fx - Wa[0]
        Ry = Fy - Wa[1]
        # LM regularization if needed
        JtJ = J.T @ J
        JtR = J.T @ np.array([Rx, Ry])
        if lm_lam > 0:
            JtJ[0, 0] += lm_lam
            JtJ[1, 1] += lm_lam
            corrector_type = "lm"
        det = JtJ[0, 0] * JtJ[1, 1] - JtJ[0, 1] * JtJ[1, 0]
        if abs(det) < 1e-30:
            lm_lam = max(lm_lam * 10, cfg.lm_lambda_init)
            if lm_lam > cfg.lm_lambda_max:
                break
            continue
        dq = -np.linalg.solve(JtJ, JtR)
        ddX, ddY = float(dq[0]), float(dq[1])
        # Step cap
        cap = cfg.step_cap / max(abs(ddX), abs(ddY), 1e-20)
        if cap < 1:
            ddX *= cap
            ddY *= cap
        # Backtracking — trial budget for screening; accept with full budget.
        ok = False
        for alpha in [1.0, 0.5, 0.25]:
            Xt = X + alpha * ddX
            Yt = Y + alpha * ddY
            if math.sqrt(Xt**2 + Yt**2) >= cfg.eps_max:
                continue
            mt, _, _ = eval_fn_trial(Xt, Yt, g_cur)
            rt = math.sqrt((mt["Fx"] - Wa[0])**2 + (mt["Fy"] - Wa[1])**2) / max(Wn, 1e-20)
            if rt < rR:
                # Re-evaluate this accepted candidate at full budget so the
                # state we keep has fully converged PS at this node.
                mF, PF, thF = eval_fn(Xt, Yt, g_cur)
                X, Y = Xt, Yt
                Fx, Fy = mF["Fx"], mF["Fy"]
                hm, pm, cv, fr, pl, qo = (mF["h_min"], mF["p_max"],
                    mF["cav_frac"], mF["friction"], mF["Ploss"], mF["Qout"])
                rR = math.sqrt((Fx - Wa[0])**2 + (Fy - Wa[1])**2) / max(Wn, 1e-20)
                g_cur_out = _pack_g(PF, thF)
                g_cur = g_cur_out
                ok = True
                n_it += 1
                lm_lam = 0.0
                break
        if not ok:
            lm_lam = max(lm_lam * 10, cfg.lm_lambda_init)
            if lm_lam > cfg.lm_lambda_max:
                break

    eps = math.sqrt(X**2 + Y**2)
    att = math.degrees(math.atan2(Y, X))
    if rR <= cfg.corrector_tol:
        st = "hard_converged"
    elif rR <= cfg.corrector_soft_tol:
        st = "soft_converged"
    else:
        st = "failed"
    return dict(X=X, Y=Y, eps=eps, attitude_deg=att,
                h_min=hm, p_max=pm, cav_frac=cv, friction=fr,
                Ploss=pl, Qout=qo, rel_residual=rR,
                status=st, corrector_type=corrector_type), g_cur_out, n_it


def _pack_g(P, theta):
    g = np.where(P > 1e-14, P, theta - 1.0)
    return np.ascontiguousarray(g, dtype=np.float64)


def _classify_solved_dict(
    d: Dict[str, Any],
    *,
    timed_out: bool = False,
    plateau_locked: bool = False,
    eps_max: float = 0.92,
    hmin_guard: float = HMIN_GUARD_DEFAULT,
) -> str:
    """Map raw corrector dict + flags to basin-patch status string."""
    return classify_node_status(
        rel_residual=float(d["rel_residual"]),
        eps=float(d["eps"]),
        h_min_m=float(d["h_min"]),
        eps_max=eps_max,
        hmin_guard=hmin_guard,
        timed_out=timed_out,
        plateau_locked=plateau_locked,
        corrector_failed=(d.get("status", "") == "failed"),
    )


def _make_failed_node(phi, d, nit, depth, pred_type, corrector_type=None):
    return SolvedNode(
        phi_deg=phi, predictor_type=pred_type,
        nr_iters=nit, subdiv_depth=depth, status=STATUS_FAILED,
        corrector_type=corrector_type or d.get("corrector_type", "newton"),
        X=d["X"], Y=d["Y"], eps=d["eps"],
        attitude_deg=d["attitude_deg"],
        h_min=d["h_min"], p_max=d["p_max"],
        cav_frac=d["cav_frac"], friction=d["friction"],
        Ploss=d["Ploss"], Qout=d["Qout"],
        rel_residual=d["rel_residual"])


def run_continuation_cycle(
        phi_targets_deg: List[float],
        load_fn: Callable[[float], Tuple[float, float]],
        eval_factory: Callable,
        cfg: ContinuationConfig = None,
        anchor_X: float = 0.0,
        anchor_Y: float = -0.4,
        anchor_g: Optional[np.ndarray] = None,
        phi_anchor_deg: float = None,
        anchor_state: Optional[Any] = None,
        node_callback: Optional[Callable] = None,
        caps: Optional[WallClockCaps] = None,
        direction: str = "forward",
        max_steps: Optional[int] = None,
        eps_max: float = 0.92,
        hmin_guard: float = HMIN_GUARD_DEFAULT,
        segment_id: int = 0,
) -> List[SolvedNode]:
    """Run continuation over cycle.

    ``eval_factory`` may be either:
      * legacy zero-arg: ``eval_factory()`` -> eval_fn(X, Y, g_init), or
      * mode-aware:      ``eval_factory(mode_str)`` -> eval_fn(X, Y, g_init).

    Anchor policy: this runner does NOT solve the anchor itself anymore.
    Callers MUST supply an accepted anchor via ``anchor_state`` (an object
    with attributes phi_deg, X, Y, eps, attitude_deg, h_min, p_max,
    cav_frac, friction, Ploss, Qout, rel_residual, g, status). When
    ``anchor_state`` is provided, the runner uses it as node 0 and only
    marches the remaining angles through the regular predictor/corrector
    pipeline.

    The legacy fallback (anchor_state=None, generic seed + subdivision-style
    corrector solve at the anchor angle) is retained for backwards compat
    with non-Stage I-A users only and is NOT recommended for Stage I-A —
    that path violates the anchor-policy hard rules of the patch spec.
    """
    if cfg is None:
        cfg = ContinuationConfig()
    if caps is None:
        caps = WallClockCaps()

    # Mode-aware factories take a mode kwarg; legacy ones take none.
    try:
        eval_fn = eval_factory("accepted_node")
        eval_fn_mid = eval_factory("midpoint_rescue")
        eval_fn_trial = eval_factory("trial")
        try:
            eval_fn_scout = eval_factory("scout")
        except (KeyError, ValueError):
            eval_fn_scout = eval_fn_trial
    except TypeError:
        eval_fn = eval_factory()
        eval_fn_mid = eval_fn
        eval_fn_trial = eval_fn
        eval_fn_scout = eval_fn

    # Rotate cycle to start at anchor
    if phi_anchor_deg is None:
        phi_anchor_deg = (anchor_state.phi_deg if anchor_state is not None
                          else phi_targets_deg[0])
    idx_anchor = 0
    min_dist = 1e9
    for i, p in enumerate(phi_targets_deg):
        d = abs(p - phi_anchor_deg)
        if d < min_dist:
            min_dist = d
            idx_anchor = i
    n_targets = len(phi_targets_deg)
    if direction == "forward":
        ordered = (list(range(idx_anchor, n_targets))
                   + list(range(0, idx_anchor)))
    elif direction == "backward":
        ordered = ([idx_anchor] + list(range(idx_anchor - 1, -1, -1))
                   + list(range(n_targets - 1, idx_anchor, -1)))
    else:
        raise ValueError(f"unknown direction {direction!r}")
    if max_steps is not None and max_steps > 0:
        ordered = ordered[:max_steps + 1]  # +1 for the anchor itself

    results: List[SolvedNode] = []
    accepted_history: List[Tuple[float, float, float]] = []
    g_prev = anchor_g

    phi0 = phi_targets_deg[ordered[0]]

    if anchor_state is not None:
        # Use externally-solved anchor as node 0. Do NOT re-solve. Status
        # is reclassified through the basin taxonomy so headline/coverage
        # metrics use a single vocabulary.
        anchor_status = classify_node_status(
            rel_residual=float(anchor_state.rel_residual),
            eps=float(anchor_state.eps),
            h_min_m=float(anchor_state.h_min),
            eps_max=eps_max,
            hmin_guard=hmin_guard,
        )
        node0 = SolvedNode(
            phi_deg=float(anchor_state.phi_deg),
            X=anchor_state.X, Y=anchor_state.Y,
            eps=anchor_state.eps,
            attitude_deg=anchor_state.attitude_deg,
            h_min=anchor_state.h_min, p_max=anchor_state.p_max,
            cav_frac=anchor_state.cav_frac,
            friction=anchor_state.friction,
            Ploss=anchor_state.Ploss, Qout=anchor_state.Qout,
            rel_residual=anchor_state.rel_residual,
            status=anchor_status,
            nr_iters=0, predictor_type="anchor_external",
            corrector_type="anchor_solver",
            g_source="anchor")
        results.append(node0)
        accepted_history.append((node0.phi_deg, node0.X, node0.Y))
        g_prev = getattr(anchor_state, "g", anchor_g)
        if node_callback is not None:
            node_callback(node0, 0, len(ordered))
    else:
        # Legacy path (kept for non-Stage I-A callers).
        Wx0, Wy0 = load_fn(phi0)
        Wa0 = np.array([Wx0, Wy0], dtype=float)
        eps_seed = min(0.85, 0.4 * (float(np.linalg.norm(Wa0)) / 1000) ** 0.25)
        if abs(anchor_X) < 1e-6 and abs(anchor_Y + 0.4) < 1e-6:
            anchor_X, anchor_Y = 0.0, -eps_seed
        anchor_cfg = ContinuationConfig(
            corrector_max_iter=cfg.corrector_max_iter * 3,
            corrector_tol=cfg.corrector_tol,
            corrector_soft_tol=cfg.corrector_soft_tol,
            step_cap=cfg.step_cap,
            eps_max=cfg.eps_max)
        d0, g0, nit0 = corrector_solve(Wa0, eval_fn, anchor_X, anchor_Y,
                                         g_prev, anchor_cfg,
                                         eval_fn_trial=eval_fn_trial)
        node0 = SolvedNode(
            phi_deg=phi0, predictor_type="seed_legacy",
            nr_iters=nit0, **{k: d0[k] for k in (
                "X", "Y", "eps", "attitude_deg", "h_min", "p_max",
                "cav_frac", "friction", "Ploss", "Qout",
                "rel_residual", "status")},
            corrector_type=d0["corrector_type"])
        results.append(node0)
        if d0["status"] != "failed":
            accepted_history.append((phi0, d0["X"], d0["Y"]))
            g_prev = g0
        if node_callback is not None:
            node_callback(node0, 0, len(ordered))

    # ── Basin-aware march ──
    # Anchor is already accepted; only after that do we engage subdivision,
    # plateau detection, segment-stop and reseed.
    anchor_W = None
    anchor_load_angle = None
    if anchor_state is not None:
        wx_a, wy_a = load_fn(float(anchor_state.phi_deg))
        anchor_W = float(math.hypot(wx_a, wy_a))
        anchor_load_angle = math.degrees(math.atan2(wy_a, wx_a))
    anchor_g_for_reseed = (getattr(anchor_state, "g", None)
                            if anchor_state is not None else anchor_g)

    target_records: List[NodeAttemptRecord] = []
    segment_fail_streak = 0
    segment_t0 = time.time()

    for idx in range(1, len(ordered)):
        # Segment wall-clock cap.
        if (time.time() - segment_t0) > caps.segment_sec:
            stop_node = SolvedNode(
                phi_deg=phi_targets_deg[ordered[idx]],
                X=0.0, Y=0.0, eps=0.0, attitude_deg=0.0,
                h_min=0.0, p_max=0.0, cav_frac=0.0, friction=0.0,
                Ploss=0.0, Qout=0.0, rel_residual=float("inf"),
                status=STATUS_TIMEOUT_FAILED, nr_iters=0,
                predictor_type="segment_cap",
                reason_for_stop="segment_wall_clock")
            results.append(stop_node)
            if node_callback is not None:
                node_callback(stop_node, idx, len(ordered))
            break

        target_i = ordered[idx]
        phi_target = phi_targets_deg[target_i]
        Wx, Wy = load_fn(phi_target)
        Wa = np.array([Wx, Wy], dtype=float)

        node_t0 = time.time()
        midpoints_collected: List[SolvedNode] = []
        node, g_out = _solve_with_subdivision(
            phi_target, Wa, eval_fn, load_fn, accepted_history,
            g_prev, cfg, depth=0,
            eval_fn_rescue=eval_fn_mid,
            eval_fn_trial=eval_fn_trial,
            eval_fn_scout=eval_fn_scout,
            anchor_g=anchor_g_for_reseed,
            anchor_W=anchor_W,
            anchor_load_angle_deg=anchor_load_angle,
            caps=caps,
            node_start_time=node_t0,
            eps_max=eps_max, hmin_guard=hmin_guard,
            midpoints_collected=midpoints_collected,
        )

        # Persist successful midpoints into history + result list.
        for mn in midpoints_collected:
            if mn.status in USABLE_FOR_HISTORY:
                results.append(mn)
                accepted_history.append((mn.phi_deg, mn.X, mn.Y))
                if node_callback is not None:
                    node_callback(mn, idx, len(ordered))

        results.append(node)

        # History + g warm-start: only metric/bridge nodes (Section 2 / 6.1).
        if node.status in USABLE_FOR_HISTORY:
            accepted_history.append((node.phi_deg, node.X, node.Y))
            g_prev = g_out
            segment_fail_streak = 0
        else:
            segment_fail_streak += 1
            # NEVER reuse g from failed/plateau/capacity-limited node.

        # Plateau detector on rolling base-target attempts.
        target_records.append(NodeAttemptRecord(
            phi_deg=node.phi_deg,
            W_N=float(math.hypot(Wx, Wy)),
            load_angle_deg=math.degrees(math.atan2(Wy, Wx)),
            eps=node.eps, attitude_deg=node.attitude_deg,
            rel_residual=node.rel_residual,
            status=node.status,
            elapsed_sec=node.node_elapsed_sec,
        ))
        plateau = detect_plateau_lock(target_records)
        if plateau:
            node.detector_triggered = (
                node.detector_triggered + ";plateau_lock"
            ).strip(";")

        if node_callback is not None:
            node_callback(node, idx, len(ordered))

        # Section 9.3: segment stop triggers.
        stop_reason = ""
        if segment_fail_streak >= 3:
            stop_reason = "segment_fail_streak_3"
        elif plateau:
            stop_reason = "plateau_lock"
        elif (idx >= 4 and not any(
                r.status in USABLE_FOR_HISTORY
                for r in target_records[-3:])):
            stop_reason = "no_new_metric_in_last_3"

        if stop_reason:
            node.reason_for_stop = (
                node.reason_for_stop + ";" + stop_reason
            ).strip(";")
            break

    return results


def run_continuation_segment(
    phi_targets_deg: List[float],
    load_fn: Callable[[float], Tuple[float, float]],
    eval_factory: Callable,
    anchor_state: Any,
    *,
    cfg: Optional[ContinuationConfig] = None,
    caps: Optional[WallClockCaps] = None,
    direction: str = "forward",
    max_steps: Optional[int] = None,
    eps_max: float = 0.92,
    hmin_guard: float = HMIN_GUARD_DEFAULT,
    segment_id: int = 0,
    node_callback: Optional[Callable] = None,
) -> List[SolvedNode]:
    """Multi-shoot continuation segment from a single anchor.

    Section 9: spawn one segment per anchor, march in a chosen direction
    until a segment-stop trigger fires, then return the partial node list.
    The orchestrator (run_diesel_stage1._run_continuation) is responsible
    for iterating over the anchor pool and merging segments per phi.
    """
    return run_continuation_cycle(
        phi_targets_deg, load_fn, eval_factory,
        cfg=cfg,
        anchor_g=getattr(anchor_state, "g", None),
        phi_anchor_deg=getattr(anchor_state, "phi_deg", None),
        anchor_state=anchor_state,
        node_callback=node_callback,
        caps=caps,
        direction=direction,
        max_steps=max_steps,
        eps_max=eps_max,
        hmin_guard=hmin_guard,
        segment_id=segment_id,
    )


def _node_from_solved(
    phi: float, d: Dict[str, Any], nit: int, depth: int, pred_type: str,
    *, status_override: Optional[str] = None,
    is_midpoint: bool = False,
    detector_triggered: str = "",
    reseed_used: bool = False,
    reseed_candidate_rank: int = -1,
    g_source: str = "previous",
    ps_budget_mode: str = "accepted_node",
    node_elapsed_sec: float = 0.0,
    reason_for_stop: str = "",
    plateau_locked: bool = False,
    eps_max: float = 0.92,
    hmin_guard: float = HMIN_GUARD_DEFAULT,
) -> SolvedNode:
    """Build a SolvedNode from a corrector-output dict, classifying status."""
    st = status_override or _classify_solved_dict(
        d, plateau_locked=plateau_locked,
        eps_max=eps_max, hmin_guard=hmin_guard,
    )
    return SolvedNode(
        phi_deg=phi, predictor_type=pred_type,
        nr_iters=nit, subdiv_depth=depth, status=st,
        is_midpoint=is_midpoint,
        corrector_type=d.get("corrector_type", "newton"),
        X=d["X"], Y=d["Y"], eps=d["eps"],
        attitude_deg=d["attitude_deg"],
        h_min=d["h_min"], p_max=d["p_max"],
        cav_frac=d["cav_frac"], friction=d["friction"],
        Ploss=d["Ploss"], Qout=d["Qout"],
        rel_residual=d["rel_residual"],
        detector_triggered=detector_triggered,
        reseed_used=reseed_used,
        reseed_candidate_rank=reseed_candidate_rank,
        g_source=g_source,
        ps_budget_mode=ps_budget_mode,
        node_elapsed_sec=node_elapsed_sec,
        reason_for_stop=reason_for_stop,
    )


def _try_multistart_reseed(
    phi_target: float,
    Wa: np.ndarray,
    cfg: ContinuationConfig,
    *,
    eval_fn: Callable,
    eval_fn_scout: Callable,
    eval_fn_trial: Callable,
    nearest_g: Optional[np.ndarray] = None,
    nearest_phi: Optional[float] = None,
    nearest_X: Optional[float] = None,
    nearest_Y: Optional[float] = None,
    anchor_g: Optional[np.ndarray] = None,
    anchor_W: Optional[float] = None,
    anchor_load_angle_deg: Optional[float] = None,
    keep_top: int = 3,
    eps_max: float = 0.92,
    hmin_guard: float = HMIN_GUARD_DEFAULT,
    detector_tag: str = "reseed",
) -> Tuple[Optional[SolvedNode], Optional[np.ndarray]]:
    """Section 5: multi-start reseed of a single target.

    Cheap-eval scout selects best ``keep_top`` candidates, then a short
    corrector (current cfg) is run on them. The first successful (metric
    or bridge) result is returned.
    """
    Wn = float(np.linalg.norm(Wa))
    load_angle = math.degrees(math.atan2(float(Wa[1]), float(Wa[0])))
    nearest_dXY = 1.0
    if nearest_X is not None and nearest_Y is not None:
        nearest_dXY = float(math.hypot(nearest_X, nearest_Y))
    nearest_dphi = (angle_diff_deg(phi_target, nearest_phi)
                     if nearest_phi is not None else 360.0)
    anchor_diff = (angle_diff_deg(load_angle, anchor_load_angle_deg)
                    if anchor_load_angle_deg is not None else None)

    cands = build_multistart_seeds(
        Wa,
        eps_max=eps_max,
        nearest_accepted_g=nearest_g,
        nearest_accepted_phi_diff_deg=float(nearest_dphi),
        nearest_accepted_dXY=float(nearest_dXY),
        anchor_g=anchor_g,
        W_anchor=anchor_W,
        load_angle_diff_to_anchor_deg=anchor_diff,
    )
    scored = scout_seeds(cands, Wa, eval_fn_scout, keep_top=keep_top)

    best_node: Optional[SolvedNode] = None
    best_g: Optional[np.ndarray] = None
    for rank, (cand, _r) in enumerate(scored):
        t0 = time.time()
        d, g_out, nit = corrector_solve(
            Wa, eval_fn, cand.X, cand.Y, cand.g_init, cfg,
            eval_fn_trial=eval_fn_trial)
        dt = time.time() - t0
        node = _node_from_solved(
            phi_target, d, nit, depth=0, pred_type="reseed",
            reseed_used=True,
            reseed_candidate_rank=rank,
            g_source=cand.g_source,
            ps_budget_mode="accepted_node",
            node_elapsed_sec=dt,
            detector_triggered=detector_tag,
            eps_max=eps_max, hmin_guard=hmin_guard,
        )
        if node.status in USABLE_FOR_HISTORY:
            return node, g_out
        if best_node is None or node.rel_residual < best_node.rel_residual:
            best_node, best_g = node, g_out
    return best_node, best_g


def _solve_with_subdivision(
        phi_target: float,
        Wa: np.ndarray,
        eval_fn: Callable,
        load_fn: Callable[[float], Tuple[float, float]],
        history: List[Tuple[float, float, float]],
        g_prev: Optional[np.ndarray],
        cfg: ContinuationConfig,
        depth: int,
        eval_fn_rescue: Optional[Callable] = None,
        eval_fn_trial: Optional[Callable] = None,
        eval_fn_scout: Optional[Callable] = None,
        anchor_g: Optional[np.ndarray] = None,
        anchor_W: Optional[float] = None,
        anchor_load_angle_deg: Optional[float] = None,
        caps: Optional[WallClockCaps] = None,
        node_start_time: Optional[float] = None,
        eps_max: float = 0.92,
        hmin_guard: float = HMIN_GUARD_DEFAULT,
        midpoints_collected: Optional[List[SolvedNode]] = None,
) -> Tuple[SolvedNode, Optional[np.ndarray]]:
    """Solve at phi_target with bounded subdivision and basin-aware rescue.

    Returns (node, g_out). ``midpoints_collected``, when supplied, receives
    every successful midpoint solved as a side-product so the caller can
    persist them to history and CSV (Section 2.3 / 7.2).
    """
    if caps is None:
        caps = WallClockCaps()
    if node_start_time is None:
        node_start_time = time.time()

    # Predictor — wrap-safe phi differences (cycle wraps at 720 deg).
    if len(history) >= 2:
        phi1, X1, Y1 = history[-2]
        phi2, X2, Y2 = history[-1]
        dphi_prev = _wrap_dphi(phi2, phi1)
        dphi_target = _wrap_dphi(phi_target, phi2)
        X_pred, Y_pred = _secant_predict((X1, Y1), (X2, Y2),
                                          dphi_prev, dphi_target)
        pred_type = "secant"
    elif len(history) >= 1:
        _, X_pred, Y_pred = history[-1]
        pred_type = "previous"
    else:
        Wn = float(np.linalg.norm(Wa))
        eps_s = min(0.85, 0.4 * (Wn / 1000) ** 0.25)
        X_pred, Y_pred = 0.0, -eps_s
        pred_type = "seed"

    # Clip predictor to eps_max
    eps_pred = math.sqrt(X_pred**2 + Y_pred**2)
    if eps_pred >= cfg.eps_max:
        scale = (cfg.eps_max - 0.01) / eps_pred
        X_pred *= scale
        Y_pred *= scale

    # ── Corrector ──
    t_corr0 = time.time()
    d, g_out, nit = corrector_solve(Wa, eval_fn, X_pred, Y_pred,
                                      g_prev, cfg,
                                      eval_fn_trial=eval_fn_trial)
    dt_corr = time.time() - t_corr0

    # Branch-jump check (only matters if corrector reported success).
    if d.get("status") != "failed":
        if _check_branch_jump(X_pred, Y_pred, d["X"], d["Y"], cfg):
            d["status"] = "failed"

    timed_out_node = (time.time() - node_start_time) > caps.node_hard_sec

    node = _node_from_solved(
        phi_target, d, nit, depth, pred_type,
        status_override=(STATUS_TIMEOUT_FAILED if timed_out_node else None),
        node_elapsed_sec=time.time() - node_start_time,
        ps_budget_mode="accepted_node",
        eps_max=eps_max, hmin_guard=hmin_guard,
    )

    if node.status in USABLE_FOR_HISTORY:
        return node, g_out

    # ── Fast wrong-basin reseed (Section 3.2 / 4.1) ──
    Wn = float(np.linalg.norm(Wa))
    fake_rec = NodeAttemptRecord(
        phi_deg=phi_target, W_N=Wn,
        load_angle_deg=math.degrees(math.atan2(float(Wa[1]), float(Wa[0]))),
        eps=node.eps, attitude_deg=node.attitude_deg,
        rel_residual=node.rel_residual, status=node.status,
    )
    fast_basin = detect_fast_wrong_basin(fake_rec, W_percentile_70=Wn * 1.5)
    if (fast_basin and eval_fn_scout is not None
            and (time.time() - node_start_time) < caps.node_hard_sec):
        nearest_phi = history[-1][0] if history else None
        nearest_X = history[-1][1] if history else None
        nearest_Y = history[-1][2] if history else None
        rs_node, rs_g = _try_multistart_reseed(
            phi_target, Wa, cfg,
            eval_fn=eval_fn,
            eval_fn_scout=eval_fn_scout,
            eval_fn_trial=eval_fn_trial,
            nearest_g=g_prev,
            nearest_phi=nearest_phi,
            nearest_X=nearest_X, nearest_Y=nearest_Y,
            anchor_g=anchor_g, anchor_W=anchor_W,
            anchor_load_angle_deg=anchor_load_angle_deg,
            eps_max=eps_max, hmin_guard=hmin_guard,
            detector_tag="fast_wrong_basin",
        )
        if rs_node is not None and rs_node.status in USABLE_FOR_HISTORY:
            return rs_node, rs_g
        node.detector_triggered = "fast_wrong_basin"

    # ── Subdivision (Section 7) ──
    if depth >= cfg.max_subdiv_depth:
        node.reason_for_stop = "max_subdiv_depth"
        return node, None
    if (time.time() - node_start_time) > caps.node_hard_sec:
        node.status = STATUS_TIMEOUT_FAILED
        node.reason_for_stop = "node_hard_cap"
        return node, None

    phi_prev = history[-1][0] if history else phi_target - 20
    dphi = _wrap_dphi(phi_target, phi_prev)
    if abs(dphi) < cfg.min_dphi_deg:
        node.reason_for_stop = "min_dphi"
        return node, None

    phi_mid = (phi_prev + dphi / 2.0) % 720.0
    Wx_mid, Wy_mid = load_fn(phi_mid)
    Wa_mid = np.array([Wx_mid, Wy_mid], dtype=float)
    eval_mid = eval_fn_rescue if eval_fn_rescue is not None else eval_fn

    mid_node, g_mid = _solve_with_subdivision(
        phi_mid, Wa_mid, eval_mid, load_fn, history,
        g_prev, cfg, depth + 1,
        eval_fn_rescue=eval_fn_rescue,
        eval_fn_trial=eval_fn_trial,
        eval_fn_scout=eval_fn_scout,
        anchor_g=anchor_g, anchor_W=anchor_W,
        anchor_load_angle_deg=anchor_load_angle_deg,
        caps=caps,
        node_start_time=node_start_time,
        eps_max=eps_max, hmin_guard=hmin_guard,
        midpoints_collected=midpoints_collected,
    )
    mid_node.is_midpoint = True
    if mid_node.status in USABLE_FOR_HISTORY:
        # Section 2.3 / 7.2: persist midpoint as a real node.
        if midpoints_collected is not None:
            midpoints_collected.append(mid_node)
        history_ext = list(history) + [(phi_mid, mid_node.X, mid_node.Y)]
        target_node, g_target = _solve_with_subdivision(
            phi_target, Wa, eval_fn, load_fn, history_ext,
            g_mid, cfg, depth + 1,
            eval_fn_rescue=eval_fn_rescue,
            eval_fn_trial=eval_fn_trial,
            eval_fn_scout=eval_fn_scout,
            anchor_g=anchor_g, anchor_W=anchor_W,
            anchor_load_angle_deg=anchor_load_angle_deg,
            caps=caps,
            node_start_time=node_start_time,
            eps_max=eps_max, hmin_guard=hmin_guard,
            midpoints_collected=midpoints_collected,
        )
        if target_node.status in USABLE_FOR_HISTORY:
            return target_node, g_target
        # Section 7.3: if midpoint succeeded but target still fails and
        # depth grew without residual improvement, stop.
        if target_node.rel_residual >= node.rel_residual * 0.9:
            target_node.reason_for_stop = "midpoint_no_improvement"
            return target_node, None

    # Final cheap rescue: short corrector with cold reseed at depth+1.
    if (time.time() - node_start_time) > caps.node_hard_sec:
        node.status = STATUS_TIMEOUT_FAILED
        node.reason_for_stop = "node_hard_cap_after_subdiv"
        return node, None
    eps_s = min(0.85, 0.4 * (Wn / 1000) ** 0.25)
    d2, g2, nit2 = corrector_solve(Wa, eval_mid, 0.0, -eps_s, None, cfg,
                                      eval_fn_trial=eval_fn_trial)
    timed_out2 = (time.time() - node_start_time) > caps.node_hard_sec
    node2 = _node_from_solved(
        phi_target, d2, nit2, depth + 1, pred_type="reseed",
        status_override=(STATUS_TIMEOUT_FAILED if timed_out2 else None),
        reseed_used=True,
        reseed_candidate_rank=0,
        g_source="none",
        ps_budget_mode="midpoint_rescue",
        node_elapsed_sec=time.time() - node_start_time,
        eps_max=eps_max, hmin_guard=hmin_guard,
    )
    if node2.status in USABLE_FOR_HISTORY:
        return node2, g2
    return node2, None
