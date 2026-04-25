"""Continuation cycle runner for Stage I-A.

Natural-parameter continuation in phi_crank with:
  * Secant predictor from 2 previous accepted states
  * Local corrector (damped NR, short budget, LM safeguard)
  * Adaptive subdivision on failure (midpoint stepping stone)
  * Branch-jump detection
  * Rotated-cycle start from anchor
  * Pressure warm-start chain
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ContinuationConfig:
    corrector_max_iter: int = 10
    corrector_tol: float = 5e-3
    corrector_soft_tol: float = 2e-2
    step_cap: float = 0.05
    eps_max: float = 0.92
    # Subdivision
    max_subdiv_depth: int = 4
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


def _make_failed_node(phi, d, nit, depth, pred_type, corrector_type=None):
    return SolvedNode(
        phi_deg=phi, predictor_type=pred_type,
        nr_iters=nit, subdiv_depth=depth, status="failed",
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

    # Mode-aware factories take a mode kwarg; legacy ones take none.
    try:
        eval_fn = eval_factory("accepted_node")
        eval_fn_mid = eval_factory("midpoint_rescue")
        eval_fn_trial = eval_factory("trial")
    except TypeError:
        eval_fn = eval_factory()
        eval_fn_mid = eval_fn
        eval_fn_trial = eval_fn

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
    ordered = (list(range(idx_anchor, len(phi_targets_deg)))
               + list(range(0, idx_anchor)))

    results: List[SolvedNode] = []
    accepted_history: List[Tuple[float, float, float]] = []
    g_prev = anchor_g

    phi0 = phi_targets_deg[ordered[0]]

    if anchor_state is not None:
        # Use externally-solved anchor as node 0. Do NOT re-solve.
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
            status=getattr(anchor_state, "status", "hard_converged"),
            nr_iters=0, predictor_type="anchor_external",
            corrector_type="anchor_solver")
        results.append(node0)
        accepted_history.append((node0.phi_deg, node0.X, node0.Y))
        g_prev = getattr(anchor_state, "g", anchor_g)
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
                                         g_prev, anchor_cfg)
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

    # March through remaining targets using the regular continuation logic.
    # Anchor is already accepted (or fallback-attempted) above; only after
    # that do we engage subdivision / midpoint rescue / branch-jump checks.
    for idx in range(1, len(ordered)):
        target_i = ordered[idx]
        phi_target = phi_targets_deg[target_i]
        Wx, Wy = load_fn(phi_target)
        Wa = np.array([Wx, Wy], dtype=float)

        node, g_out = _solve_with_subdivision(
            phi_target, Wa, eval_fn, load_fn, accepted_history,
            g_prev, cfg, depth=0,
            eval_fn_rescue=eval_fn_mid,
            eval_fn_trial=eval_fn_trial)
        results.append(node)
        if node.status != "failed":
            accepted_history.append((node.phi_deg, node.X, node.Y))
            g_prev = g_out

    return results


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
) -> Tuple[SolvedNode, Optional[np.ndarray]]:
    """Solve at phi_target, with subdivision on failure.

    Returns (node, g_out).
    """
    # Predictor
    if len(history) >= 2:
        phi1, X1, Y1 = history[-2]
        phi2, X2, Y2 = history[-1]
        dphi_prev = phi2 - phi1
        dphi_target = phi_target - phi2
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

    # Corrector — trial-budget probes, full-budget accepted update.
    d, g_out, nit = corrector_solve(Wa, eval_fn, X_pred, Y_pred,
                                      g_prev, cfg,
                                      eval_fn_trial=eval_fn_trial)

    # Branch-jump check
    if d["status"] != "failed":
        if _check_branch_jump(X_pred, Y_pred, d["X"], d["Y"], cfg):
            d["status"] = "failed"

    if d["status"] != "failed":
        node = SolvedNode(
            phi_deg=phi_target, predictor_type=pred_type,
            nr_iters=nit, subdiv_depth=depth,
            corrector_type=d["corrector_type"],
            **{k: d[k] for k in (
                "X", "Y", "eps", "attitude_deg", "h_min", "p_max",
                "cav_frac", "friction", "Ploss", "Qout",
                "rel_residual", "status")})
        return node, g_out

    # --- Subdivision fallback ---

    if depth >= cfg.max_subdiv_depth:
        return _make_failed_node(phi_target, d, nit, depth, pred_type), None

    phi_prev = history[-1][0] if history else phi_target - 20
    dphi = phi_target - phi_prev
    if abs(dphi) < cfg.min_dphi_deg:
        return _make_failed_node(phi_target, d, nit, depth, pred_type), None

    # Step 1: solve midpoint as stepping stone — uses rescue PS budget if
    # caller provided a separate eval_fn_rescue (heavier than accepted_node).
    phi_mid = phi_prev + dphi / 2.0
    Wx_mid, Wy_mid = load_fn(phi_mid)
    Wa_mid = np.array([Wx_mid, Wy_mid], dtype=float)
    eval_mid = eval_fn_rescue if eval_fn_rescue is not None else eval_fn

    mid_node, g_mid = _solve_with_subdivision(
        phi_mid, Wa_mid, eval_mid, load_fn, history,
        g_prev, cfg, depth + 1,
        eval_fn_rescue=eval_fn_rescue,
        eval_fn_trial=eval_fn_trial)

    if mid_node.status != "failed":
        # Midpoint succeeded — use it as stepping stone to target
        history_ext = list(history) + [(phi_mid, mid_node.X, mid_node.Y)]
        target_node, g_target = _solve_with_subdivision(
            phi_target, Wa, eval_fn, load_fn, history_ext,
            g_mid, cfg, depth + 1,
            eval_fn_rescue=eval_fn_rescue,
            eval_fn_trial=eval_fn_trial)
        if target_node.status != "failed":
            return target_node, g_target

    # Step 2: midpoint failed or stepping stone failed — try fresh seed
    Wn = float(np.linalg.norm(Wa))
    eps_s = min(0.85, 0.4 * (Wn / 1000) ** 0.25)
    d2, g2, nit2 = corrector_solve(Wa, eval_mid, 0.0, -eps_s, None, cfg,
                                      eval_fn_trial=eval_fn_trial)
    if d2["status"] != "failed":
        node = SolvedNode(
            phi_deg=phi_target, predictor_type="reseed",
            nr_iters=nit2, subdiv_depth=depth + 1,
            corrector_type=d2["corrector_type"],
            **{k: d2[k] for k in (
                "X", "Y", "eps", "attitude_deg", "h_min", "p_max",
                "cav_frac", "friction", "Ploss", "Qout",
                "rel_residual", "status")})
        return node, g2

    return _make_failed_node(phi_target, d, nit + nit2, depth + 1,
                              pred_type, "reseed_failed"), None
