"""Deterministic objective for co-design (ТЗ §7).

Никаких скрытых глобальных состояний. `compute_J_eps(metrics)` ─ чистая
функция, возвращает один и тот же float для одного и того же dict. Тест
T8 защищает эту инвариантность.

Hard gates:
  * per-ε texture fail       → HARD_TEXTURE_FAIL_PER_EPS
  * profile-level screen_fail → ≥ SCREEN_FAIL_AT_EPS_FAIL_COUNT fails
  * equilibrium useful gate  → EQ_USEFUL_GATES
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .coexp_schema import (
    J_EPS_COEFFS, J_EQ_COEFFS, J_SCREEN_WEIGHTS,
    HARD_TEXTURE_FAIL_PER_EPS, SCREEN_FAIL_AT_EPS_FAIL_COUNT,
    EQ_USEFUL_GATES,
)


def compute_incremental_ratios(metrics_smooth: Dict[str, float],
                                metrics_tex: Dict[str, float]
                                ) -> Dict[str, float]:
    """Вернуть (h_r, p_r, f_r, c_d) из per-case metrics."""
    eps_denom = 1e-30
    return dict(
        h_r=float(metrics_tex["h_min"] / max(metrics_smooth["h_min"], eps_denom)),
        p_r=float(metrics_tex["p_max"] / max(metrics_smooth["p_max"], eps_denom)),
        f_r=float(metrics_tex["friction"] / max(metrics_smooth["friction"], eps_denom)),
        c_d=float(metrics_tex["cav_frac"] - metrics_smooth["cav_frac"]),
    )


def per_eps_texture_fail(ratios: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Hard per-ε fail по ТЗ §7.3. Возвращает (fail, reasons)."""
    reasons: List[str] = []
    lim = HARD_TEXTURE_FAIL_PER_EPS
    if ratios["h_r"] < lim["h_r_min"]:
        reasons.append(f"h_r={ratios['h_r']:.4f} < {lim['h_r_min']}")
    if ratios["p_r"] > lim["p_r_max"]:
        reasons.append(f"p_r={ratios['p_r']:.4f} > {lim['p_r_max']}")
    if ratios["f_r"] > lim["f_r_max"]:
        reasons.append(f"f_r={ratios['f_r']:.4f} > {lim['f_r_max']}")
    if ratios["c_d"] > lim["c_d_max"]:
        reasons.append(f"c_d={ratios['c_d']:.4f} > {lim['c_d_max']}")
    return (len(reasons) > 0), reasons


def _safe_ln(x: float) -> float:
    if x <= 0.0:
        # Log of non-positive ratio — treat as deep penalty. Consistent
        # value for T8 determinism.
        return -1e3
    return math.log(float(x))


def compute_J_eps(ratios: Dict[str, float]) -> float:
    """Per-ε score (ТЗ §7.2)."""
    c = J_EPS_COEFFS
    h_r = float(ratios["h_r"])
    p_r = float(ratios["p_r"])
    f_r = float(ratios["f_r"])
    c_d = float(ratios["c_d"])
    return float(
        c["h"] * _safe_ln(h_r)
        - c["p"] * _safe_ln(p_r)
        - c["f"] * _safe_ln(f_r)
        - c["cav"] * max(0.0, c_d)
    )


def compute_J_eq(ratios: Dict[str, float]) -> float:
    """Equilibrium-phase score (ТЗ §7.5)."""
    c = J_EQ_COEFFS
    h_r = float(ratios["h_r"])
    p_r = float(ratios["p_r"])
    f_r = float(ratios["f_r"])
    c_d = float(ratios["c_d"])
    return float(
        c["h"] * _safe_ln(h_r)
        - c["p"] * _safe_ln(p_r)
        - c["f"] * _safe_ln(f_r)
        - c["cav"] * max(0.0, c_d)
    )


def compute_J_screen(J_per_eps: Dict[float, float]) -> float:
    """Aggregate screening score (ТЗ §7.4).

    Parameters
    ----------
    J_per_eps : dict {eps: J_eps}
        Must contain keys exactly from J_SCREEN_WEIGHTS (0.30, 0.40, 0.50).
    """
    total = 0.0
    for eps, w in J_SCREEN_WEIGHTS.items():
        if eps not in J_per_eps:
            raise KeyError(f"missing J_eps for eps={eps}")
        total += float(w) * float(J_per_eps[eps])
    return float(total)


def classify_profile(J_per_eps: Dict[float, float],
                      ratios_per_eps: Dict[float, Dict[str, float]]
                      ) -> Dict[str, Any]:
    """Один dict с итогом по профилю:
      * screen_fail : bool
      * J_screen    : float (NaN если screen_fail)
      * n_eps_fail  : int
      * fail_reasons: dict {eps: [reasons]}
    """
    fail_reasons: Dict[float, List[str]] = {}
    n_fail = 0
    for eps, ratios in ratios_per_eps.items():
        fail, reasons = per_eps_texture_fail(ratios)
        if fail:
            n_fail += 1
            fail_reasons[float(eps)] = reasons
    screen_fail = (n_fail >= SCREEN_FAIL_AT_EPS_FAIL_COUNT)
    if screen_fail:
        J_screen = float("nan")
    else:
        J_screen = compute_J_screen(J_per_eps)
    return dict(
        screen_fail=bool(screen_fail),
        J_screen=float(J_screen),
        n_eps_fail=int(n_fail),
        fail_reasons={str(k): v for k, v in fail_reasons.items()},
        J_per_eps={str(k): float(v) for k, v in J_per_eps.items()},
    )


def equilibrium_useful(ratios: Dict[str, float]) -> bool:
    """Gate «physically useful» (ТЗ §7.5).

    Возвращает True iff все четыре условия одновременно:
      h_r > 1.005,  p_r <= 1.000,  f_r <= 1.02,  c_d <= 0.02
    """
    g = EQ_USEFUL_GATES
    return (
        float(ratios["h_r"]) > g["h_r_min_excl"]
        and float(ratios["p_r"]) <= g["p_r_max"]
        and float(ratios["f_r"]) <= g["f_r_max"]
        and float(ratios["c_d"]) <= g["c_d_max"]
    )


__all__ = [
    "compute_incremental_ratios",
    "per_eps_texture_fail",
    "compute_J_eps",
    "compute_J_eq",
    "compute_J_screen",
    "classify_profile",
    "equilibrium_useful",
]
