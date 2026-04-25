"""Stage I-A continuation basin policy.

Pure-pipeline module. Solver interface unchanged.

Provides:
  * Node status taxonomy (Section 1) — metric_hard / metric_soft / bridge /
    failed / timeout_failed / capacity_limited_fullfilm.
  * Plateau / lock-in detectors (Section 3) — fast wrong-basin, rolling
    3-node and 4-node plateau, cap-lock / capacity-limited.
  * Multi-start reseed candidate generator (Section 5) — load-aligned
    epsilons + attitude offsets.
  * g_init policy (Section 6) — gates for using anchor / nearest-accepted
    g; mandatory `None` candidate for plateau escape.
  * Wall-clock cap helpers (Section 10).

The detectors operate on already-computed node attempts (the "history"
list), they do not call the solver themselves.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ─── 1. Node status taxonomy ──────────────────────────────────────

# String enum (kept as plain strings for CSV/JSON friendliness).
STATUS_METRIC_HARD = "metric_hard"
STATUS_METRIC_SOFT = "metric_soft"
STATUS_BRIDGE = "bridge"
STATUS_FAILED = "failed"
STATUS_TIMEOUT_FAILED = "timeout_failed"
STATUS_CAPACITY_LIMITED = "capacity_limited_fullfilm"

METRIC_STATUSES = (STATUS_METRIC_HARD, STATUS_METRIC_SOFT)
USABLE_FOR_HISTORY = (STATUS_METRIC_HARD, STATUS_METRIC_SOFT, STATUS_BRIDGE)
DROP_STATUSES = (STATUS_FAILED, STATUS_TIMEOUT_FAILED, STATUS_CAPACITY_LIMITED)


# Recommended thresholds (Section 1).
TOL_METRIC_HARD = 5e-3
TOL_METRIC_SOFT = 2e-2
TOL_BRIDGE = 5e-2
EPS_MAX_DEFAULT = 0.92
HMIN_GUARD_DEFAULT = 6e-6  # m, for the current fallback case


def classify_node_status(
    rel_residual: float,
    eps: float,
    h_min_m: float,
    *,
    eps_max: float = EPS_MAX_DEFAULT,
    hmin_guard: float = HMIN_GUARD_DEFAULT,
    timed_out: bool = False,
    plateau_locked: bool = False,
    corrector_failed: bool = False,
) -> str:
    """Classify a finished node attempt into the basin-patch taxonomy.

    Order of precedence:
        timeout > capacity_limited > corrector_failed > residual-band.
    """
    if timed_out:
        return STATUS_TIMEOUT_FAILED
    # Capacity-limited overrides plain "failed" (Section 3.5).
    if (eps > eps_max - 0.015 and h_min_m < hmin_guard
            and rel_residual > 3e-2):
        return STATUS_CAPACITY_LIMITED
    if corrector_failed and rel_residual > TOL_BRIDGE:
        return STATUS_FAILED
    if rel_residual <= TOL_METRIC_HARD:
        return STATUS_METRIC_HARD
    if rel_residual <= TOL_METRIC_SOFT:
        return STATUS_METRIC_SOFT
    if rel_residual <= TOL_BRIDGE:
        # bridge band — but veto if plateau-locked or capacity-limited.
        if plateau_locked:
            return STATUS_FAILED
        if h_min_m < hmin_guard:
            return STATUS_FAILED
        return STATUS_BRIDGE
    return STATUS_FAILED


# ─── 3. Plateau / lock-in detectors ───────────────────────────────

@dataclass
class NodeAttemptRecord:
    """Minimal attempt record consumed by the detectors.

    Use one record per *base target* attempt (midpoints don't count
    here unless they are themselves promoted to base targets).
    """
    phi_deg: float
    W_N: float
    load_angle_deg: float
    eps: float
    attitude_deg: float
    rel_residual: float
    status: str
    elapsed_sec: float = 0.0


def eps_expected(W_N: float) -> float:
    """Expected eps for a given load magnitude (Section 3.1)."""
    raw = 0.4 * (max(float(W_N), 0.0) / 1000.0) ** 0.25
    return max(0.18, min(0.88, raw))


def _angle_span_deg(values: Iterable[float]) -> float:
    vs = sorted(float(v) for v in values)
    if len(vs) < 2:
        return 0.0
    return vs[-1] - vs[0]


def _wrap_unwrapped_angle_span_deg(angles: Iterable[float]) -> float:
    """Span of a sequence of angles after unwrapping by 360°."""
    arr = [float(a) for a in angles]
    if len(arr) < 2:
        return 0.0
    unwrapped = [arr[0]]
    for a in arr[1:]:
        prev = unwrapped[-1]
        d = ((a - prev + 180.0) % 360.0) - 180.0
        unwrapped.append(prev + d)
    return max(unwrapped) - min(unwrapped)


def _W_rel_span(values: Iterable[float]) -> float:
    vs = [float(v) for v in values]
    if not vs:
        return 0.0
    base = max(min(vs), 1e-3)
    return (max(vs) - min(vs)) / base


def detect_fast_wrong_basin(
    rec: NodeAttemptRecord,
    *,
    W_percentile_70: float,
) -> bool:
    """Section 3.2: fast high-eps wrong-basin detector.

    Triggers as soon as a single failed node looks numerically wrong
    (high eps for its load, large residual). Requires immediate reseed.
    """
    if rec.status != STATUS_FAILED:
        return False
    if rec.rel_residual <= 0.10:
        return False
    if rec.eps <= eps_expected(rec.W_N) + 0.18:
        return False
    if rec.W_N >= W_percentile_70:
        return False
    return True


def detect_plateau_lock_3(records: Sequence[NodeAttemptRecord]) -> bool:
    """Section 3.3: rolling 3-node plateau detector."""
    if len(records) < 3:
        return False
    last = list(records[-3:])
    if not all(r.status in (STATUS_FAILED, STATUS_TIMEOUT_FAILED)
               for r in last):
        return False
    eps_span = _angle_span_deg(r.eps for r in last)
    att_span = _angle_span_deg(r.attitude_deg for r in last)
    if eps_span >= 0.04 or att_span >= 12.0:
        return False
    load_span = _wrap_unwrapped_angle_span_deg(r.load_angle_deg for r in last)
    W_span = _W_rel_span(r.W_N for r in last)
    res_first = max(last[0].rel_residual, 1e-12)
    res_growth = last[-1].rel_residual / res_first
    return (load_span > 20.0 or W_span > 0.35 or res_growth > 1.7)


def detect_plateau_lock_4(records: Sequence[NodeAttemptRecord]) -> bool:
    """Section 3.4: rolling 4-node plateau detector."""
    if len(records) < 4:
        return False
    last = list(records[-4:])
    n_bad = sum(1 for r in last
                if r.status in (STATUS_FAILED, STATUS_TIMEOUT_FAILED))
    if n_bad < 3:
        return False
    eps_span = _angle_span_deg(r.eps for r in last)
    if eps_span >= 0.05:
        return False
    if max(r.rel_residual for r in last) <= 0.20:
        return False
    load_span = _wrap_unwrapped_angle_span_deg(r.load_angle_deg for r in last)
    W_span = _W_rel_span(r.W_N for r in last)
    return (load_span > 30.0 or W_span > 0.50)


def detect_plateau_lock(records: Sequence[NodeAttemptRecord]) -> bool:
    return (detect_plateau_lock_3(records)
            or detect_plateau_lock_4(records))


def detect_capacity_limited(
    rel_residual: float,
    eps: float,
    h_min_m: float,
    *,
    eps_max: float = EPS_MAX_DEFAULT,
    hmin_guard: float = HMIN_GUARD_DEFAULT,
) -> bool:
    """Section 3.5: cap-lock / capacity detector.

    Pure boolean check; classify_node_status() is the canonical mapper
    that consumes it.
    """
    return (eps > eps_max - 0.015
            and h_min_m < hmin_guard
            and rel_residual > 3e-2)


# ─── 5. Multi-start reseed candidates ─────────────────────────────

@dataclass
class SeedCandidate:
    X: float
    Y: float
    eps: float
    att_offset_deg: float
    g_source: str  # "nearest_accepted" / "anchor" / "none"
    g_init: Optional[np.ndarray] = None


_DEFAULT_ATT_OFFSETS_DEG: Tuple[float, ...] = (-60.0, -35.0, -20.0, 0.0,
                                                 20.0, 35.0, 60.0)


def _eps_candidates_for(W_N: float, eps_max: float) -> List[float]:
    eps0 = eps_expected(W_N)
    raw = [
        0.55 * eps0,
        0.75 * eps0,
        1.00 * eps0,
        1.15 * eps0,
        0.25, 0.50, 0.75, 0.88,
    ]
    out: List[float] = []
    seen: List[float] = []
    lo = 0.12
    hi = max(lo + 1e-3, eps_max - 0.02)
    for v in raw:
        c = max(lo, min(hi, float(v)))
        if all(abs(c - s) > 1e-3 for s in seen):
            seen.append(c)
            out.append(c)
    return out


def build_multistart_seeds(
    Wa: np.ndarray,
    *,
    eps_max: float = EPS_MAX_DEFAULT,
    att_offsets_deg: Sequence[float] = _DEFAULT_ATT_OFFSETS_DEG,
    include_g_none: bool = True,
    nearest_accepted_g: Optional[np.ndarray] = None,
    nearest_accepted_phi_diff_deg: float = 360.0,
    nearest_accepted_dXY: float = 1.0,
    anchor_g: Optional[np.ndarray] = None,
    W_anchor: Optional[float] = None,
    load_angle_diff_to_anchor_deg: Optional[float] = None,
) -> List[SeedCandidate]:
    """Build a load-aligned multi-start seed list (Section 5).

    Sign convention: equilibrium F = Wa, with F co-aligned with the
    journal eccentricity vector (X, Y). So a load Wa pulling in direction
    theta_W places the journal in roughly the same direction → seed
    (X, Y) = eps * (cos theta, sin theta), then rotated by attitude
    offset.

    The g_init policy (Section 6) is applied here:
      * `nearest_accepted_g` is included only if the gates pass.
      * `anchor_g` is included only if the load-state is close.
      * A `g_init=None` candidate is always included when
        `include_g_none` is True (mandatory for plateau escape).
    """
    Wn = float(np.linalg.norm(Wa))
    if Wn < 1e-12:
        theta_W = 0.0
    else:
        theta_W = math.atan2(float(Wa[1]), float(Wa[0]))
    eps_list = _eps_candidates_for(Wn, eps_max)

    # Build (X, Y) candidates in load-aligned coords.
    xy_atts: List[Tuple[float, float, float, float]] = []  # (X, Y, eps, off)
    for eps in eps_list:
        for off in att_offsets_deg:
            theta = theta_W + math.radians(float(off))
            X = eps * math.cos(theta)
            Y = eps * math.sin(theta)
            xy_atts.append((X, Y, eps, float(off)))

    # Decide which g_inits to attach (Section 6).
    g_choices: List[Tuple[Optional[np.ndarray], str]] = []
    use_nearest = (
        nearest_accepted_g is not None
        and abs(nearest_accepted_phi_diff_deg) < 30.0
        and nearest_accepted_dXY < 0.12
    )
    if use_nearest:
        g_choices.append((nearest_accepted_g, "nearest_accepted"))
    use_anchor = (
        anchor_g is not None
        and W_anchor is not None
        and W_anchor > 1e-3
        and abs(Wn - W_anchor) / W_anchor < 0.5
        and (load_angle_diff_to_anchor_deg is None
             or abs(load_angle_diff_to_anchor_deg) < 30.0)
    )
    if use_anchor:
        g_choices.append((anchor_g, "anchor"))
    if include_g_none or not g_choices:
        g_choices.append((None, "none"))

    cands: List[SeedCandidate] = []
    for g, src in g_choices:
        for X, Y, eps, off in xy_atts:
            cands.append(SeedCandidate(
                X=X, Y=Y, eps=eps, att_offset_deg=off,
                g_source=src, g_init=g,
            ))
    return cands


def scout_seeds(
    candidates: Sequence[SeedCandidate],
    Wa: np.ndarray,
    eval_scout: Callable,
    *,
    keep_top: int = 3,
) -> List[Tuple[SeedCandidate, float]]:
    """Cheap-eval scout: rank candidates by control residual (Section 5.5).

    ``eval_scout(X, Y, g_init)`` -> (metrics, P, theta), expected to use a
    very cheap PS budget. Returns sorted list of (candidate, residual)
    truncated to ``keep_top``.
    """
    Wn = max(float(np.linalg.norm(Wa)), 1e-20)
    scored: List[Tuple[SeedCandidate, float]] = []
    for c in candidates:
        try:
            m, _, _ = eval_scout(c.X, c.Y, c.g_init)
            r = math.hypot(m["Fx"] - Wa[0], m["Fy"] - Wa[1]) / Wn
        except Exception:
            r = float("inf")
        scored.append((c, r))
    scored.sort(key=lambda pr: pr[1])
    return scored[:max(1, keep_top)]


# ─── 10. Wall-clock caps ──────────────────────────────────────────

@dataclass
class WallClockCaps:
    """Per-node and per-segment wall-clock budgets (Section 10)."""
    node_soft_sec: float = 8.0
    node_hard_sec: float = 15.0
    midpoint_soft_sec: float = 12.0
    midpoint_hard_sec: float = 25.0
    anchor_soft_sec: float = 60.0
    anchor_hard_sec: float = 120.0
    segment_sec: float = 7 * 60.0  # 5–7 min target per geometry
    geometry_sec: float = 7 * 60.0
    full_run_sec: float = 15 * 60.0  # 15 min acceptable upper bound


@dataclass
class CapTimer:
    """Convenience wrapper for checking caps between PS calls."""
    start: float
    soft_sec: float
    hard_sec: float

    def elapsed(self) -> float:
        return time.time() - self.start

    def soft_exceeded(self) -> bool:
        return self.elapsed() > self.soft_sec

    def hard_exceeded(self) -> bool:
        return self.elapsed() > self.hard_sec


def make_timer(soft_sec: float, hard_sec: float) -> CapTimer:
    return CapTimer(start=time.time(), soft_sec=soft_sec,
                    hard_sec=hard_sec)


# ─── Helpers exposed for runner / tests ───────────────────────────

def angle_diff_deg(a: float, b: float) -> float:
    """Shortest signed angular delta a-b in (-180, 180]."""
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


__all__ = [
    # status
    "STATUS_METRIC_HARD", "STATUS_METRIC_SOFT", "STATUS_BRIDGE",
    "STATUS_FAILED", "STATUS_TIMEOUT_FAILED", "STATUS_CAPACITY_LIMITED",
    "METRIC_STATUSES", "USABLE_FOR_HISTORY", "DROP_STATUSES",
    "TOL_METRIC_HARD", "TOL_METRIC_SOFT", "TOL_BRIDGE",
    "EPS_MAX_DEFAULT", "HMIN_GUARD_DEFAULT",
    "classify_node_status",
    # detectors
    "NodeAttemptRecord", "eps_expected",
    "detect_fast_wrong_basin",
    "detect_plateau_lock", "detect_plateau_lock_3", "detect_plateau_lock_4",
    "detect_capacity_limited",
    # multi-start
    "SeedCandidate", "build_multistart_seeds", "scout_seeds",
    # wall-clock
    "WallClockCaps", "CapTimer", "make_timer",
    # misc
    "angle_diff_deg",
]
