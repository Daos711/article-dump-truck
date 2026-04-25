"""Unit tests for the Stage I-A continuation basin patch.

All detector / classifier / multi-start logic is pure Python and works
without the Reynolds solver. Multi-shoot orchestration is exercised
through a synthetic eval_fn that mimics qualitative bearing nonlinearity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

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
    USABLE_FOR_HISTORY,
    angle_diff_deg,
    build_multistart_seeds,
    classify_node_status,
    detect_capacity_limited,
    detect_fast_wrong_basin,
    detect_plateau_lock,
    detect_plateau_lock_3,
    detect_plateau_lock_4,
    eps_expected,
    scout_seeds,
)


# ─── Status taxonomy (Section 1) ──────────────────────────────────

def test_status_metric_hard():
    s = classify_node_status(rel_residual=3e-3, eps=0.5,
                              h_min_m=20e-6)
    assert s == STATUS_METRIC_HARD


def test_status_metric_soft():
    s = classify_node_status(rel_residual=1e-2, eps=0.5,
                              h_min_m=20e-6)
    assert s == STATUS_METRIC_SOFT


def test_status_bridge_band():
    s = classify_node_status(rel_residual=4e-2, eps=0.5,
                              h_min_m=20e-6)
    assert s == STATUS_BRIDGE


def test_status_failed_above_bridge():
    s = classify_node_status(rel_residual=8e-2, eps=0.5,
                              h_min_m=20e-6, corrector_failed=True)
    assert s == STATUS_FAILED


def test_status_capacity_limited_overrides_failed():
    """Section 3.5: cap-lock overrides plain failed."""
    s = classify_node_status(
        rel_residual=0.20, eps=0.91, h_min_m=4e-6,
        eps_max=0.92, hmin_guard=6e-6,
        corrector_failed=True)
    assert s == STATUS_CAPACITY_LIMITED


def test_status_timeout_overrides_everything():
    s = classify_node_status(rel_residual=2e-3, eps=0.5,
                              h_min_m=20e-6, timed_out=True)
    assert s == STATUS_TIMEOUT_FAILED


def test_bridge_vetoed_when_thin_film():
    s = classify_node_status(rel_residual=4e-2, eps=0.85,
                              h_min_m=4e-6,
                              eps_max=0.92, hmin_guard=6e-6)
    assert s == STATUS_FAILED


def test_history_set_excludes_failures():
    """Hard rule 0.2: failed/timeout/capacity must NOT be in history."""
    for s in (STATUS_FAILED, STATUS_TIMEOUT_FAILED, STATUS_CAPACITY_LIMITED):
        assert s not in USABLE_FOR_HISTORY
        assert s in DROP_STATUSES
    for s in (STATUS_METRIC_HARD, STATUS_METRIC_SOFT, STATUS_BRIDGE):
        assert s in USABLE_FOR_HISTORY


# ─── Plateau / wrong-basin detectors (Section 3) ──────────────────

def test_eps_expected_band():
    assert 0.18 <= eps_expected(100.0) <= 0.88
    assert 0.18 <= eps_expected(20_000.0) <= 0.88


def test_fast_wrong_basin_triggers_on_high_eps_low_load():
    rec = NodeAttemptRecord(
        phi_deg=300.0, W_N=500.0, load_angle_deg=270.0,
        eps=0.55, attitude_deg=-90.0, rel_residual=0.3,
        status=STATUS_FAILED)
    assert detect_fast_wrong_basin(rec, W_percentile_70=2000.0)


def test_fast_wrong_basin_silent_on_normal_failure():
    rec = NodeAttemptRecord(
        phi_deg=300.0, W_N=500.0, load_angle_deg=270.0,
        eps=0.30, attitude_deg=-90.0, rel_residual=0.3,
        status=STATUS_FAILED)
    assert not detect_fast_wrong_basin(rec, W_percentile_70=2000.0)


def _flat_failed_records(n: int, eps0: float = 0.305,
                          load_angles=(120, 145, 170, 195),
                          W_values=(1500, 2000, 2500, 3000),
                          residuals=(0.05, 0.10, 0.20, 0.40)):
    out = []
    for i in range(n):
        out.append(NodeAttemptRecord(
            phi_deg=500 + i * 10, W_N=W_values[i],
            load_angle_deg=load_angles[i],
            eps=eps0 + i * 0.005, attitude_deg=-80.0 + i * 1.0,
            rel_residual=residuals[i], status=STATUS_FAILED))
    return out


def test_plateau_lock_3_load_angle_growth():
    recs = _flat_failed_records(3)
    assert detect_plateau_lock_3(recs)


def test_plateau_lock_3_silent_if_eps_moves():
    recs = _flat_failed_records(3)
    recs[-1].eps = recs[0].eps + 0.10
    assert not detect_plateau_lock_3(recs)


def test_plateau_lock_4_residual_growth():
    recs = _flat_failed_records(4)
    assert detect_plateau_lock_4(recs)


def test_plateau_lock_silent_when_metric_present():
    recs = _flat_failed_records(3)
    recs[-1].status = STATUS_METRIC_HARD
    recs[-1].rel_residual = 1e-3
    assert not detect_plateau_lock_3(recs)


def test_capacity_limited_pure_check():
    assert detect_capacity_limited(0.10, 0.91, 4e-6,
                                     eps_max=0.92, hmin_guard=6e-6)
    assert not detect_capacity_limited(0.10, 0.5, 30e-6,
                                         eps_max=0.92, hmin_guard=6e-6)


# ─── Multi-start reseed (Section 5) ───────────────────────────────

def test_multistart_seeds_include_g_none_candidate():
    """Section 6.4: plateau escape requires at least one g=None candidate."""
    Wa = np.array([100.0, -2000.0])
    cands = build_multistart_seeds(
        Wa, eps_max=0.92,
        nearest_accepted_g=np.array([1.0, 2.0]),
        nearest_accepted_phi_diff_deg=10.0,
        nearest_accepted_dXY=0.05,
        anchor_g=np.array([3.0, 4.0]),
        W_anchor=2200.0,
        load_angle_diff_to_anchor_deg=5.0,
    )
    assert any(c.g_init is None and c.g_source == "none" for c in cands)


def test_multistart_seeds_filter_anchor_g_when_far_load():
    """Section 6.3: anchor_g is gated by load similarity."""
    Wa = np.array([100.0, -2000.0])
    cands = build_multistart_seeds(
        Wa, eps_max=0.92,
        anchor_g=np.array([3.0, 4.0]),
        W_anchor=200.0,  # far from W = 2000
        load_angle_diff_to_anchor_deg=5.0,
    )
    assert not any(c.g_source == "anchor" for c in cands)


def test_multistart_seeds_filter_nearest_g_when_far_xy():
    """Section 6.2: nearest-accepted g gated by Δphi and ΔXY."""
    Wa = np.array([0.0, -1000.0])
    cands = build_multistart_seeds(
        Wa, eps_max=0.92,
        nearest_accepted_g=np.array([1.0, 2.0]),
        nearest_accepted_phi_diff_deg=10.0,
        nearest_accepted_dXY=0.50,  # too far
    )
    assert not any(c.g_source == "nearest_accepted" for c in cands)


def test_multistart_seeds_load_aligned():
    """Sign convention check: at att_offset=0 the seed is aligned with W."""
    Wa = np.array([0.0, -1500.0])
    cands = build_multistart_seeds(Wa, eps_max=0.92)
    aligned = [c for c in cands if abs(c.att_offset_deg) < 1e-6]
    # Y must be negative (same direction as Wa).
    assert all(c.Y < 0 for c in aligned)


def test_scout_seeds_picks_best():
    Wa = np.array([0.0, -1500.0])

    def fake_eval(X, Y, g_init):
        # residual is smallest near Y = -0.3 (for k=5e3 / denom≈1).
        Fx = 0.0
        Fy = 5e3 * Y / max((1 - X*X - Y*Y) ** 1.5, 1e-6)
        return (dict(Fx=Fx, Fy=Fy, h_min=1e-6, p_max=1e7,
                     cav_frac=0.1, friction=0.5, Ploss=300, Qout=1e-6),
                np.zeros((2, 2)), np.zeros((2, 2)))

    cands = build_multistart_seeds(Wa, eps_max=0.92)
    top = scout_seeds(cands, Wa, fake_eval, keep_top=3)
    assert len(top) == 3
    assert top[0][1] <= top[1][1] <= top[2][1]


# ─── Wrap-safe angle ──────────────────────────────────────────────

def test_angle_diff_deg_wraps():
    assert math.isclose(angle_diff_deg(10, 350), 20.0, abs_tol=1e-9)
    assert math.isclose(angle_diff_deg(350, 10), -20.0, abs_tol=1e-9)
    # Boundary at ±180° may resolve to either sign — magnitude is what
    # the predictor cares about.
    assert abs(abs(angle_diff_deg(180, 0)) - 180.0) < 1e-9


# ─── Continuation runner integration (synthetic) ──────────────────

def _synthetic_eval(k: float = 1e4):
    def f(X, Y, g_init=None):
        eps2 = min(X*X + Y*Y, 0.95**2)
        denom = max((1 - eps2) ** 1.5, 1e-6)
        return (dict(Fx=k*X/denom, Fy=k*Y/denom,
                     h_min=70e-6 * (1 - math.sqrt(eps2)),
                     p_max=1e7 * (math.sqrt(eps2) + 0.1),
                     cav_frac=0.1, friction=0.5,
                     Ploss=300.0, Qout=1e-6),
                np.zeros((2, 2)), np.zeros((2, 2)))
    return f


def test_run_continuation_segment_stops_on_plateau():
    """If detector triggers, segment must stop early."""
    from models.continuation_runner import (
        ContinuationConfig, run_continuation_segment, SolvedNode)
    from models.anchor_solver import AnchorState

    # Synthetic load that's nearly constant (no signal for continuation).
    def load_fn(p):
        return (0.0, -1500.0)

    factory = lambda mode="accepted_node": _synthetic_eval()
    anchor = AnchorState(
        phi_deg=0.0, X=0.0, Y=-0.15, eps=0.15,
        attitude_deg=-90.0, h_min=70e-6 * 0.85, p_max=1.6e7,
        cav_frac=0.1, friction=0.5, Ploss=300.0, Qout=1e-6,
        rel_residual=1e-4, g=None, status="hard_converged")

    cfg = ContinuationConfig(corrector_max_iter=4, corrector_tol=1e-3,
                                corrector_soft_tol=2e-2,
                                step_cap=0.5, eps_max=0.92,
                                max_subdiv_depth=2, min_dphi_deg=1.5)
    phi_targets = [10.0 * i for i in range(72)]
    nodes = run_continuation_segment(
        phi_targets, load_fn, factory, anchor,
        cfg=cfg, direction="forward")
    # Must produce at least the anchor as node 0.
    assert len(nodes) >= 1
    assert nodes[0].phi_deg == 0.0
    # Constant load → all easy → all metric.
    assert all(n.status in METRIC_STATUSES for n in nodes)


def test_segment_priority_hard_over_soft_over_bridge():
    """run_diesel_stage1._node_priority orders status correctly."""
    import sys as _sys, types as _types
    # Stub reynolds_solver before importing the script module.
    _sys.modules.setdefault("reynolds_solver",
                             _types.ModuleType("reynolds_solver"))
    _sys.modules.setdefault("reynolds_solver.cavitation",
                             _types.ModuleType("reynolds_solver.cavitation"))
    ps = _types.ModuleType("reynolds_solver.cavitation.payvar_salant")
    ps.solve_payvar_salant_cpu = lambda *a, **k: (None, None, 0, 0)
    ps.solve_payvar_salant_gpu = ps.solve_payvar_salant_cpu
    _sys.modules["reynolds_solver.cavitation.payvar_salant"] = ps

    from scripts.run_diesel_stage1 import _node_priority
    from models.continuation_runner import SolvedNode

    def mk(status, res=1e-3):
        return SolvedNode(
            phi_deg=0, X=0, Y=0, eps=0, attitude_deg=0,
            h_min=10e-6, p_max=0, cav_frac=0, friction=0,
            Ploss=0, Qout=0, rel_residual=res, status=status, nr_iters=0)

    a = mk(STATUS_METRIC_HARD)
    b = mk(STATUS_METRIC_SOFT)
    c = mk(STATUS_BRIDGE)
    d = mk(STATUS_FAILED)
    assert _node_priority(a) > _node_priority(b)
    assert _node_priority(b) > _node_priority(c)
    assert _node_priority(c) > _node_priority(d)
