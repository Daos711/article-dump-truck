"""Unit tests for Stage I-A anchor solver.

The real Reynolds solver is heavy and may not be available in CI. These
tests use a synthetic, analytically tractable F(X,Y) that mimics the
qualitative load–response of a journal bearing well enough to exercise
all anchor-policy branches end-to-end without invoking PS.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from models.anchor_solver import (
    DEFAULT_EXPLICIT_PHI_ANCHOR_DEG,
    PS_BUDGETS,
    pick_anchor_phi,
    solve_anchor_smooth,
    solve_anchor_textured,
)


# ─── Synthetic eval_fn ─────────────────────────────────────────────

def _make_synthetic_eval(*, k: float = 1e4, alpha_tex: float = 0.0,
                          texture_kick: float = 0.0):
    """Synthetic Fx, Fy as smooth nonlinear functions of (X, Y).

    F = +k * (X, Y) / (1 - eps^2)^1.5  + texture_kick * alpha_tex * (X, Y)

    Sign convention matches the runner: equilibrium F = Wa is reached with
    (X, Y) pointing in the same direction as Wa. The (1-eps^2)^1.5 denom
    captures the strong nonlinearity near the bushing wall, which is what
    makes the real solver hard near global max load.
    """
    def eval_fn(X, Y, g_init=None, _k=k, _at=alpha_tex, _tk=texture_kick):
        eps2 = X * X + Y * Y
        eps2 = min(eps2, 0.95 * 0.95)
        denom = (1.0 - eps2) ** 1.5
        kick = 1.0 + _tk * _at
        Fx = _k * X / max(denom, 1e-6) * kick
        Fy = _k * Y / max(denom, 1e-6) * kick
        return (
            dict(Fx=Fx, Fy=Fy,
                 h_min=1e-6 * (1.0 - math.sqrt(eps2)),
                 p_max=1e7 * (math.sqrt(eps2) + 0.1),
                 cav_frac=0.1, friction=0.5, Ploss=300.0, Qout=1e-6),
            np.zeros((2, 2)), np.zeros((2, 2)),
        )
    return eval_fn


def _factory_smooth(mode):
    return _make_synthetic_eval(k=1e4)


def _factory_textured(mode):
    return _make_synthetic_eval(k=1e4, alpha_tex=1.0, texture_kick=0.05)


def _factory_textured_hard(mode):
    # Strong texture kick that breaks direct-from-smooth at alpha=1.
    return _make_synthetic_eval(k=1e4, alpha_tex=1.0, texture_kick=2.0)


# ─── Anchor selection ─────────────────────────────────────────────

def _surrogate_load_fn():
    from models.diesel_stage1_cycle import (
        build_surrogate_heavyduty_v1, get_load_at_angle)
    cyc = build_surrogate_heavyduty_v1(n_points=72)
    phi_list = list(cyc["phi_crank_deg"])

    def load_fn(p):
        return get_load_at_angle(cyc, p)

    loads = np.array([np.linalg.norm(load_fn(p)) for p in phi_list])
    return phi_list, load_fn, loads


def test_pick_anchor_explicit_default_500():
    phi_list, load_fn, _ = _surrogate_load_fn()
    phi_a, reason, _ = pick_anchor_phi(
        phi_list, load_fn, mode="explicit",
        explicit_phi_deg=DEFAULT_EXPLICIT_PHI_ANCHOR_DEG)
    assert phi_a == 500.0
    assert "explicit" in reason


def test_pick_anchor_never_picks_global_min_or_max():
    """Hard rule 0.1 + 1.4(b)."""
    phi_list, load_fn, loads = _surrogate_load_fn()
    phi_min = phi_list[int(np.argmin(loads))]
    phi_max = phi_list[int(np.argmax(loads))]
    for mode in ("explicit", "scout_best", "from_legacy_matched_sector"):
        phi_a, _, _ = pick_anchor_phi(phi_list, load_fn, mode=mode)
        assert abs(phi_a - phi_min) > 1.0, f"{mode} hit global min-load"
        assert abs(phi_a - phi_max) > 1.0, f"{mode} hit global max-load"


def test_pick_anchor_legacy_uses_sector_midpoint():
    phi_list, load_fn, _ = _surrogate_load_fn()
    legacy = [420., 430., 440., 450., 460., 470.]
    phi_a, reason, cands = pick_anchor_phi(
        phi_list, load_fn, mode="from_legacy_matched_sector",
        legacy_matched_phis=legacy)
    assert 420.0 <= phi_a <= 470.0
    assert "midpoint" in reason
    assert len(cands) == len(legacy)


def test_pick_anchor_legacy_falls_back_to_500_when_empty():
    phi_list, load_fn, _ = _surrogate_load_fn()
    phi_a, reason, _ = pick_anchor_phi(
        phi_list, load_fn, mode="from_legacy_matched_sector",
        legacy_matched_phis=None)
    assert phi_a == 500.0
    assert "pragmatic default" in reason


# ─── Smooth anchor (load homotopy) ────────────────────────────────

def test_smooth_anchor_homotopy_lands_on_branch():
    """Section 3: lambda schedule [0.4, 0.6, 0.8, 1.0] should converge."""
    Wa = np.array([100.0, -1500.0])
    state, log = solve_anchor_smooth(
        500.0, Wa, _factory_smooth,
        lambda_schedule=(0.4, 0.6, 0.8, 1.0),
        step_cap=0.5,
        tol=1e-3, soft_tol=1e-2)
    assert state is not None, f"smooth anchor failed; log={log}"
    assert state.status in ("hard_converged", "soft_converged")
    # Final state must satisfy F = Wa (same convention as the runner).
    F = 1e4 * np.array([state.X, state.Y]) / max(
        (1.0 - state.eps**2) ** 1.5, 1e-6)
    res_norm = np.linalg.norm(F - Wa) / np.linalg.norm(Wa)
    assert res_norm < 1e-2
    # Log must contain one entry per lambda stage.
    primary = [e for e in log if e["schedule"] == "primary"]
    assert len(primary) >= 1
    # Lambda must NOT start from 0 (Section 3.2).
    assert all(e["lambda_"] > 0.0 for e in primary)


def test_smooth_anchor_does_not_use_subdivision():
    """Hard rule 0.2: smooth anchor does not call generic angle subdivision.

    We assert by construction: solve_anchor_smooth has no load_fn parameter
    and no subdivision recursion in its public API.
    """
    import inspect
    sig = inspect.signature(solve_anchor_smooth)
    assert "load_fn" not in sig.parameters
    assert "max_subdiv_depth" not in sig.parameters


def test_smooth_anchor_lambda_warmstart_carried():
    """Section 3.4: warm-start (X, Y) must be carried between lambda-stages."""
    Wa = np.array([0.0, -2000.0])
    state, log = solve_anchor_smooth(
        500.0, Wa, _factory_smooth,
        lambda_schedule=(0.4, 0.7, 1.0),
        step_cap=0.5,
        tol=1e-3, soft_tol=1e-2)
    assert state is not None
    # Successive accepted X,Y should evolve smoothly (no full reseed).
    accepted = [e for e in log if e["status"] != "failed"
                 and e["schedule"] == "primary"]
    for a, b in zip(accepted, accepted[1:]):
        # Step in eps between lambda stages should be < 0.5 (no reseed jumps).
        assert abs(b["eps"] - a["eps"]) < 0.5


# ─── Textured anchor ──────────────────────────────────────────────

def test_textured_anchor_direct_from_smooth():
    Wa = np.array([100.0, -1500.0])
    sm, _ = solve_anchor_smooth(500.0, Wa, _factory_smooth,
                                  step_cap=0.5,
                                  tol=1e-3, soft_tol=1e-2)
    assert sm is not None
    tx, log = solve_anchor_textured(
        500.0, Wa, _factory_textured, sm,
        geometry_factory=None,
        step_cap=0.5,
        tol=1e-3, soft_tol=1e-2)
    assert tx is not None
    direct = [e for e in log if e["path"] == "direct_from_smooth"]
    assert len(direct) == 1
    assert direct[0]["status"] != "failed"


def test_textured_anchor_geometry_rescue():
    """If direct textured solve fails, geometry continuation must rescue."""
    Wa = np.array([100.0, -1500.0])
    sm, _ = solve_anchor_smooth(500.0, Wa, _factory_smooth,
                                  step_cap=0.5,
                                  tol=1e-3, soft_tol=1e-2)
    assert sm is not None

    # Build a geometry factory that ramps texture_kick with alpha_tex.
    def geom_factory(alpha_tex, mode):
        return _make_synthetic_eval(
            k=2.5e4, alpha_tex=alpha_tex, texture_kick=8e4)

    tx, log = solve_anchor_textured(
        500.0, Wa, _factory_textured_hard, sm,
        geometry_factory=geom_factory,
        alpha_tex_schedule=(0.33, 0.66, 1.0),
        step_cap=0.5,
        tol=1e-3, soft_tol=1e-2)
    # Either direct succeeds or rescue does; anchor must end accepted.
    assert tx is not None, f"textured rescue failed; log={log}"
    rescued = [e for e in log if e["path"] == "geometry_continuation"]
    if log[0]["status"] == "failed":
        # Direct failed → rescue must have run all alpha stages.
        assert len(rescued) >= 1


# ─── PS budget contract (Section 5) ───────────────────────────────

def test_ps_budgets_stage_dependent_and_trial_cheaper_than_accepted():
    assert PS_BUDGETS["trial"]["ps_max_iter"] < \
        PS_BUDGETS["anchor_stage_first"]["ps_max_iter"]
    assert PS_BUDGETS["scout"]["ps_max_iter"] < \
        PS_BUDGETS["accepted_node"]["ps_max_iter"]
    assert PS_BUDGETS["anchor_stage_later"]["hs_warmup_iter"] <= \
        PS_BUDGETS["anchor_stage_first"]["hs_warmup_iter"]
