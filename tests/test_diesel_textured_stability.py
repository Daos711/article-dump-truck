"""Stage Texture Stability Diesel — pipeline-side regression tests.

Tests cover (Section 6 of the patch spec):
    1. Paired smooth-vs-textured comparison common_valid mask has the
       right shape and excludes solver_failed / above_range angles.
    2. Cold-start retry contract: when the first attempt fails, the
       wrapper retries with P_init=None (textured only), never reuses
       the failed pressure for the next angle, and never propagates
       NaN pressure forward through the warm-start chain.
    3. NaN-tolerant interpolation: a W_table with NaN points does not
       crash the bracket localisation in find_epsilon_for_load (it
       must behave like a finite-subset bisection or report
       ``wtable_failed`` cleanly).

Tests in this file are independent of ``reynolds_solver``: they
exercise the pipeline-side helpers directly.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from models.diesel_quasistatic import (
    _compute_paired,
    _solver_result_is_sane,
)


# ─── 1. Paired comparison common_valid mask ────────────────────────

def _mk_cfg(*, label, oil_name, textured, color="k", ls="-"):
    return dict(label=label, textured=textured, color=color, ls=ls,
                oil={"name": oil_name, "eta_diesel": 0.01,
                     "eta_pump": 0.022, "rho": 875.0})


def _arr(values, n_phi):
    """Build a (1, n_phi) row of given scalar values, padded with zeros."""
    a = np.zeros(n_phi)
    a[: len(values)] = values
    return a


def test_paired_uses_common_valid_mask_only():
    """Paired stats must come from the SAME angles for both sides."""
    cfgs = [
        _mk_cfg(label="smooth_min", oil_name="mineral", textured=False),
        _mk_cfg(label="textured_min", oil_name="mineral", textured=True),
    ]
    n_phi = 6
    # Smooth valid on angles 0..3; textured valid on 1..4.
    # Common is {1, 2, 3} → 3 angles.
    valid = np.array([
        [True, True, True, True, False, False],
        [False, True, True, True, True, False],
    ])
    T_eff = np.array([
        [110.0, 111.0, 112.0, 113.0, 114.0, 115.0],
        [120.0, 121.0, 122.0, 123.0, 124.0, 125.0],
    ])
    # Sentinel arrays — not the focus, but must exist with right shape.
    eta = np.full_like(T_eff, 0.01)
    P_loss = T_eff * 10.0
    hmin = np.full_like(T_eff, 30e-6)
    pmax = np.full_like(T_eff, 1e7)
    eps = np.full_like(T_eff, 0.5)
    F_tr = T_eff.copy()
    paired = _compute_paired(cfgs, valid, T_eff, eta, P_loss,
                              hmin, pmax, eps, F_tr)
    assert len(paired) == 1
    rec = paired[0]
    assert rec["common_valid_count"] == 3
    # On angles {1, 2, 3} the textured-smooth deltas are uniformly +10.
    assert rec["mean_dT_eff"] == pytest.approx(10.0)


def test_paired_reports_zero_overlap_cleanly():
    """No common_valid angles ⇒ stats stay NaN, no crash."""
    cfgs = [
        _mk_cfg(label="s", oil_name="mineral", textured=False),
        _mk_cfg(label="t", oil_name="mineral", textured=True),
    ]
    n_phi = 4
    valid = np.array([[True, False, True, False],
                      [False, True, False, True]])
    T_eff = np.zeros((2, n_phi))
    z = np.zeros((2, n_phi))
    paired = _compute_paired(cfgs, valid, T_eff, z, z, z, z, z, z)
    rec = paired[0]
    assert rec["common_valid_count"] == 0
    assert np.isnan(rec["mean_dT_eff"])
    assert np.isnan(rec["mean_dP_loss"])


def test_paired_excludes_solver_failed_and_above_range():
    """valid_fullfilm is False for above_range / solver_failed angles, so
    those angles never enter common_valid. We model this by feeding
    valid=False for the angles in question and verifying the paired
    stats are computed only on the surviving angles.
    """
    cfgs = [
        _mk_cfg(label="s", oil_name="mineral", textured=False),
        _mk_cfg(label="t", oil_name="mineral", textured=True),
    ]
    # 5 angles. Smooth: 4 ok (T=100..103), 1 above_range (must be excluded).
    # Textured: 3 ok (T=200, 201, 202), 1 solver_failed (excluded), 1 above_range (excluded).
    valid = np.array([
        [True,  True,  True,  True,  False],
        [True,  True,  True,  False, False],
    ])
    T_eff = np.array([
        [100.0, 101.0, 102.0, 103.0, 999.9],   # 999.9 wouldn't be visible in mean
        [200.0, 201.0, 202.0, 888.8, 777.7],
    ])
    z = np.zeros_like(T_eff)
    paired = _compute_paired(cfgs, valid, T_eff, z, z, z, z, z, z)
    rec = paired[0]
    assert rec["common_valid_count"] == 3
    # Only angles 0, 1, 2 contribute. Mean Δ = 100.
    assert rec["mean_dT_eff"] == pytest.approx(100.0)
    # 999.9 / 888.8 / 777.7 must not have leaked into the paired stat.
    assert rec["mean_T_smooth"] == pytest.approx(101.0)
    assert rec["mean_T_textured"] == pytest.approx(201.0)


def test_paired_skips_pairs_without_both_sides():
    """Oil with only smooth (or only textured) configured ⇒ no record."""
    cfgs = [
        _mk_cfg(label="s_min", oil_name="mineral", textured=False),
        _mk_cfg(label="s_rape", oil_name="rapeseed", textured=False),
    ]
    valid = np.zeros((2, 4), dtype=bool)
    z = np.zeros((2, 4))
    paired = _compute_paired(cfgs, valid, z, z, z, z, z, z, z)
    # Both sides smooth → no textured to compare against → empty list.
    assert paired == []


# ─── 2. Sanity threshold guard ─────────────────────────────────────

def test_solver_result_sane_accepts_finite_in_range():
    P = np.full((4, 4), 1e7)        # 10 MPa
    F = 5.0e4                        # 50 kN
    assert _solver_result_is_sane(F, P)


def test_solver_result_sane_rejects_nan_F():
    P = np.zeros((4, 4))
    assert not _solver_result_is_sane(float("nan"), P)


def test_solver_result_sane_rejects_huge_F():
    P = np.zeros((4, 4))
    assert not _solver_result_is_sane(1.0e15, P)


def test_solver_result_sane_rejects_huge_P():
    P = np.full((4, 4), 1.0e15)
    assert not _solver_result_is_sane(1.0, P)


# ─── 3. Cold-retry contract via _solve_and_check (synthetic solver) ──

def test_solve_and_check_retries_on_first_failure_with_cold_start():
    """When the first attempt with warm P_init triggers SOR-warning,
    the wrapper must retry with P_init=None and return the second
    result. Counter records the retry tag so the script can log it.
    """
    import warnings as _warnings
    import models.diesel_quasistatic as dq

    # Build a fake solve_and_compute that fails on the first call (only
    # if P_init is given) and succeeds on the cold restart.
    call_log = []

    def fake_solve(H, d_phi, d_Z, R, L, eta, n, c,
                   phi_1D, Z_1D, Phi_mesh, **kw):
        p_init = kw.get("P_init")
        call_log.append(p_init is not None)
        if p_init is not None:
            _warnings.warn(
                "SOR не сошёлся: delta=2.0e-04, n_iter=50000",
                UserWarning)
            P = np.full(H.shape, np.nan)
            return (P, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"),
                    0, None, 0.0)
        # Cold start: clean finite result.
        P = np.full(H.shape, 1.0e6)
        return (P, 1.0e4, 0.01, 1.0e-5, 30e-6, 1.0e7, 50.0,
                100, None, 0.0)

    H = np.ones((4, 4))
    monkey_target = dq.solve_and_compute
    try:
        dq.solve_and_compute = fake_solve
        warm = np.full((4, 4), 5.0e5)
        out = dq._solve_and_check(
            H, 0.1, 0.1, 0.05, 0.04, 0.01, 1000.0, 5e-5,
            np.arange(4), np.linspace(-1, 1, 4), np.zeros((4, 4)),
            P_init=warm,
            allow_cold_retry=True,
        )
    finally:
        dq.solve_and_compute = monkey_target
    assert call_log == [True, False], call_log
    P_out, F_out = out[0], out[1]
    ok = out[7]; reason = out[8]
    assert ok is True
    assert F_out == pytest.approx(1.0e4)
    assert reason in ("retried_cold_start", "retried_with_kw")
    # The cold-start result must NOT be NaN — never propagate the
    # poisoned warm-start pressure forward.
    assert np.all(np.isfinite(P_out))


def test_solve_and_check_no_retry_when_disabled_or_cold_start():
    """allow_cold_retry=False, OR P_init already None, must NOT retry."""
    import warnings as _warnings
    import models.diesel_quasistatic as dq

    call_log = []

    def fake_solve(H, d_phi, d_Z, R, L, eta, n, c,
                   phi_1D, Z_1D, Phi_mesh, **kw):
        call_log.append(kw.get("P_init") is not None)
        _warnings.warn(
            "SOR не сошёлся: delta=2.0e-04, n_iter=50000",
            UserWarning)
        return (np.full(H.shape, np.nan), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"),
                float("nan"), 0, None, 0.0)

    H = np.ones((4, 4))
    monkey_target = dq.solve_and_compute
    try:
        dq.solve_and_compute = fake_solve
        warm = np.full((4, 4), 5.0e5)
        # allow_cold_retry=False
        out = dq._solve_and_check(
            H, 0.1, 0.1, 0.05, 0.04, 0.01, 1000.0, 5e-5,
            np.arange(4), np.linspace(-1, 1, 4), np.zeros((4, 4)),
            P_init=warm,
            allow_cold_retry=False,
        )
        assert call_log == [True]
        assert out[7] is False

        call_log.clear()
        # P_init=None already cold; no retry possible
        out2 = dq._solve_and_check(
            H, 0.1, 0.1, 0.05, 0.04, 0.01, 1000.0, 5e-5,
            np.arange(4), np.linspace(-1, 1, 4), np.zeros((4, 4)),
            P_init=None,
            allow_cold_retry=True,
        )
        assert call_log == [False]
        assert out2[7] is False
    finally:
        dq.solve_and_compute = monkey_target


# ─── 4. W_table NaN-tolerant interpolation ─────────────────────────

def test_find_epsilon_handles_nan_w_table_gracefully():
    """A W_table with NaN points must NOT crash; the bracket localiser
    must work on the finite subset OR report wtable_failed when too
    few finite entries remain.
    """
    import models.diesel_quasistatic as dq

    eps_table = np.array([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95])
    W_table = np.array([1.0e3, 5.0e3, 2.0e4, np.nan, 1.0e5, 2.0e5, np.nan])

    # We only need to exercise the bracket logic — stub make_H +
    # solve_and_compute so no real solver is required.
    fake_eval_count = {"n": 0}

    def fake_make_H(eps, *_a, **_kw):
        return np.full((4, 4), 1.0)

    def fake_solve(*a, **kw):
        # Pretend F follows linear interpolation on the finite W_table.
        # We don't depend on the exact value here; only that it's
        # finite, sane, and the bisection terminates without crashing.
        fake_eval_count["n"] += 1
        return (np.full((4, 4), 1.0e6), 5.0e4, 0.01, 1.0e-5,
                30.0e-6, 1.0e7, 50.0, 100, None, 0.0)

    monkeyH = dq.make_H
    monkeyS = dq.solve_and_compute
    try:
        dq.make_H = fake_make_H
        dq.solve_and_compute = fake_solve
        out = dq.find_epsilon_for_load(
            5.0e4, eps_table, W_table,
            np.zeros((4, 4)), np.zeros((4, 4)),
            np.arange(4), np.linspace(-1, 1, 4), 0.1, 0.1,
            oil={"eta_diesel": 0.01, "eta_pump": 0.022, "rho": 875},
            textured=False,
        )
    finally:
        dq.make_H = monkeyH
        dq.solve_and_compute = monkeyS
    eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, status = out
    assert status in ("ok", "below_range", "above_range",
                      "solver_failed", "wtable_failed")
    # The fake solve always returns the same F_hyd, so 5 inner solves
    # is what the bisection performs — verify we did not skip the
    # bisection loop nor crash mid-flight.
    assert fake_eval_count["n"] == 5


def test_find_epsilon_reports_wtable_failed_when_subset_too_small():
    import models.diesel_quasistatic as dq
    eps_table = np.array([0.05, 0.10, 0.20])
    W_table = np.array([np.nan, 1.0e3, np.nan])

    out = dq.find_epsilon_for_load(
        5.0e4, eps_table, W_table,
        np.zeros((4, 4)), np.zeros((4, 4)),
        np.arange(4), np.linspace(-1, 1, 4), 0.1, 0.1,
        oil={"eta_diesel": 0.01, "eta_pump": 0.022, "rho": 875},
        textured=False,
    )
    assert out[7] == "wtable_failed"
