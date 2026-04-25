"""Stage Diesel Transient Production Metrics — postprocessing tests.

These exercise the per-config production metrics + paired-extended
deltas that the Stage_Diesel_Transient_Production_Metrics patch
introduces. All tests are pipeline-side and avoid the real solver.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from models.diesel_transient import (
    CONFIGS,
    DEFAULT_FIRING_SECTOR_DEG,
    _auc_eps_over_range,
    _compute_paired_extended,
    _compute_production_metrics,
    _firing_mask,
    _percentile,
    _recovery_angle_deg,
)


# ─── 1. Firing-sector mask ─────────────────────────────────────────

def test_firing_sector_filter_correctly_identifies_steps():
    phi = np.array([100.0, 340.0, 360.0, 480.0, 500.0,
                    340.0 - 0.001, 480.0 + 0.001])
    m = _firing_mask(phi, (340.0, 480.0))
    expected = np.array([False, True, True, True, False,
                          False, False])
    assert m.tolist() == expected.tolist()


# ─── 2. Pressure percentiles ───────────────────────────────────────

def test_pmax_firing_percentiles_synthetic():
    """100 evenly spaced pressures from 1 to 100 MPa: p95=95,
    p99=99, max=100."""
    arr = np.linspace(1e6, 100e6, 100)
    assert _percentile(arr, 95) == pytest.approx(95.05e6, rel=1e-3)
    assert _percentile(arr, 99) == pytest.approx(99.01e6, rel=1e-3)
    assert float(arr.max()) == 100e6


# ─── 3. h_min below-threshold counts (per-config helper) ──────────

def test_hmin_below_threshold_counts_via_production_metrics():
    """Build a one-config last-cycle synthetic where h_min sweeps
    a known shape and verify steps_below_*um counts."""
    n = 50
    phi = np.linspace(0.0, 720.0, n, endpoint=False)
    # h_min stair: first 10 below 6 µm, next 10 below 8, next 10
    # below 10, last 20 above 10.
    hmin = np.empty(n)
    hmin[:10] = 5.0e-6
    hmin[10:20] = 7.0e-6
    hmin[20:30] = 9.0e-6
    hmin[30:] = 15.0e-6
    eps_x = np.zeros((1, n))
    eps_y = np.full((1, n), -0.5)
    pmax = np.full((1, n), 1e6)
    P_loss = np.full((1, n), 100.0)
    valid = np.ones((1, n), dtype=bool)
    cfgs = [{"label": "test", "textured": False,
              "oil": {"name": "x", "eta_diesel": 0.01,
                      "eta_pump": 0.022, "rho": 875}}]
    out = _compute_production_metrics(
        cfg_list=cfgs, last_start=0,
        phi_crank_deg=phi,
        eps_x_all=eps_x, eps_y_all=eps_y,
        hmin_all=hmin.reshape(1, n),
        pmax_all=pmax, P_loss_all=P_loss,
        valid_dynamic_all=valid, valid_no_clamp_all=valid,
        omega_rad_s=199.0,
        firing_sector_deg=(340.0, 480.0),
    )
    rec = out[0]
    assert rec["steps_hmin_below_6um"] == 10
    assert rec["steps_hmin_below_8um"] == 20  # below 6 + below 8
    assert rec["steps_hmin_below_10um"] == 30
    # Angle metrics > 0 only when there are below-threshold steps.
    assert rec["angle_hmin_below_6um"] > 0


# ─── 4. Recovery angle ─────────────────────────────────────────────

def test_recovery_angle_to_threshold():
    """ε starts at 0.95 at φ=350°, drops below 0.90 at +10°,
    below 0.85 at +20°, below 0.80 at +30°."""
    phi = np.array([350.0, 360.0, 370.0, 380.0])
    eps = np.array([0.95, 0.89, 0.84, 0.79])
    a90, ok90 = _recovery_angle_deg(phi, eps, 0.90)
    a85, ok85 = _recovery_angle_deg(phi, eps, 0.85)
    a80, ok80 = _recovery_angle_deg(phi, eps, 0.80)
    assert ok90 is True and a90 == pytest.approx(10.0)
    assert ok85 is True and a85 == pytest.approx(20.0)
    assert ok80 is True and a80 == pytest.approx(30.0)


def test_recovery_failed_when_eps_never_drops():
    phi = np.array([350.0, 360.0, 370.0, 380.0])
    eps = np.array([0.95, 0.94, 0.92, 0.91])
    a, ok = _recovery_angle_deg(phi, eps, 0.85)
    assert ok is False
    assert np.isnan(a)


# ─── 5. AUC ε ──────────────────────────────────────────────────────

def test_auc_eps_360_480_synthetic():
    """ε constant = 0.8 sampled densely over [360°, 480°]:
    AUC = 0.8 × (480-360) = 96."""
    phi = np.linspace(360.0, 480.0, 121)
    eps = np.full_like(phi, 0.8)
    auc = _auc_eps_over_range(phi, eps, 360.0, 480.0)
    assert auc == pytest.approx(96.0, abs=1e-9)


def test_auc_eps_returns_nan_when_window_empty():
    phi = np.array([10.0, 20.0])
    eps = np.array([0.5, 0.5])
    auc = _auc_eps_over_range(phi, eps, 360.0, 480.0)
    assert np.isnan(auc)


# ─── 6. P_loss impulse units ───────────────────────────────────────

def test_ploss_impulse_firing_units():
    """P_loss = 100 W constant over 360° to 480° at omega=199 rad/s.
    Time span = (480-360) deg in rad / omega = 2.094 rad / 199
    ≈ 0.01053 s. Impulse should be ≈ 100 W × 0.01053 s ≈ 1.053 J.
    """
    n = 121
    phi = np.linspace(360.0, 480.0, n)
    P_loss = np.full((1, n), 100.0)
    pmax = np.full((1, n), 1e6)
    hmin = np.full((1, n), 30e-6)
    eps_x = np.zeros((1, n))
    eps_y = np.full((1, n), -0.5)
    valid = np.ones((1, n), dtype=bool)
    cfgs = [{"label": "test", "textured": False,
              "oil": {"name": "x", "eta_diesel": 0.01,
                      "eta_pump": 0.022, "rho": 875}}]
    out = _compute_production_metrics(
        cfg_list=cfgs, last_start=0,
        phi_crank_deg=phi,
        eps_x_all=eps_x, eps_y_all=eps_y,
        hmin_all=hmin, pmax_all=pmax, P_loss_all=P_loss,
        valid_dynamic_all=valid, valid_no_clamp_all=valid,
        omega_rad_s=199.0,
        firing_sector_deg=(360.0, 480.0),
    )
    expected_t = np.deg2rad(480.0 - 360.0) / 199.0
    expected_J = 100.0 * expected_t
    assert out[0]["ploss_impulse_firing_J"] == pytest.approx(
        expected_J, rel=1e-4)
    assert out[0]["ploss_firing_mean_W"] == pytest.approx(100.0)
    assert out[0]["ploss_firing_max_W"] == pytest.approx(100.0)


# ─── 7. Paired extended deltas use common_no_clamp only ──────────

def test_paired_extended_metrics_use_common_no_clamp_only():
    """Smooth and textured each have their own valid_no_clamp pattern;
    the paired delta must come from the intersection only."""
    n = 8
    phi = np.linspace(0.0, 720.0, n, endpoint=False)
    cfgs = [
        {"label": "smooth_min", "textured": False,
         "oil": {"name": "mineral", "eta_diesel": 0.01,
                 "eta_pump": 0.022, "rho": 875}},
        {"label": "tex_min", "textured": True,
         "oil": {"name": "mineral", "eta_diesel": 0.01,
                 "eta_pump": 0.022, "rho": 875}},
    ]
    # Smooth no_clamp valid on 0..5; textured on 1..6. Common = 1..5.
    valid_noc = np.array([
        [True, True, True, True, True, True, False, False],
        [False, True, True, True, True, True, True, False],
    ])
    valid_dyn = valid_noc.copy()
    pmax = np.array([
        [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 9e9, 9e9],   # 9e9 in invalid tail
        [2e6, 2e6, 2e6, 2e6, 2e6, 2e6, 9e9, 9e9],
    ])
    hmin = np.full((2, n), 30e-6)
    eps_x = np.zeros((2, n))
    eps_y = np.full((2, n), -0.5)
    P_loss = np.full((2, n), 50.0)
    paired_basic = []
    prod = _compute_production_metrics(
        cfg_list=cfgs, last_start=0,
        phi_crank_deg=phi,
        eps_x_all=eps_x, eps_y_all=eps_y,
        hmin_all=hmin, pmax_all=pmax, P_loss_all=P_loss,
        valid_dynamic_all=valid_dyn, valid_no_clamp_all=valid_noc,
        omega_rad_s=199.0,
        firing_sector_deg=(0.0, 720.0),
    )
    extended = _compute_paired_extended(
        cfgs, paired_basic, prod, last_start=0,
        valid_no_clamp_all=valid_noc,
        valid_dynamic_all=valid_dyn,
        eps_x_all=eps_x, eps_y_all=eps_y,
        hmin_all=hmin, pmax_all=pmax, P_loss_all=P_loss,
        omega_rad_s=199.0,
        firing_sector_deg=(0.0, 720.0),
        phi_crank_deg=phi,
    )
    rec = extended[0]
    assert rec["common_valid_no_clamp_count"] == 5
    # Δ should reflect the textured-smooth difference on common only:
    # smooth pmax=1e6, textured=2e6 → delta=+1 MPa. The 9e9 outliers
    # in invalid tail must NOT leak in.
    assert rec["delta_pmax_firing_max"] == pytest.approx(1e6, abs=1)


# ─── 8. Orbit plot — file lands in run_dir ────────────────────────

def test_orbit_plot_creates_file(tmp_path):
    """Smoke: _plot_orbit_lastcycle writes orbit_lastcycle.png with
    a synthetic 1-cycle results dict."""
    pytest.importorskip("matplotlib")
    from scripts.run_diesel_thd_transient import _plot_orbit_lastcycle
    n = 12
    phi = np.linspace(0.0, 720.0, n, endpoint=False)
    eps_x = np.cos(np.deg2rad(phi)).reshape(1, n) * 0.5
    eps_y = np.sin(np.deg2rad(phi)).reshape(1, n) * 0.5
    results = dict(
        configs=[{"label": "smooth-test", "textured": False,
                   "color": "blue", "ls": "-"}],
        eps_x=eps_x, eps_y=eps_y,
        last_start=0, n_steps_per_cycle=n,
        phi_crank_deg=phi, phi_last=phi,
    )
    _plot_orbit_lastcycle(results, str(tmp_path))
    assert os.path.isfile(tmp_path / "orbit_lastcycle.png")


# ─── 9. Production metrics integration into run_transient ────────

def test_production_metrics_present_in_run_transient_smoke():
    """Smoke: with a synthetic solve_reynolds, run_transient still
    returns the production_metrics + paired_extended + firing_sector
    keys (even on a tiny short run)."""
    import warnings as _w
    import models.diesel_transient as dt

    def fake_solve(H, d_phi, d_Z, R, L, **kw):
        # Constant non-dim P=1 → P_dim = p_scale ~ small, OK sanity.
        return (np.full(H.shape, 1.0), 1e-6, 100, True)

    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake_solve
        res = dt.run_transient(
            F_max=200_000.0,
            configs=[CONFIGS[0], CONFIGS[2]],
            n_grid=20, n_cycles=1,
            d_phi_base_deg=20.0, d_phi_peak_deg=10.0,
        )
    finally:
        dt.solve_reynolds = target
    assert "production_metrics" in res
    assert "paired_extended" in res
    assert "firing_sector_deg" in res
    assert isinstance(res["production_metrics"], list)
    assert len(res["production_metrics"]) == 2
    rec = res["production_metrics"][0]
    for key in ("pmax_firing_p95", "pmax_firing_p99",
                "pmax_firing_max", "hmin_p5", "hmin_min",
                "steps_hmin_below_10um", "max_eps_lastcycle",
                "phi_at_max_eps", "eps_at_phi_421",
                "auc_eps_360_480",
                "ploss_impulse_firing_J", "ploss_firing_mean_W"):
        assert key in rec, f"production_metrics missing {key}"
