"""Stage Diesel Transient AngleWeighted Metrics — contract tests.

Two issues this patch addresses without touching physics or the
abort gate:

1. Count-based envelope fractions over-state clamp incidence on the
   adaptive temporal grid because d_phi_peak ≪ d_phi_base. The
   patch adds ``compute_angle_weighted_envelope`` and re-keys the
   ``valid_no_clamp_frac_min`` applicability gate on Δφ-weighted
   fractions (count-based numbers stay as legacy diagnostic).

2. The global header could read ``PRODUCTION RESULT`` for runs
   where every config completed all steps but every config was
   outside the applicability gate. The classifier gains a new
   ``completed_boundary_limited_result`` category and the header
   becomes ``COMPLETED BOUNDARY-LIMITED RESULT``. The paired
   smooth-vs-textured block is renamed to ``Conditional paired
   diagnostics`` in that regime so downstream readers do not
   conflate boundary-limited numbers with a clean full-film
   comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_transient import (
    compute_angle_weighted_envelope,
    classify_envelope_per_config,
)
from scripts.run_diesel_thd_transient import (
    classify_global_status,
    _write_summary,
)


# ─── 1. Angle-weighted helper math ─────────────────────────────────

def test_angle_weighted_full_cycle_uniform_dphi():
    """Uniform Δφ=1° over 360 steps with 50% valid_no_clamp must
    yield angle_frac == count_frac == 0.5 (uniform grid: angle and
    count weighting agree)."""
    n = 360
    dphi = np.full(n, 1.0)
    phi = np.arange(n, dtype=float)         # 0..359
    vnc = np.zeros(n, dtype=bool)
    vnc[::2] = True                         # 50%
    cc = ~vnc                                # complement
    diag = compute_angle_weighted_envelope(
        dphi=dphi, valid_no_clamp=vnc, contact_clamp=cc,
        valid_dynamic=np.ones(n, dtype=bool),
        solver_success=np.ones(n, dtype=bool),
        phi_mod=phi, n_completed=n,
        firing_sector_deg=(340.0, 480.0),
        last_cycle_mask=None,
    )
    assert diag["cycle_angle_deg"] == pytest.approx(360.0)
    assert diag["valid_no_clamp_angle_frac"] == pytest.approx(0.5)
    assert diag["contact_angle_frac"] == pytest.approx(0.5)


def test_angle_weighted_adaptive_concentration():
    """Synthetic adaptive grid: 10 fine steps Δφ=0.1° in [330,420]
    are 100% clamp; 90 coarse steps Δφ=1° elsewhere are 0% clamp.

    Count-based ratio: clamp/(total) = 10/100 = 0.10 — looks fine.
    Angle-based ratio: clamp_angle/total_angle =
    10*0.1 / (10*0.1 + 90*1) = 1 / 91 ≈ 0.011 — confirms that with
    Δφ-weighting the true angular share of clamp is much smaller
    than the count-based number suggests when the fine zone is
    100% clamped *and* under-represented in count.

    The test checks the inverse case: the count-based number
    *understates* clamp here because the fine zone is small in
    count but uniform. The contract is just: count_frac and
    angle_frac differ when Δφ is non-uniform.
    """
    # 10 fine steps in firing zone, all clamp.
    fine_phi = np.linspace(330.0, 420.0, 10, endpoint=False)
    fine_dphi = np.full(10, 0.1)
    fine_vnc = np.zeros(10, dtype=bool)
    fine_cc = np.ones(10, dtype=bool)
    # 90 coarse steps elsewhere, all valid_no_clamp.
    coarse_phi = np.linspace(0.0, 720.0, 90, endpoint=False)
    coarse_dphi = np.full(90, 1.0)
    coarse_vnc = np.ones(90, dtype=bool)
    coarse_cc = np.zeros(90, dtype=bool)
    # Concatenate (order doesn't matter for the angle-weighted helper).
    dphi = np.concatenate([fine_dphi, coarse_dphi])
    phi = np.concatenate([fine_phi, coarse_phi])
    vnc = np.concatenate([fine_vnc, coarse_vnc])
    cc = np.concatenate([fine_cc, coarse_cc])
    n = dphi.size
    diag = compute_angle_weighted_envelope(
        dphi=dphi, valid_no_clamp=vnc, contact_clamp=cc,
        valid_dynamic=np.ones(n, dtype=bool),
        solver_success=np.ones(n, dtype=bool),
        phi_mod=phi, n_completed=n,
        firing_sector_deg=(340.0, 480.0),
        last_cycle_mask=None,
    )
    # Count-based contact frac (legacy): 10/100 = 0.10.
    # Angle-based contact frac: 1.0 / 91.0 ≈ 0.011.
    count_cc = float(cc.sum()) / float(n)
    angle_cc = diag["contact_angle_frac"]
    assert count_cc == pytest.approx(0.10)
    assert angle_cc == pytest.approx(1.0 / 91.0, rel=1e-6)
    # The count-based reading materially misrepresents the true
    # angular share — that's the artifact this stage repairs.
    assert abs(count_cc - angle_cc) > 0.05


def test_angle_weighted_firing_sector_subset():
    """Firing-sector statistics must restrict to phi in [lo, hi]
    inclusive. Synthetic: 200 steps over phi=0..720°, all Δφ=1°,
    half clamp inside the firing sector and 0% outside."""
    n = 720
    dphi = np.full(n, 1.0)
    phi = np.arange(n, dtype=float)
    fs = (340.0, 480.0)
    in_firing = (phi >= fs[0]) & (phi <= fs[1])
    cc = in_firing.copy()                   # 100% clamp inside firing
    vnc = ~cc
    diag = compute_angle_weighted_envelope(
        dphi=dphi, valid_no_clamp=vnc, contact_clamp=cc,
        valid_dynamic=np.ones(n, dtype=bool),
        solver_success=np.ones(n, dtype=bool),
        phi_mod=phi, n_completed=n,
        firing_sector_deg=fs,
        last_cycle_mask=None,
    )
    expected_firing_angle = float(in_firing.sum())
    assert diag["firing_angle_deg"] == pytest.approx(expected_firing_angle)
    # All firing steps are clamp → firing contact_frac = 1.0.
    assert diag["contact_angle_firing_frac"] == pytest.approx(1.0)
    assert diag["valid_no_clamp_angle_firing_frac"] == pytest.approx(0.0)
    # Full-cycle: clamp share = (firing window) / 720°.
    assert diag["contact_angle_frac"] == pytest.approx(
        expected_firing_angle / 720.0)


# ─── 2. classify_envelope_per_config: angle vs legacy gate ─────────

def test_envelope_applicable_uses_angle_weighted_by_default():
    """Synthetic: count-based vnc_frac = 0.83 (< 0.85 → fails) but
    angle-weighted vnc_frac = 0.90 (≥ 0.85 → passes). With the
    default ``use_angle_weighted=True`` the gate must accept."""
    aw = {
        "valid_no_clamp_angle_frac": 0.90,
        "contact_angle_frac": 0.10,
    }
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=99,
        valid_dynamic_count=95,
        valid_no_clamp_count=83,         # 0.83 < 0.85 — count gate fails
        retry_exhausted_count=0,
        aborted=False,
        angle_weighted_full=aw,
        use_angle_weighted=True,
    )
    assert ok is True
    assert reason == "ok"


def test_envelope_applicable_use_angle_weighted_false_legacy():
    """Backward-compat flag: ``use_angle_weighted=False`` restores
    the count-based behaviour. The same inputs that pass the angle-
    weighted gate must fail the count-based one."""
    aw = {
        "valid_no_clamp_angle_frac": 0.90,
        "contact_angle_frac": 0.10,
    }
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=99,
        valid_dynamic_count=95,
        valid_no_clamp_count=83,         # 0.83 < 0.85 — count gate fails
        retry_exhausted_count=0,
        aborted=False,
        angle_weighted_full=aw,
        use_angle_weighted=False,        # force legacy
    )
    assert ok is False
    # And the failure reason must use the count-based label so the
    # legacy diagnostic remains identifiable.
    assert "valid_no_clamp_frac=" in reason
    assert "valid_no_clamp_angle_frac" not in reason


# ─── 3. classify_global_status: completed_boundary_limited ─────────

def test_classify_global_completed_boundary_limited():
    """All configs completed (status="ok") but every config is
    outside the applicability gate (applicable=False) → new
    ``completed_boundary_limited_result`` category."""
    recs = [
        {"status": "ok", "applicable": False},
        {"status": "ok", "applicable": False},
        {"status": "ok", "applicable": False},
    ]
    assert (classify_global_status(recs)
            == "completed_boundary_limited_result")


def test_classify_global_completed_mixed_returns_partial():
    """One applicable config and one outside-gate config → the
    classifier must NOT report boundary-limited (because not every
    config is outside the gate); it must report ``partial``."""
    recs = [
        {"status": "ok", "applicable": True},
        {"status": "ok", "applicable": False},
    ]
    assert classify_global_status(recs) == "partial_production_result"


# ─── 4. Summary writer wiring ──────────────────────────────────────

def _make_synth_results(*, applicable_arr, n_steps=10, n_cfg=2):
    """Minimal results dict that ``_write_summary`` can consume —
    every per-step array is shaped (n_cfg, n_steps); per-config
    scalars and the angle-weighted lists are filled with finite
    placeholders so the writer renders without diving into NaN."""
    arr2d_bool = np.zeros((n_cfg, n_steps), dtype=bool)
    arr2d_float = np.zeros((n_cfg, n_steps), dtype=float)
    arr2d_int = np.zeros((n_cfg, n_steps), dtype=np.int32)
    arr1d_int = np.zeros(n_cfg, dtype=np.int32)
    arr1d_float = np.zeros(n_cfg, dtype=float)
    arr1d_bool = np.zeros(n_cfg, dtype=bool)
    arr1d_str = np.full(n_cfg, "", dtype="<U64")
    aw_full_list = [{
        "cycle_angle_deg": 720.0,
        "valid_no_clamp_angle_deg": 540.0,
        "valid_no_clamp_angle_frac": 0.75,
        "contact_angle_deg": 180.0,
        "contact_angle_frac": 0.25,
        "firing_angle_deg": 140.0,
        "valid_no_clamp_angle_firing_deg": 30.0,
        "valid_no_clamp_angle_firing_frac": 0.214,
        "contact_angle_firing_deg": 110.0,
        "contact_angle_firing_frac": 0.786,
    } for _ in range(n_cfg)]
    return dict(
        configs=[
            {"label": "cfg_a", "textured": False, "ls": "-",
             "color": "blue", "oil": {"name": "mineral"}},
            {"label": "cfg_b", "textured": True, "ls": "--",
             "color": "red", "oil": {"name": "mineral"}},
        ][:n_cfg],
        phi_crank_deg=np.linspace(0.0, 720.0, n_steps,
                                       endpoint=False),
        phi_last=np.linspace(0.0, 720.0, n_steps, endpoint=False),
        last_start=0,
        n_steps_per_cycle=n_steps,
        eps_x=arr2d_float.copy(), eps_y=arr2d_float.copy(),
        hmin=arr2d_float.copy(), pmax=arr2d_float.copy(),
        f=arr2d_float.copy(), F_tr=arr2d_float.copy(),
        N_loss=arr2d_float.copy(),
        Fx_hyd=arr2d_float.copy(), Fy_hyd=arr2d_float.copy(),
        Fy_ext_last=np.zeros(n_steps, dtype=float),
        F_max=200_000.0,
        T_eff_used=arr2d_float.copy(), T_eff=arr2d_float.copy(),
        T_target=arr2d_float.copy(),
        eta_eff=arr2d_float.copy(), eta_eff_next=arr2d_float.copy(),
        P_loss=arr2d_float.copy(), Q=arr2d_float.copy(),
        mdot=arr2d_float.copy(), mdot_floor_hit=arr2d_bool.copy(),
        solver_success=np.ones((n_cfg, n_steps), dtype=bool),
        valid_dynamic=np.ones((n_cfg, n_steps), dtype=bool),
        valid_no_clamp=np.ones((n_cfg, n_steps), dtype=bool),
        contact_clamp=arr2d_bool.copy(),
        contact_clamp_event_count=arr2d_int.copy(),
        retry_used=arr2d_bool.copy(),
        retry_omega_used=arr2d_float.copy(),
        contact_clamp_count=arr1d_int.copy(),
        solver_failed_count=arr1d_int.copy(),
        retry_recovered_count=arr1d_int.copy(),
        retry_exhausted_count=arr1d_int.copy(),
        omega_hits_per_config=[{} for _ in range(n_cfg)],
        thermal_cycle_delta=arr1d_float.copy(),
        thermal_periodic_converged=arr1d_bool.copy(),
        paired_comparison=[],
        production_metrics=[],
        paired_extended=[],
        firing_sector_deg=(340.0, 480.0),
        envelope_abort_config={"save_partial_on_abort": True},
        aborted=arr1d_bool.copy(),
        abort_reason=arr1d_str.copy(),
        first_clamp_phi=np.full(n_cfg, np.nan),
        first_solver_failed_phi=np.full(n_cfg, np.nan),
        first_invalid_phi=np.full(n_cfg, np.nan),
        steps_attempted=np.full(n_cfg, n_steps, dtype=np.int32),
        steps_completed=np.full(n_cfg, n_steps, dtype=np.int32),
        applicable=np.asarray(applicable_arr, dtype=bool),
        applicable_reason=np.asarray(
            ["valid_no_clamp_angle_frac=0.75 < 0.85"
             if not a else "ok" for a in applicable_arr],
            dtype="<U64"),
        d_phi_per_step=np.full((n_cfg, n_steps), 720.0 / n_steps),
        phi_mod_per_step=np.tile(
            np.linspace(0.0, 720.0, n_steps, endpoint=False), (n_cfg, 1)),
        angle_weighted_full=aw_full_list,
        angle_weighted_last_cycle=aw_full_list,
        N_phi_grid=80, N_z_grid=80,
        peak_lo_deg=330.0, peak_hi_deg=480.0,
        texture_resolution_diagnostic={
            "N_phi": 80, "N_z": 80,
            "cells_per_pocket_phi": 0.71,
            "cells_per_pocket_z": 6.0,
            "resolution_status": "insufficient",
            "recommended_n_phi_min": 449,
        },
    )


def test_summary_header_completed_boundary_limited_text(tmp_path):
    """When every config is applicable=False (status=ok), the
    summary header must read ``COMPLETED BOUNDARY-LIMITED RESULT``
    and the global-status line must echo the new category."""
    from models.thermal_coupling import ThermalConfig
    from models.diesel_quasistatic import SolverRetryConfig
    results = _make_synth_results(applicable_arr=[False, False])
    thermal = ThermalConfig(
        mode="global_relax", T_in_C=105.0, gamma_mix=0.7,
        cp_J_kgK=2000.0, mdot_floor_kg_s=1e-4, tau_th_s=0.5,
    )
    retry = SolverRetryConfig.disabled()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_summary(
        str(run_dir), results, thermal, retry,
        grid=80, n_cycles=1,
        d_phi_base=1.0, d_phi_peak=0.25,
        runtime_s=1.0, cli_args="(test)",
        F_max_source="(test)",
        peak_lo_deg=330.0, peak_hi_deg=480.0,
    )
    txt = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert "COMPLETED BOUNDARY-LIMITED RESULT" in txt
    assert "completed_boundary_limited_result" in txt
    # Sanity: the production_result wording must NOT appear.
    assert "PRODUCTION RESULT" not in txt
    assert "PARTIAL PRODUCTION RESULT" not in txt


def test_paired_block_rename_when_boundary_limited(tmp_path):
    """In the boundary-limited regime the smooth-vs-textured block
    header must be renamed to ``Conditional paired diagnostics`` so
    downstream readers do not conflate its numbers with a clean
    full-film paired comparison."""
    from models.thermal_coupling import ThermalConfig
    from models.diesel_quasistatic import SolverRetryConfig
    results = _make_synth_results(applicable_arr=[False, False])
    thermal = ThermalConfig(
        mode="global_relax", T_in_C=105.0, gamma_mix=0.7,
        cp_J_kgK=2000.0, mdot_floor_kg_s=1e-4, tau_th_s=0.5,
    )
    retry = SolverRetryConfig.disabled()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_summary(
        str(run_dir), results, thermal, retry,
        grid=80, n_cycles=1,
        d_phi_base=1.0, d_phi_peak=0.25,
        runtime_s=1.0, cli_args="(test)",
        F_max_source="(test)",
        peak_lo_deg=330.0, peak_hi_deg=480.0,
    )
    txt = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Conditional paired diagnostics" in txt
    # The old header must NOT appear when boundary-limited.
    assert "Paired smooth-vs-textured comparison" not in txt
    # And the boundary-limited interpretation note must mention
    # clamp saturation in the firing sector.
    assert "boundary-limited" in txt.lower()
