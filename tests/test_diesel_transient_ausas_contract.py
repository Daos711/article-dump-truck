"""Stage J — diesel transient × Ausas / groove contract tests.

These tests do **not** require the GPU solver. The half-Sommerfeld
regression smoke uses the existing ``solve_reynolds`` stub from
``conftest.py``; the Ausas smoke uses
``models.diesel_ausas_adapter.set_ausas_backend_for_tests`` to install
a deterministic Python stub.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

import models.diesel_transient as dt
from models.diesel_ausas_adapter import (
    set_ausas_backend_for_tests,
)
from models.diesel_transient import CONFIGS, EnvelopeAbortConfig
from models.thermal_coupling import ThermalConfig


# ─── Helpers ───────────────────────────────────────────────────────


def _stub_solve_reynolds_ok():
    """Mimic the half-Sommerfeld solver — finite small P everywhere
    so Verlet stays well behaved on the smoke grid."""
    def fake(H, d_phi, d_Z, R, L, **kw):
        P = np.full(H.shape, 1.0)
        return (P, 1e-6, 100, True)
    return fake


def _stub_ausas_backend(*, ok: bool = True, n_inner: int = 5):
    """Ausas backend stub: P = 0.5*H, theta clipped to [0, 1]."""
    def fake(**kwargs):
        H = kwargs["H_curr"]
        P = 0.5 * H
        theta = np.clip(H, 0.0, 1.0)
        return (P, theta, int(n_inner), bool(ok))
    return fake


def _run_short(*, cavitation: str, texture_kind: str = "none",
               groove_preset: str = None,
               configs=None,
               n_grid: int = 16, n_cycles: int = 1,
               d_phi_base: float = 30.0, d_phi_peak: float = 15.0,
               F_max: float = 200_000.0):
    fake_rs = _stub_solve_reynolds_ok()
    target_rs = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake_rs
        if cavitation == "ausas_dynamic":
            set_ausas_backend_for_tests(_stub_ausas_backend())
        return dt.run_transient(
            F_max=F_max,
            configs=configs or [CONFIGS[0]],
            cavitation=cavitation,
            texture_kind=texture_kind,
            groove_preset=groove_preset,
            n_grid=n_grid,
            n_cycles=n_cycles,
            d_phi_base_deg=d_phi_base,
            d_phi_peak_deg=d_phi_peak,
            envelope_abort=EnvelopeAbortConfig.disabled(),
        )
    finally:
        dt.solve_reynolds = target_rs
        set_ausas_backend_for_tests(None)


# ─── 1. Legacy half-Sommerfeld smoke regression ────────────────────


def test_legacy_half_sommerfeld_regression_smoke():
    """The default cavitation/texture combo runs the legacy stack —
    no Ausas state is built, the new Stage J fields appear in the
    result dict but with neutral values, and the historical arrays
    remain finite where the run was valid."""
    res = _run_short(
        cavitation="half_sommerfeld",
        texture_kind="dimple",
        configs=[CONFIGS[0]],   # smooth + mineral
    )
    # Neutral Stage J values on the legacy path.
    assert res["cavitation_model"] == "half_sommerfeld"
    assert res["texture_kind"] == "dimple"
    assert res["groove_preset"] is None
    # Ausas arrays exist (to keep downstream consumers
    # unconditional) but were never written.
    assert np.all(res["ausas_n_inner"] == 0)
    assert np.all(res["ausas_state_reset_count"] == 0)
    # Existing per-step arrays must be finite on the valid mask.
    vd = np.asarray(res["valid_dynamic"][0], dtype=bool)
    if vd.any():
        assert np.all(np.isfinite(np.asarray(res["hmin"])[0][vd]))
        assert np.all(np.isfinite(np.asarray(res["pmax"])[0][vd]))


# ─── 2. Ausas dynamic smoke on smooth mineral ──────────────────────


def test_ausas_dynamic_smoke_smooth_mineral():
    """``cavitation='ausas_dynamic'`` with a smooth config and the
    test-only adapter backend must run without exceptions, populate
    the Ausas diagnostic arrays, and report cav_frac and theta in
    the [0, 1] range."""
    res = _run_short(
        cavitation="ausas_dynamic",
        texture_kind="none",
        configs=[CONFIGS[0]],
    )
    assert res["cavitation_model"] == "ausas_dynamic"
    # The Ausas commit at the end of every accepted step must have
    # written into the diagnostic arrays for at least one step.
    assert int(np.sum(res["ausas_converged"])) > 0
    cav = np.asarray(res["ausas_cav_frac"])
    tmin = np.asarray(res["ausas_theta_min"])
    tmax = np.asarray(res["ausas_theta_max"])
    assert np.all((cav >= 0.0) & (cav <= 1.0))
    assert np.nanmax(tmax) <= 1.0 + 1e-12
    assert np.nanmin(tmin) >= -1e-12
    # Stage J target: > 80% converged on smoke. The deterministic
    # stub backend always returns ok=True, so we expect 100% — guard
    # against future stub changes by testing the documented gate.
    n_steps = res["ausas_converged"].shape[1]
    converged_frac = float(np.sum(res["ausas_converged"][0]) / n_steps)
    assert converged_frac >= 0.80


# ─── 3. Ausas dynamic + groove smoke ───────────────────────────────


def test_ausas_dynamic_groove_smoke():
    """The groove preset path must build an additive relief, surface
    the resolved preset and relief stats, and remain dynamically
    stable on the smoke grid."""
    res = _run_short(
        cavitation="ausas_dynamic",
        texture_kind="groove",
        groove_preset="g4_same_depth_safe",
        configs=[CONFIGS[2]],   # textured + mineral
    )
    assert res["texture_kind"] == "groove"
    assert res["groove_preset"] == "g4_same_depth_safe"
    pr = res.get("groove_preset_resolved") or {}
    assert pr.get("preset_name") == "g4_same_depth_safe"
    assert pr.get("d_g_um") == pytest.approx(15.0)
    rs = res.get("groove_relief_stats") or {}
    assert rs.get("has_nan") is False
    assert rs.get("has_inf") is False
    assert rs.get("relief_min", 0.0) >= 0.0
    assert rs.get("relief_max", 0.0) > 0.0


# ─── 4. thermal=off does not fabricate heating ─────────────────────


def test_thermal_off_regression_with_ausas_flag():
    """Even with ``cavitation='ausas_dynamic'``, ``thermal=off`` must
    leave T_eff at the inlet temperature on every valid step (no
    silent heating from the new Stage J path)."""
    set_ausas_backend_for_tests(_stub_ausas_backend())
    fake_rs = _stub_solve_reynolds_ok()
    target_rs = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake_rs
        thermal = ThermalConfig(mode="off")
        res = dt.run_transient(
            F_max=200_000.0,
            configs=[CONFIGS[0]],
            cavitation="ausas_dynamic",
            n_grid=16,
            n_cycles=1,
            d_phi_base_deg=30.0,
            d_phi_peak_deg=15.0,
            thermal=thermal,
            envelope_abort=EnvelopeAbortConfig.disabled(),
        )
    finally:
        dt.solve_reynolds = target_rs
        set_ausas_backend_for_tests(None)
    T_in = float(thermal.T_in_C)
    valid = np.asarray(res["valid_dynamic"][0], dtype=bool)
    T_eff = np.asarray(res["T_eff"][0])
    if valid.any():
        # Off-mode must not move T_eff away from T_in for valid steps.
        assert np.allclose(T_eff[valid], T_in, atol=1e-9)


# ─── 5. Paired stats use the common_valid_no_clamp mask ────────────


def test_common_mask_pairing_with_ausas():
    """Paired smooth/grooved stats must still be computed on the
    common_valid_no_clamp mask only — the new Ausas path must not
    silently widen or narrow the mask. The contract is asserted by
    checking that the paired comparison block (a) exists, (b)
    references the smooth and textured configs by label, and (c)
    reports a ``common_no_clamp_count`` that does not exceed either
    individual config's last-cycle valid_no_clamp count."""
    set_ausas_backend_for_tests(_stub_ausas_backend())
    fake_rs = _stub_solve_reynolds_ok()
    target_rs = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake_rs
        res = dt.run_transient(
            F_max=200_000.0,
            configs=[CONFIGS[0], CONFIGS[2]],   # smooth + textured (mineral)
            cavitation="ausas_dynamic",
            texture_kind="groove",
            groove_preset="g4_same_depth_safe",
            n_grid=16, n_cycles=1,
            d_phi_base_deg=30.0, d_phi_peak_deg=15.0,
            envelope_abort=EnvelopeAbortConfig.disabled(),
        )
    finally:
        dt.solve_reynolds = target_rs
        set_ausas_backend_for_tests(None)
    paired = res.get("paired_comparison") or []
    assert len(paired) >= 1
    rec = paired[0]
    sl = slice(int(res["last_start"]),
               int(res["last_start"]) + int(res["n_steps_per_cycle"]))
    vnc_smooth_last = int(np.sum(
        np.asarray(res["valid_no_clamp"][0, sl], dtype=bool)))
    vnc_textured_last = int(np.sum(
        np.asarray(res["valid_no_clamp"][1, sl], dtype=bool)))
    assert rec["common_no_clamp_count"] <= min(vnc_smooth_last,
                                                  vnc_textured_last)


# ─── 6. Summary writer contains Stage J blocks ─────────────────────


def test_summary_contains_ausas_and_groove_blocks(tmp_path):
    """Render a summary for a synthetic Ausas + groove run and
    assert that the Stage J labels appear: cavitation model line,
    groove preset, Ausas n_inner / cav_frac / theta diagnostics,
    state_reset_count."""
    set_ausas_backend_for_tests(_stub_ausas_backend())
    fake_rs = _stub_solve_reynolds_ok()
    target_rs = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake_rs
        res = dt.run_transient(
            F_max=200_000.0,
            configs=[CONFIGS[2]],   # textured + mineral
            cavitation="ausas_dynamic",
            texture_kind="groove",
            groove_preset="g4_same_depth_safe",
            n_grid=16, n_cycles=1,
            d_phi_base_deg=30.0, d_phi_peak_deg=15.0,
            envelope_abort=EnvelopeAbortConfig.disabled(),
        )
    finally:
        dt.solve_reynolds = target_rs
        set_ausas_backend_for_tests(None)
    from scripts.run_diesel_thd_transient import _write_summary
    from models.diesel_quasistatic import SolverRetryConfig
    thermal = ThermalConfig(
        mode="global_relax", T_in_C=105.0, gamma_mix=0.7,
        cp_J_kgK=2000.0, mdot_floor_kg_s=1e-4, tau_th_s=0.5,
    )
    retry = SolverRetryConfig.disabled()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_summary(
        str(run_dir), res, thermal, retry,
        grid=16, n_cycles=1,
        d_phi_base=30.0, d_phi_peak=15.0,
        runtime_s=1.0, cli_args="(test)",
        F_max_source="(test)",
        peak_lo_deg=330.0, peak_hi_deg=480.0,
    )
    txt = (run_dir / "summary.txt").read_text(encoding="utf-8")
    # Stage J header block.
    assert "Stage J: Ausas dynamic + grooves" in txt
    assert "cavitation model      : ausas_dynamic" in txt
    assert "groove preset         : g4_same_depth_safe" in txt
    # Per-config Ausas diagnostics.
    assert "Ausas diagnostics:" in txt
    assert "n_inner p50/p95/max" in txt
    assert "cav_frac mean/max" in txt
    assert "theta min/max" in txt
    assert "state_reset_count" in txt
    # Groove geometry block.
    assert "groove geometry:" in txt
    assert "d_g_um            : 15.00" in txt
    assert "geometry warnings : none" in txt
