"""Stage Diesel Transient Load-Envelope-0 — abort + sweep contract.

These tests stay pipeline-side (no real solver needed): we monkey-patch
``models.diesel_transient.solve_reynolds`` with a synthetic substitute
that emits the exact failure modes the abort policy is designed to
catch (clamp explosion, persistent NaN, persistent SOR-non-convergence)
and verify each abort_reason code fires at the right threshold,
``data_partial.npz`` lands in the run dir, the next config / next
sweep scale continues cleanly, and the load-envelope CSV carries every
required column.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings as _warnings

import numpy as np
import pytest

from models.diesel_transient import (
    CONFIGS, EnvelopeAbortConfig,
    classify_envelope_per_config, classify_paired_envelope,
)


# ─── 1. Per-config envelope classification thresholds ──────────────

def test_envelope_classification_thresholds_pass():
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=99,
        valid_dynamic_count=95,
        valid_no_clamp_count=90,
        retry_exhausted_count=0,
        aborted=False,
    )
    assert ok is True
    assert reason == "ok"


def test_envelope_classification_thresholds_fail_solver():
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=80,    # 0.80 < 0.95
        valid_dynamic_count=80,
        valid_no_clamp_count=80,
        retry_exhausted_count=0,
        aborted=False,
    )
    assert ok is False
    assert "solver_success_frac" in reason


def test_envelope_classification_thresholds_fail_valid_dynamic():
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=99,
        valid_dynamic_count=70,     # 0.70 < 0.90
        valid_no_clamp_count=70,
        retry_exhausted_count=0,
        aborted=False,
    )
    assert ok is False
    assert "valid_dynamic_frac" in reason


def test_envelope_classification_thresholds_fail_no_clamp():
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=99,
        valid_dynamic_count=95,
        valid_no_clamp_count=80,    # 0.80 < 0.85
        retry_exhausted_count=0,
        aborted=False,
    )
    assert ok is False
    assert "valid_no_clamp_frac" in reason


def test_envelope_classification_aborted_is_not_applicable():
    ok, reason = classify_envelope_per_config(
        n_completed=10,
        solver_success_count=10,
        valid_dynamic_count=10,
        valid_no_clamp_count=10,
        retry_exhausted_count=0,
        aborted=True,
    )
    assert ok is False
    assert reason == "aborted_outside_envelope"


def test_envelope_classification_retry_exhausted_blocks():
    ok, reason = classify_envelope_per_config(
        n_completed=100,
        solver_success_count=99,
        valid_dynamic_count=95,
        valid_no_clamp_count=90,
        retry_exhausted_count=2,    # > 0
        aborted=False,
    )
    assert ok is False
    assert "retry_exhausted" in reason


def test_paired_envelope_threshold():
    ok, _ = classify_paired_envelope(common_no_clamp_count=85,
                                        n_steps_min=100)
    assert ok is True
    ok2, reason = classify_paired_envelope(common_no_clamp_count=75,
                                              n_steps_min=100)
    assert ok2 is False
    assert "common_valid_no_clamp_frac" in reason


# ─── 2. EnvelopeAbortConfig defaults & disabled ────────────────────

def test_envelope_abort_defaults():
    cfg = EnvelopeAbortConfig()
    assert cfg.enabled is True
    assert cfg.clamp_frac_max == pytest.approx(0.30)
    assert cfg.solver_fail_frac_max == pytest.approx(0.30)
    assert cfg.consecutive_invalid_max == 30
    assert cfg.save_partial_on_abort is True
    d = cfg.to_dict()
    for k in ("enabled", "clamp_frac_max", "solver_fail_frac_max",
              "consecutive_invalid_max", "save_partial_on_abort",
              "warmup_steps"):
        assert k in d


def test_envelope_abort_disabled():
    cfg = EnvelopeAbortConfig.disabled()
    assert cfg.enabled is False


# ─── 3. CSV writer column contract ─────────────────────────────────

def _stub_solve_reynolds_factory(*, mode: str = "ok"):
    """``mode``:
      * ``ok``           — every solve succeeds.
      * ``fail_after``   — first 5 succeed, then SOR-non-converge forever.
      * ``always_fail``  — every solve emits SOR-non-converge.
    """
    state = {"n": 0}

    def fake(H, d_phi, d_Z, R, L, **kw):
        state["n"] += 1
        if mode == "ok":
            ok = True
        elif mode == "fail_after":
            ok = state["n"] <= 5
        else:
            ok = False
        if not ok:
            _warnings.warn(
                "SOR не сошёлся: delta=2.0e-04, n_iter=50000",
                UserWarning)
            return (np.full(H.shape, np.nan), 1.0, 50000, False)
        # Successful: small finite non-dim P (must stay <1e10 after
        # multiplication by p_scale ~ 1-10 MPa).
        P = np.full(H.shape, 1.0)
        return (P, 1e-6, 100, True)

    return fake


def test_load_envelope_csv_has_required_columns(tmp_path):
    """The CSV writer must emit every column listed in Section 2."""
    from scripts.run_diesel_thd_transient import (
        _ENVELOPE_CSV_COLUMNS, _write_envelope_csv,
    )
    required = {
        "F_scale", "F_max_used", "config",
        "n_steps_attempted", "n_steps_completed",
        "solver_success_count", "valid_dynamic_count",
        "valid_no_clamp_count",
        "contact_clamp_count", "solver_failed_count",
        "retry_recovered_count", "retry_exhausted_count",
        "first_clamp_phi", "first_invalid_phi",
        "max_epsilon", "min_h_min_valid",
        "p95_pmax_valid", "p99_pmax_valid", "max_pmax_valid",
        "min_T_eff_valid", "mean_T_eff_valid", "max_T_eff_valid",
        "mean_P_loss_valid", "mdot_floor_hit_count",
        "thermal_cycle_delta", "aborted", "abort_reason",
    }
    cols = set(_ENVELOPE_CSV_COLUMNS)
    missing = required - cols
    assert not missing, f"missing CSV columns: {missing}"
    # Round-trip a tiny CSV to verify writer doesn't crash with one row.
    rows = [{c: 0 for c in _ENVELOPE_CSV_COLUMNS}]
    rows[0]["config"] = "test_cfg"
    rows[0]["abort_reason"] = ""
    rows[0]["applicable_reason"] = "ok"
    _write_envelope_csv(str(tmp_path), rows)
    out = (tmp_path / "load_envelope.csv").read_text(encoding="utf-8")
    for col in required:
        assert col in out


# ─── 4. Solver-mocked end-to-end abort tests ───────────────────────

def _run_short(envelope_abort, mode_for_solver, n_cycles=1, n_grid=20,
               d_phi_base=20.0, d_phi_peak=10.0, configs=None,
               F_max=200_000.0):
    import models.diesel_transient as dt
    fake = _stub_solve_reynolds_factory(mode=mode_for_solver)
    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake
        return dt.run_transient(
            F_max=F_max,
            configs=configs or [CONFIGS[0]],
            n_grid=n_grid,
            n_cycles=n_cycles,
            d_phi_base_deg=d_phi_base,
            d_phi_peak_deg=d_phi_peak,
            envelope_abort=envelope_abort,
        )
    finally:
        dt.solve_reynolds = target


def test_abort_on_solver_fail_frac_exceeded():
    """``always_fail`` solver triggers solver_fail_frac_exceeded
    abort within a few warmup steps."""
    cfg = EnvelopeAbortConfig(
        clamp_frac_max=1.0,        # ignore clamp gate
        solver_fail_frac_max=0.30,
        consecutive_invalid_max=10_000,
        warmup_steps=2,
    )
    res = _run_short(cfg, mode_for_solver="always_fail")
    assert bool(res["aborted"][0]) is True
    assert str(res["abort_reason"][0]) == "solver_fail_frac_exceeded"
    assert int(res["steps_completed"][0]) < int(res["steps_attempted"][0]) \
        + 1   # short of full n_steps


def test_abort_consecutive_invalid_exceeded():
    """A small ``consecutive_invalid_max`` plus always-failing solver
    triggers the streak abort first, before the fractional guard."""
    cfg = EnvelopeAbortConfig(
        clamp_frac_max=1.0,
        solver_fail_frac_max=1.0,
        consecutive_invalid_max=4,
        warmup_steps=2,
    )
    res = _run_short(cfg, mode_for_solver="always_fail")
    assert bool(res["aborted"][0]) is True
    assert str(res["abort_reason"][0]) \
        == "consecutive_invalid_exceeded"


def test_abort_on_clamp_frac_exceeded():
    """A run with very thin clearance + heavy initial conditions
    naturally clamps every step, which trips the clamp-fraction
    abort even when the solver would have succeeded.

    We force this via an absurd F_max that pushes the shaft past
    eps_max on the very first Verlet predict.
    """
    cfg = EnvelopeAbortConfig(
        clamp_frac_max=0.30,
        solver_fail_frac_max=1.0,
        consecutive_invalid_max=10_000,
        warmup_steps=2,
    )
    # F_max = 1e8 N is absurd for the bearing; first Verlet step
    # pushes ε past eps_max → clamp every step from the start.
    res = _run_short(cfg, mode_for_solver="ok",
                     F_max=1.0e8, d_phi_base=20.0, d_phi_peak=10.0)
    assert bool(res["aborted"][0]) is True
    assert str(res["abort_reason"][0]) == "clamp_frac_exceeded"


def test_save_partial_on_abort_writes_partial_file(tmp_path):
    """Aborted run lands data_partial.npz, not data.npz."""
    import models.diesel_transient as dt
    from scripts.run_diesel_thd_transient import _save_data
    fake = _stub_solve_reynolds_factory(mode="always_fail")
    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake
        results = dt.run_transient(
            F_max=200_000.0,
            configs=[CONFIGS[0]],
            n_grid=20, n_cycles=1,
            d_phi_base_deg=20.0, d_phi_peak_deg=10.0,
            envelope_abort=EnvelopeAbortConfig(
                clamp_frac_max=1.0,
                solver_fail_frac_max=0.10,
                consecutive_invalid_max=10_000,
                warmup_steps=2,
                save_partial_on_abort=True,
            ),
        )
    finally:
        dt.solve_reynolds = target
    from models.thermal_coupling import ThermalConfig
    from models.diesel_quasistatic import SolverRetryConfig
    _save_data(str(tmp_path), results,
               ThermalConfig(), SolverRetryConfig())
    files = sorted(os.listdir(str(tmp_path)))
    assert "data_partial.npz" in files
    assert "data.npz" not in files


# ─── 5. Stage Transient Summary Wording Fix ────────────────────────


def test_global_status_all_ok_is_production_result():
    from scripts.run_diesel_thd_transient import classify_global_status
    recs = [{"status": "ok"}, {"status": "ok"}, {"status": "ok"}]
    assert classify_global_status(recs) == "production_result"


def test_global_status_all_aborted_is_aborted():
    from scripts.run_diesel_thd_transient import classify_global_status
    recs = [{"status": "aborted_outside_envelope"},
            {"status": "aborted_outside_envelope"}]
    assert classify_global_status(recs) == "aborted_outside_envelope"


def test_global_status_mixed_is_partial():
    from scripts.run_diesel_thd_transient import classify_global_status
    recs = [
        {"status": "ok"},
        {"status": "aborted_outside_envelope"},
        {"status": "ok"},
        {"status": "ok"},
    ]
    assert classify_global_status(recs) == "partial_production_result"


def test_per_config_status_line_full_applicable():
    from scripts.run_diesel_thd_transient import per_config_status_line
    rec = {"status": "ok"}
    assert per_config_status_line(rec, 1.0) == "full / applicable"
    assert per_config_status_line(rec, 0.96) == "full / applicable"


def test_per_config_status_line_near_edge():
    from scripts.run_diesel_thd_transient import per_config_status_line
    rec = {"status": "ok"}
    assert per_config_status_line(rec, 0.83) \
        == "full / near-edge applicable"
    assert per_config_status_line(rec, 0.94) \
        == "full / near-edge applicable"


def test_per_config_status_line_aborted_overrides_frac():
    from scripts.run_diesel_thd_transient import per_config_status_line
    rec = {"status": "aborted_outside_envelope"}
    # Even with frac=1.0 the aborted status wins.
    assert per_config_status_line(rec, 1.0) \
        == "aborted_outside_envelope"


def test_sweep_handles_aborted_scale_cleanly():
    """Sweep loop must not stop after an aborted scale — the next
    config / next scale runs anyway. We simulate by running two
    configs back to back through run_transient: when the first config
    aborts, the second still gets attempted and produces its own
    aborted/applicable record without crashing.
    """
    import models.diesel_transient as dt
    fake = _stub_solve_reynolds_factory(mode="always_fail")
    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake
        res = dt.run_transient(
            F_max=200_000.0,
            configs=[CONFIGS[0], CONFIGS[2]],   # smooth + textured
            n_grid=20, n_cycles=1,
            d_phi_base_deg=20.0, d_phi_peak_deg=10.0,
            envelope_abort=EnvelopeAbortConfig(
                clamp_frac_max=1.0,
                solver_fail_frac_max=0.10,
                consecutive_invalid_max=10_000,
                warmup_steps=2,
            ),
        )
    finally:
        dt.solve_reynolds = target
    # Both configs aborted, both have a recorded reason — and the
    # second config attempted some steps (i.e. the runner did not
    # crash after the first abort).
    assert bool(res["aborted"][0]) is True
    assert bool(res["aborted"][1]) is True
    assert int(res["steps_attempted"][1]) > 0
