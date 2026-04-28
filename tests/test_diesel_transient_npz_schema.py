"""Stage J followup-2 Step 10 — npz writer schema test.

Pins the contract that ``scripts.run_diesel_thd_transient._save_data``
includes the five Stage J coupling-diagnostic arrays in the
``np.savez`` call. Without this test, the writer's hard-coded
whitelist of fields silently dropped the new ``stage_j_*`` keys
even when ``run_transient`` populated them in the ``results`` dict —
caught only by post-run npz inspection.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

# Load _save_data without triggering scripts.run_diesel_thd_transient
# top-level imports (which require GPU-only reynolds_solver).
SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "run_diesel_thd_transient.py",
)


def _load_save_data():
    """Best-effort: try the normal import; if reynolds_solver is
    missing (CPU box), skip — the contract is verified on GPU
    side runs anyway. The point of this test is to catch
    regressions where _save_data is edited and a stage_j_* key
    drops out of np.savez; on a developer box that lacks the GPU
    deps the test is informational at best.
    """
    spec = importlib.util.spec_from_file_location(
        "run_diesel_thd_transient", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        pytest.skip(f"GPU-only dep missing: {e}")
    return mod._save_data


def _minimal_results(n_cfg: int = 1, n_steps: int = 5) -> dict:
    """A skeletal ``results`` dict with the fields ``_save_data``
    references; enough to drive the writer through one np.savez
    call without hitting KeyError."""
    shape = (n_cfg, n_steps)
    z_int = np.zeros(shape, dtype=np.int32)
    z_float = np.zeros(shape, dtype=float)
    z_bool = np.zeros(shape, dtype=bool)
    z_one = np.ones(shape, dtype=float)
    contact_clamp = np.zeros(shape, dtype=bool)
    return {
        "aborted": np.zeros(n_cfg, dtype=bool),
        "envelope_abort_config": {"save_partial_on_abort": False},
        "phi_crank_deg": z_float.copy(),
        "phi_last": z_float[0].copy(),
        "last_start": 0,
        "n_steps_per_cycle": n_steps,
        "eps_x": z_float.copy(),
        "eps_y": z_float.copy(),
        "hmin": z_one.copy(),
        "pmax": z_float.copy(),
        "f": z_float.copy(),
        "F_tr": z_float.copy(),
        "N_loss": z_float.copy(),
        "Fx_hyd": z_float.copy(),
        "Fy_hyd": z_float.copy(),
        "Fy_ext_last": z_float[0].copy(),
        "F_max": 1.0e3,
        "T_eff_used": np.zeros(n_cfg),
        "T_eff": np.zeros(n_cfg),
        "T_target": np.zeros(n_cfg),
        "eta_eff": np.zeros(n_cfg),
        "eta_eff_next": np.zeros(n_cfg),
        "P_loss": np.zeros(n_cfg),
        "Q": np.zeros(n_cfg),
        "mdot": np.zeros(n_cfg),
        "mdot_floor_hit": np.zeros(n_cfg, dtype=bool),
        "solver_success": z_bool.copy(),
        "valid_dynamic": z_bool.copy(),
        "valid_no_clamp": z_bool.copy(),
        "contact_clamp": contact_clamp,
        "configs": [{"label": "c0", "textured": False}],
        "retry_used": z_bool.copy(),
        "retry_omega_used": z_float.copy(),
        "contact_clamp_count": np.zeros(n_cfg, dtype=np.int32),
        "solver_failed_count": np.zeros(n_cfg, dtype=np.int32),
        "retry_recovered_count": np.zeros(n_cfg, dtype=np.int32),
        "retry_exhausted_count": np.zeros(n_cfg, dtype=np.int32),
        "thermal_cycle_delta": np.zeros(n_cfg),
        "thermal_periodic_converged": np.zeros(n_cfg, dtype=bool),
        "abort_reason": np.array(["none"], dtype=object),
        "first_clamp_phi": np.full(n_cfg, np.nan),
        "first_solver_failed_phi": np.full(n_cfg, np.nan),
        "first_invalid_phi": np.full(n_cfg, np.nan),
        "steps_attempted": np.zeros(n_cfg, dtype=np.int32),
        "steps_completed": np.zeros(n_cfg, dtype=np.int32),
        "applicable": np.zeros(n_cfg, dtype=bool),
        "applicable_reason": np.array(["ok"], dtype=object),
        # Stage J fu-2 Step 10 — populated by run_transient
        # (real arrays). Test data uses identifiable values so
        # we can assert they pass through unchanged.
        "stage_j_picard_shrinks": np.full(shape, 7, dtype=np.int32),
        "stage_j_mech_relax_min_seen": np.full(shape, 0.125,
                                                  dtype=float),
        "stage_j_fp_converged": np.ones(shape, dtype=bool),
        "stage_j_n_trials": np.full(shape, 4, dtype=np.int32),
        "stage_j_rejection_reason": np.full(
            shape, "physical_pressure_above_dim_max", dtype=object),
    }


class _FakeThermal:
    mode = "global_relax"
    gamma_mix = 0.7
    tau_th_s = 0.5
    T_in_C = 105.0
    cp_J_kgK = 2000.0
    mdot_floor_kg_s = 1e-3


class _FakeRetryCfg:
    pass


def test_save_data_writes_all_stage_j_keys(tmp_path):
    """Contract: ``_save_data`` MUST include all 5 stage_j_*
    arrays in the ``np.savez`` call, with values that round-trip
    through the npz file unchanged."""
    save_data = _load_save_data()
    results = _minimal_results()
    save_data(str(tmp_path), results, _FakeThermal(), _FakeRetryCfg())

    npz_path = os.path.join(str(tmp_path), "data.npz")
    assert os.path.isfile(npz_path), (
        f"data.npz not written; tmp_path contents: "
        f"{os.listdir(str(tmp_path))}")
    d = np.load(npz_path, allow_pickle=True)
    expected = [
        "stage_j_picard_shrinks",
        "stage_j_mech_relax_min_seen",
        "stage_j_fp_converged",
        "stage_j_n_trials",
        "stage_j_rejection_reason",
    ]
    for key in expected:
        assert key in d.files, (
            f"{key!r} missing from data.npz; present keys: "
            f"{sorted(d.files)}")
    # Round-trip values verbatim.
    assert int(d["stage_j_picard_shrinks"][0, 0]) == 7
    assert float(d["stage_j_mech_relax_min_seen"][0, 0]) == 0.125
    assert bool(d["stage_j_fp_converged"][0, 0]) is True
    assert int(d["stage_j_n_trials"][0, 0]) == 4
    assert str(d["stage_j_rejection_reason"][0, 0]) == \
        "physical_pressure_above_dim_max"


def test_save_data_falls_back_when_results_missing_stage_j(tmp_path):
    """For older results dicts (regenerate-summary on a pre-Step-10
    npz), the writer MUST fall back to safe defaults rather than
    crashing on KeyError. Defaults: zeros for counts, NaN for
    relax_min, False for fp_converged, "none" for rejection."""
    save_data = _load_save_data()
    results = _minimal_results()
    # Strip the stage_j_* keys to simulate an older results dict.
    for key in (
            "stage_j_picard_shrinks",
            "stage_j_mech_relax_min_seen",
            "stage_j_fp_converged",
            "stage_j_n_trials",
            "stage_j_rejection_reason"):
        del results[key]
    save_data(str(tmp_path), results, _FakeThermal(),
              _FakeRetryCfg())

    d = np.load(os.path.join(str(tmp_path), "data.npz"),
                allow_pickle=True)
    assert "stage_j_picard_shrinks" in d.files
    # Fallback shape must match contact_clamp shape and have safe
    # default values.
    assert d["stage_j_picard_shrinks"].shape == \
        results["contact_clamp"].shape
    assert int(np.sum(d["stage_j_picard_shrinks"])) == 0
    assert np.all(np.isnan(d["stage_j_mech_relax_min_seen"]))
    assert not bool(np.any(d["stage_j_fp_converged"]))
    assert int(np.sum(d["stage_j_n_trials"])) == 0
    assert str(d["stage_j_rejection_reason"][0, 0]) == "none"
