"""Stage J fu-2 Task 29 — Ausas one-step dump-path contract.

Covers:

* trigger evaluation logic (per-rule isolation);
* dump file emission via ``ausas_one_step_with_state`` with
  reserved ``__dump_*__`` keys threaded through ``extra_options``;
* dump-limit honoured (``written`` plateaus, ``suppressed_after_limit``
  increments);
* default disabled path — when ``directory=None`` no files written.

Pure-Python — no GPU. Fake backend stub returns the gpu-reynolds
Task 12 dict shape with explicit nonfinite signals so we can
exercise every trigger.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pytest

from models.diesel_ausas_adapter import (
    DieselAusasState,
    ausas_one_step_with_state,
    set_ausas_backend_for_tests,
)
from models.diesel_ausas_dump_io import (
    MANDATORY_DUMP_KEYS,
    DumpConfig,
    DumpCounters,
    build_dump_filename,
    evaluate_triggers,
    validate_dump_npz,
    write_dump_npz,
)


# ─── Trigger logic ─────────────────────────────────────────────────


def test_trigger_nonfinite_state():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=True,
        failure_kind="nonfinite_state",
        residual=1e-7, n_inner=10, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    assert "nonfinite_state" in triggers


def test_trigger_invalid_input():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=True, failure_kind="invalid_input",
        residual=1e-7, n_inner=10, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    assert "invalid_input" in triggers


def test_trigger_residual_nan():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=False, failure_kind="",
        residual=float("nan"),
        n_inner=10, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    assert "residual_nan" in triggers


def test_trigger_converged_false():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=False, failure_kind="",
        residual=1e-3, n_inner=10, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    assert "converged_false" in triggers


def test_trigger_budget():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=False, failure_kind="",
        residual=1e-3, n_inner=5000, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    assert "budget" in triggers


def test_trigger_force_nan():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=True, failure_kind="",
        residual=1e-7, n_inner=10, n_inner_max=5000,
        F_hyd_x=float("nan"), F_hyd_y=0.0,
    )
    assert "force_nan" in triggers


def test_trigger_no_fire_on_healthy_step():
    cfg = DumpConfig(directory="/tmp", limit=10)
    triggers = evaluate_triggers(
        cfg=cfg, converged=True, failure_kind="",
        residual=1e-7, n_inner=10, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    assert triggers == []


def test_trigger_individual_toggle():
    """Operator can disable individual triggers via DumpConfig flags."""
    cfg = DumpConfig(directory="/tmp", limit=10,
                      on_residual_nan=False)
    triggers = evaluate_triggers(
        cfg=cfg, converged=False, failure_kind="",
        residual=float("nan"),
        n_inner=10, n_inner_max=5000,
        F_hyd_x=0.0, F_hyd_y=0.0,
    )
    # residual_nan trigger silenced; converged=False still fires.
    assert "residual_nan" not in triggers
    assert "converged_false" in triggers


# ─── Filename + writer + validator ─────────────────────────────────


def test_filename_format():
    fname = build_dump_filename(
        step=179, substep=2, trial=14, commit=True,
        primary_trigger="nonfinite_state",
    )
    assert fname == (
        "ausas_call_step0179_sub2_trial014_commit1_nonfinite_state.npz")


def test_writer_and_validator_round_trip(tmp_path):
    """Mandatory-key validator passes when payload covers all keys."""
    payload = {k: 0 for k in MANDATORY_DUMP_KEYS}
    # Override array fields to actual ndarrays.
    for k in ("H_prev", "H_curr", "P_prev", "theta_prev"):
        payload[k] = np.zeros((4, 8))
    target = write_dump_npz(str(tmp_path), "minimal.npz", payload)
    ok, missing = validate_dump_npz(target)
    assert ok, f"missing keys: {missing}"


def test_validator_rejects_incomplete_dump(tmp_path):
    """Dropping a mandatory key → validator reports it."""
    payload = {k: 0 for k in MANDATORY_DUMP_KEYS if k != "H_prev"}
    payload["H_curr"] = np.zeros((4, 8))
    target = write_dump_npz(str(tmp_path), "incomplete.npz", payload)
    ok, missing = validate_dump_npz(target)
    assert not ok
    assert "H_prev" in missing


# ─── End-to-end via ausas_one_step_with_state ──────────────────────


def _fake_backend_returns(out: Dict[str, Any]):
    """Build a fake backend that returns a fixed dict for every call."""
    def _fake(**kwargs):
        H = np.asarray(kwargs["H_curr"], dtype=float)
        # Re-shape pressure arrays to padded shape.
        n_z, n_phi_padded = H.shape
        rv = dict(out)
        # Default P / theta if not in canned output.
        if "P" not in rv:
            rv["P"] = np.full_like(H, 0.5, dtype=float)
        if "theta" not in rv:
            rv["theta"] = np.ones_like(H, dtype=float)
        return rv
    return _fake


@pytest.fixture
def _failing_dict_backend():
    """Backend that always returns failure_kind=nonfinite_state +
    residual_linf=NaN + converged=False — every call triggers the
    dump path."""
    set_ausas_backend_for_tests(_fake_backend_returns(dict(
        residual_linf=float("nan"),
        residual_rms=float("nan"),
        residual_l2_abs=float("nan"),
        n_inner=42, converged=False,
        failure_kind="nonfinite_state",
        nonfinite_count=3,
        first_nan_field="P", first_nan_index=(2, 5),
        first_nan_is_ghost=False,
        first_nan_is_axial_boundary=False,
        first_nan_is_phi_seam=False,
        nan_iter=12,
    )))
    yield
    set_ausas_backend_for_tests(None)


def _call_ausas(extra_options=None) -> None:
    """Single ausas_one_step_with_state call with minimal valid args."""
    N_z, N_phi = 4, 8
    H = np.full((N_z, N_phi), 0.5)
    state = DieselAusasState.from_initial_gap(H)
    return ausas_one_step_with_state(
        state, H_curr=H,
        dt_s=1e-5, omega_shaft=200.0,
        d_phi=2.0 * np.pi / N_phi, d_Z=2.0 / (N_z - 1),
        R=0.1, L=0.14,
        extra_options=extra_options,
        commit=True,
    )


def test_dump_default_disabled_writes_no_files(tmp_path,
                                                 _failing_dict_backend):
    """``DumpConfig.directory=None`` (default) → no files written."""
    cfg = DumpConfig(directory=None)
    counters = DumpCounters()
    _call_ausas(extra_options={
        "tol": 1e-6, "max_inner": 5000,
        "__dump_config__": cfg,
        "__dump_metadata__": dict(step=0, substep=0, trial=0,
                                    phi_deg=0.0, eps_x=0.0, eps_y=0.0,
                                    config_label="test",
                                    trial_kind="picard_trial",
                                    texture_kind="none",
                                    groove_preset="",
                                    cavitation="ausas_dynamic",
                                    F_hyd_x=0.0, F_hyd_y=0.0),
        "__dump_counters__": counters,
    })
    assert counters.written == 0
    assert list(tmp_path.iterdir()) == []


def test_dump_emits_file_on_failure(tmp_path, _failing_dict_backend):
    """When directory is set and triggers fire → exactly one .npz."""
    out = tmp_path / "dumps"
    cfg = DumpConfig(directory=str(out), limit=10)
    counters = DumpCounters()
    _call_ausas(extra_options={
        "tol": 1e-6, "max_inner": 5000,
        "__dump_config__": cfg,
        "__dump_metadata__": dict(step=42, substep=1, trial=7,
                                    phi_deg=180.0,
                                    eps_x=0.5, eps_y=-0.3,
                                    config_label="mineral_textured",
                                    trial_kind="picard_trial",
                                    texture_kind="groove",
                                    groove_preset="g4_same_depth_safe",
                                    cavitation="ausas_dynamic",
                                    F_hyd_x=0.0, F_hyd_y=0.0),
        "__dump_counters__": counters,
    })
    assert counters.written == 1
    files = list(out.iterdir())
    assert len(files) == 1
    # Filename encodes the metadata.
    fn = files[0].name
    assert "step0042" in fn
    assert "trial007" in fn
    assert "commit1" in fn
    assert "nonfinite_state" in fn
    # Mandatory-key validator passes.
    ok, missing = validate_dump_npz(str(files[0]))
    assert ok, f"missing keys: {missing}"


def test_dump_limit_honoured(tmp_path, _failing_dict_backend):
    """``limit=2`` and three failing calls → 2 written, 1 suppressed."""
    out = tmp_path / "dumps"
    cfg = DumpConfig(directory=str(out), limit=2)
    counters = DumpCounters()
    for k in range(3):
        _call_ausas(extra_options={
            "tol": 1e-6, "max_inner": 5000,
            "__dump_config__": cfg,
            "__dump_metadata__": dict(
                step=k, substep=0, trial=0,
                phi_deg=k * 1.0, eps_x=0.1, eps_y=0.0,
                config_label="t",
                trial_kind="picard_trial",
                texture_kind="none", groove_preset="",
                cavitation="ausas_dynamic",
                F_hyd_x=0.0, F_hyd_y=0.0),
            "__dump_counters__": counters,
        })
    assert counters.written == 2
    assert counters.suppressed_after_limit >= 1
    assert counters.by_trigger.get("nonfinite_state", 0) == 3
    assert len(list(out.iterdir())) == 2


def test_dump_skipped_on_healthy_step(tmp_path):
    """Healthy backend (converged=True, finite, no nan signals) →
    no dump written even with directory set."""
    set_ausas_backend_for_tests(_fake_backend_returns(dict(
        residual_linf=1e-7, n_inner=10, converged=True,
    )))
    try:
        out = tmp_path / "dumps"
        cfg = DumpConfig(directory=str(out), limit=10)
        counters = DumpCounters()
        _call_ausas(extra_options={
            "tol": 1e-6, "max_inner": 5000,
            "__dump_config__": cfg,
            "__dump_metadata__": dict(step=0, substep=0, trial=0,
                                        phi_deg=0.0, eps_x=0.0,
                                        eps_y=0.0,
                                        config_label="t",
                                        trial_kind="picard_trial",
                                        texture_kind="none",
                                        groove_preset="",
                                        cavitation="ausas_dynamic",
                                        F_hyd_x=0.0, F_hyd_y=0.0),
            "__dump_counters__": counters,
        })
        assert counters.written == 0
        # Directory may not even exist if no file was written.
        assert not out.exists() or list(out.iterdir()) == []
    finally:
        set_ausas_backend_for_tests(None)


def test_dump_handles_none_diagnostic_fields(tmp_path):
    """Stage J fu-2 Task 29 hotfix — gpu-reynolds emits unset
    diagnostic keys as ``None`` (not absent), so a naive
    ``int(out.get("nan_iter", -1))`` would raise TypeError on
    ``int(None)``. Ensure the safe-coerce helpers tolerate it."""
    set_ausas_backend_for_tests(_fake_backend_returns(dict(
        # Healthy result, but every diagnostic field explicitly None.
        residual_linf=1e-7, n_inner=10, converged=True,
        failure_kind=None,
        first_nan_field=None,
        first_nan_index=None,
        first_nan_is_ghost=None,
        first_nan_is_axial_boundary=None,
        first_nan_is_phi_seam=None,
        nan_iter=None,
        nonfinite_count=None,
        residual_rms=None,
        residual_l2_abs=None,
    )))
    try:
        result = _call_ausas(extra_options={
            "tol": 1e-6, "max_inner": 5000,
        })
        # No crash, healthy result.
        assert result.converged is True
        assert result.failure_kind == ""
        assert result.first_nan_field == ""
        assert result.first_nan_index == ()
        assert result.first_nan_is_ghost is False
        assert result.nan_iter == -1
        assert result.nonfinite_count == 0
        assert np.isnan(result.residual_rms)
        assert np.isnan(result.residual_l2_abs)
    finally:
        set_ausas_backend_for_tests(None)


def test_dump_handles_none_metadata_fields(tmp_path,
                                              _failing_dict_backend):
    """Same hotfix on the metadata side — runner may thread
    ``F_hyd_x=None`` / ``F_hyd_y=None`` for a step where the trial
    hasn't computed forces yet. Dump path must not crash."""
    out = tmp_path / "dumps"
    cfg = DumpConfig(directory=str(out), limit=10)
    counters = DumpCounters()
    _call_ausas(extra_options={
        "tol": 1e-6, "max_inner": 5000,
        "__dump_config__": cfg,
        "__dump_metadata__": dict(
            step=None, substep=None, trial=None,
            phi_deg=None, eps_x=None, eps_y=None,
            config_label=None,
            trial_kind=None, texture_kind=None,
            groove_preset=None, cavitation=None,
            F_hyd_x=None, F_hyd_y=None,
        ),
        "__dump_counters__": counters,
    })
    # Triggered by failure_kind=nonfinite_state from backend, but
    # all None metadata coerced to defaults — file written ok.
    assert counters.written == 1
    files = list(out.iterdir())
    assert len(files) == 1
    ok, missing = validate_dump_npz(str(files[0]))
    assert ok, f"missing keys: {missing}"


def test_dump_routing_keys_stripped_before_backend(tmp_path,
                                                     _failing_dict_backend):
    """The reserved ``__dump_*__`` keys must NOT reach the backend
    as solver kwargs (would TypeError on real GPU). The fake
    backend uses ``**kwargs`` and would silently swallow them, so
    we verify by inspecting the dumped ``solver_kwargs`` snapshot:
    those reserved keys must be absent."""
    out = tmp_path / "dumps"
    cfg = DumpConfig(directory=str(out), limit=10)
    counters = DumpCounters()
    _call_ausas(extra_options={
        "tol": 1e-6, "max_inner": 5000,
        "__dump_config__": cfg,
        "__dump_metadata__": dict(step=0, substep=0, trial=0,
                                    phi_deg=0.0, eps_x=0.0,
                                    eps_y=0.0, config_label="t",
                                    trial_kind="picard_trial",
                                    texture_kind="none",
                                    groove_preset="",
                                    cavitation="ausas_dynamic",
                                    F_hyd_x=0.0, F_hyd_y=0.0),
        "__dump_counters__": counters,
    })
    files = list(out.iterdir())
    assert len(files) == 1
    with np.load(str(files[0]), allow_pickle=True) as nz:
        solver_kwargs = nz["solver_kwargs"].item()
    assert "__dump_config__" not in solver_kwargs
    assert "__dump_metadata__" not in solver_kwargs
    assert "__dump_counters__" not in solver_kwargs
    assert solver_kwargs.get("tol") == pytest.approx(1e-6)
    assert solver_kwargs.get("max_inner") == 5000
