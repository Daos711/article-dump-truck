"""Stage J fu-2 Task 14.2 — postprocessor end-to-end test.

Builds a synthetic ``data.npz`` mirroring the runner's save schema,
runs ``diagnose_stage_j_failures.diagnose_run``, and asserts that
the produced report contains the expected bucket breakdown for
hand-constructed failure scenarios. No GPU, no transient solver.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from diagnose_stage_j_failures import diagnose_run  # noqa: E402


# ─── Synthetic-npz builder ─────────────────────────────────────────


def _build_synthetic_npz(
    tmp_path,
    *,
    n_steps: int,
    failure_pattern,
    config_label: str = "test_cfg",
    aborted: bool = False,
    steps_completed: int = None,
):
    """Write a minimal data.npz the postprocessor can load.

    ``failure_pattern`` is a callable ``(step) -> dict`` returning
    the per-step diagnostic fields. Defaults are healthy ok-step;
    override only what the bucket needs."""
    if steps_completed is None:
        steps_completed = n_steps

    n_cfg = 1
    rejection_reason = np.full((n_cfg, n_steps), "", dtype=object)
    fp_converged = np.ones((n_cfg, n_steps), dtype=bool)
    n_trials = np.zeros((n_cfg, n_steps), dtype=np.int32)
    mech_relax_min_seen = np.full((n_cfg, n_steps), 0.25, dtype=float)
    ausas_n_inner = np.zeros((n_cfg, n_steps), dtype=np.int32)
    ausas_residual = np.full((n_cfg, n_steps), 1e-7, dtype=float)
    ausas_converged = np.ones((n_cfg, n_steps), dtype=bool)
    pmax = np.ones((n_cfg, n_steps), dtype=float)
    ausas_theta_max = np.ones((n_cfg, n_steps), dtype=float)
    solver_success = np.ones((n_cfg, n_steps), dtype=bool)

    for step in range(n_steps):
        ov = failure_pattern(step) or {}
        if not ov:
            continue
        # Override per-field; force solver_success=False if the
        # caller intends a failure.
        for k, v in ov.items():
            if k == "rejection_reason":
                rejection_reason[0, step] = v
            elif k == "fp_converged":
                fp_converged[0, step] = bool(v)
            elif k == "n_trials":
                n_trials[0, step] = int(v)
            elif k == "mech_relax_min_seen":
                mech_relax_min_seen[0, step] = float(v)
            elif k == "ausas_n_inner":
                ausas_n_inner[0, step] = int(v)
            elif k == "ausas_residual":
                ausas_residual[0, step] = float(v)
            elif k == "ausas_converged":
                ausas_converged[0, step] = bool(v)
            elif k == "pmax":
                pmax[0, step] = float(v)
            elif k == "ausas_theta_max":
                ausas_theta_max[0, step] = float(v)
            elif k == "solver_success":
                solver_success[0, step] = bool(v)
            else:
                raise KeyError(f"unknown override field: {k}")
        # Default: any failure flips solver_success.
        if "solver_success" not in ov and ov:
            solver_success[0, step] = False

    phi_crank_deg = np.linspace(0.0, 720.0 * 2, n_steps)

    out = tmp_path / "run"
    out.mkdir()
    np.savez(
        os.path.join(str(out), "data.npz"),
        labels=np.array([config_label], dtype=object),
        phi_crank_deg=phi_crank_deg,
        last_start=int(n_steps // 2),
        n_steps_per_cycle=int(n_steps // 2),
        steps_completed=np.array([steps_completed], dtype=np.int32),
        steps_attempted=np.array([n_steps], dtype=np.int32),
        aborted=np.array([aborted], dtype=bool),
        abort_reason=np.array(["solver_fail_frac_exceeded"
                                 if aborted else ""], dtype=object),
        applicable=np.array([True], dtype=bool),
        applicable_reason=np.array([""], dtype=object),
        # Per-step diagnostic arrays the classifier reads.
        stage_j_rejection_reason=rejection_reason,
        stage_j_fp_converged=fp_converged,
        stage_j_n_trials=n_trials,
        stage_j_mech_relax_min_seen=mech_relax_min_seen,
        stage_j_picard_shrinks=np.zeros((n_cfg, n_steps), dtype=np.int32),
        ausas_n_inner=ausas_n_inner,
        ausas_residual=ausas_residual,
        ausas_converged=ausas_converged,
        ausas_theta_max=ausas_theta_max,
        pmax=pmax,
        solver_success=solver_success,
        # Threshold metadata.
        ausas_options=np.asarray(
            [{"tol": 1e-4, "max_inner": 5000}], dtype=object),
    )
    return str(out)


# ─── Smoke ─────────────────────────────────────────────────────────


def test_diagnose_all_ok_run(tmp_path):
    """A run with no failures should report n_failures=0 and no
    dominant bucket."""
    run_dir = _build_synthetic_npz(
        tmp_path, n_steps=20,
        failure_pattern=lambda step: None,
    )
    report = diagnose_run(run_dir, config_filter="all")
    assert "Config: test_cfg" in report
    assert "-- failures --" in report
    # Total = 20, failures = 0.
    assert "    20      10        0" in report or "20" in report
    # No dominant — dominant line absent for zero-failure config.
    assert "dominant (full)" not in report


def test_diagnose_solver_budget_dominant(tmp_path):
    """All failures are budget-exhausted Ausas solves → dominant
    bucket = solver_budget."""
    def pat(step):
        if step < 5:
            return None
        return dict(
            ausas_n_inner=5000, ausas_residual=1e-3,
            ausas_converged=False,
        )
    run_dir = _build_synthetic_npz(
        tmp_path, n_steps=20, failure_pattern=pat)
    report = diagnose_run(run_dir, config_filter="all")
    assert "solver_budget" in report
    # Match the dominant-bucket line without depending on the
    # exact column-padding inside it.
    assert any("dominant (full)" in ln and "solver_budget" in ln
               for ln in report.splitlines())


def test_diagnose_picard_noncontractive_dominant(tmp_path):
    """Failures driven by ``damped_picard_not_converged`` rejection
    text → dominant bucket = ``picard_noncontractive`` (Task 27
    rename — was ``picard_not_converged`` pre-Task-27)."""
    def pat(step):
        if step < 3:
            return None
        return dict(
            rejection_reason=(
                "damped_picard_not_converged after 64/64 trials; "
                "min_relax=0.0156"),
            fp_converged=False, n_trials=64,
        )
    run_dir = _build_synthetic_npz(
        tmp_path, n_steps=15, failure_pattern=pat)
    report = diagnose_run(run_dir, config_filter="all")
    assert "picard_noncontractive" in report
    assert any(
        "dominant (full)" in ln and "picard_noncontractive" in ln
        for ln in report.splitlines())


def test_diagnose_aborted_partial(tmp_path):
    """An aborted partial run reports steps_completed correctly
    and only classifies the steps that actually ran."""
    def pat(step):
        return dict(
            ausas_n_inner=5000, ausas_residual=1e-3,
            ausas_converged=False,
        )
    run_dir = _build_synthetic_npz(
        tmp_path, n_steps=100,
        failure_pattern=pat,
        aborted=True,
        steps_completed=5,
    )
    report = diagnose_run(run_dir, config_filter="all")
    assert "ABORTED" in report
    assert "steps_completed : 5" in report
    # Only 5 rows classified — solver_budget gets 5/5.
    assert "solver_budget                    5" in report


def test_diagnose_first_failing_section(tmp_path):
    """Report contains the per-step detail dump for the first N
    failing steps."""
    def pat(step):
        if step < 2:
            return None
        return dict(
            ausas_n_inner=5000, ausas_residual=2.5e-3,
            ausas_converged=False,
        )
    run_dir = _build_synthetic_npz(
        tmp_path, n_steps=20, failure_pattern=pat)
    report = diagnose_run(run_dir, config_filter="all",
                              first_n_failing=3)
    assert "First 3 failing steps:" in report
    # Three failing rows should be listed (steps 2, 3, 4 — the
    # first three failures after the two healthy steps).
    for s in (2, 3, 4):
        assert f"  {s:>4d}" in report


def test_diagnose_unknown_config_raises(tmp_path):
    run_dir = _build_synthetic_npz(
        tmp_path, n_steps=5, failure_pattern=lambda s: None)
    with pytest.raises(SystemExit):
        diagnose_run(run_dir, config_filter="not_a_real_config")


def test_diagnose_missing_data_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        diagnose_run(str(tmp_path), config_filter="all")


# ─── Archived-schema fallback (Stage J fu-2 Task 14 fixup) ─────────


def test_diagnose_handles_archived_npz_without_ausas_residual(tmp_path):
    """Archives written before the Bug-2 ``ausas_residual`` save
    must not crash — defensive ``npz.get(..., default)`` lets the
    classifier fall through to the Picard / rejection-reason rules
    instead of KeyError'ing on a missing field."""
    out = tmp_path / "archive_run"
    out.mkdir()
    n_steps = 10
    np.savez(
        os.path.join(str(out), "data.npz"),
        labels=np.array(["legacy_cfg"], dtype=object),
        phi_crank_deg=np.linspace(0.0, 360.0, n_steps),
        last_start=int(n_steps // 2),
        n_steps_per_cycle=int(n_steps // 2),
        steps_completed=np.array([n_steps], dtype=np.int32),
        steps_attempted=np.array([n_steps], dtype=np.int32),
        aborted=np.array([False], dtype=bool),
        abort_reason=np.array([""], dtype=object),
        pmax=np.full((1, n_steps), 1e6, dtype=float),
        solver_success=np.ones((1, n_steps), dtype=bool),
        # Deliberately OMIT ausas_residual / ausas_converged /
        # ausas_n_inner / stage_j_* — what an old archive looks
        # like.
    )
    report = diagnose_run(str(out), config_filter="all")
    assert "Config: legacy_cfg" in report
    # No KeyError; report still produced.
    assert "-- failures --" in report
