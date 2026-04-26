"""Stage Diesel Transient PeakWindow GridDiagnostic — contract tests.

Two numerical risks the patch addresses:

1. The fine-d_phi peak window used to terminate at 420° while the
   production metrics window extends to 480°; recovery angles after
   the firing peak were under-resolved by the coarse base step.
2. An isotropic N×N grid under-resolves the elliptical texture pocket
   in the φ direction (cells_per_pocket_phi may drop below 4); the
   patch surfaces an anisotropic Nφ × N_Z grid via the runner +
   exposes ``texture_resolution_diagnostic`` with a recommendation.

These tests exercise the helper math, the public-API kwargs through
``run_transient`` (without invoking the real solver), and the CLI
plumbing for the new flags. None of them depends on the GPU solver.
"""
from __future__ import annotations

import argparse
import inspect

import numpy as np
import pytest

import models.diesel_transient as dt
from models.diesel_transient import (
    get_step_deg, run_transient, texture_resolution_diagnostic,
)


# ─── 1. get_step_deg: peak window default + bounds ─────────────────

def test_get_step_deg_default_peak_hi_is_480():
    """The new default fine window must extend to 480° (was 420°)."""
    sig = inspect.signature(get_step_deg)
    assert sig.parameters["peak_hi_deg"].default == pytest.approx(480.0)
    assert sig.parameters["peak_lo_deg"].default == pytest.approx(330.0)


def test_get_step_deg_returns_peak_step_at_phi_421():
    """φ=421° used to fall outside the peak window; with default
    peak_hi=480° it must now return the fine peak step."""
    step = get_step_deg(
        421.0,
        d_phi_base_deg=10.0,
        d_phi_peak_deg=0.25,
    )
    assert step == pytest.approx(0.25)


def test_get_step_deg_returns_peak_step_at_phi_478():
    """Just inside the new upper bound (478° < 480°) — still fine."""
    step = get_step_deg(
        478.0,
        d_phi_base_deg=10.0,
        d_phi_peak_deg=0.25,
    )
    assert step == pytest.approx(0.25)


def test_get_step_deg_returns_base_step_at_phi_490():
    """Past the new upper bound (490° > 480°) — back to base step."""
    step = get_step_deg(
        490.0,
        d_phi_base_deg=10.0,
        d_phi_peak_deg=0.25,
    )
    assert step == pytest.approx(10.0)


def test_get_step_deg_custom_window_via_kwargs():
    """Caller can override the window — kwargs must propagate."""
    # Custom narrow window 350°-400°: 380° fine, 410° base.
    inside = get_step_deg(
        380.0,
        d_phi_base_deg=5.0, d_phi_peak_deg=0.5,
        peak_lo_deg=350.0, peak_hi_deg=400.0,
    )
    outside = get_step_deg(
        410.0,
        d_phi_base_deg=5.0, d_phi_peak_deg=0.5,
        peak_lo_deg=350.0, peak_hi_deg=400.0,
    )
    assert inside == pytest.approx(0.5)
    assert outside == pytest.approx(5.0)


# ─── 2. run_transient public-API kwargs ────────────────────────────

def test_run_transient_passes_peak_window_kwargs():
    """``run_transient`` must accept the new peak-window + anisotropic
    grid kwargs in its signature so the CLI can wire them through."""
    sig = inspect.signature(run_transient)
    for name, default in (
        ("peak_lo_deg", 330.0),
        ("peak_hi_deg", 480.0),
        ("n_phi_grid", None),
        ("n_z_grid", None),
    ):
        assert name in sig.parameters, (
            f"run_transient missing new kwarg {name!r}")
        assert sig.parameters[name].default == default, (
            f"run_transient.{name} default = "
            f"{sig.parameters[name].default!r}, expected {default!r}")


# ─── 3. texture_resolution_diagnostic helper ───────────────────────

def test_texture_resolution_diagnostic_n80_insufficient():
    """At N_phi=80 with the typical pocket geometry the helper must
    flag the grid as insufficient and recommend a much larger N_phi.
    """
    diag = texture_resolution_diagnostic(
        N_phi=80, N_z=80,
        R=0.05, L=0.055,
        a_dim=0.003, b_dim=0.0028,
    )
    # cells_per_pocket_phi = 80 * 0.0028 / (π * 0.05) ≈ 1.43.
    assert diag["cells_per_pocket_phi"] < 4.0
    assert diag["resolution_status"] == "insufficient"
    # Recommendation must be a positive integer well above 80.
    assert isinstance(diag["recommended_n_phi_min"], int)
    assert diag["recommended_n_phi_min"] > 80
    # And recommended_n_phi_min itself must satisfy cells>=4.
    n_min = diag["recommended_n_phi_min"]
    assert n_min * 0.0028 / (np.pi * 0.05) >= 4.0


def test_texture_resolution_diagnostic_n_phi_360_sufficient():
    """N_phi=360 with the same pocket geometry yields >4 cells per
    pocket — must be classified as 'ok' (or at worst 'marginal')."""
    diag = texture_resolution_diagnostic(
        N_phi=360, N_z=160,
        R=0.05, L=0.055,
        a_dim=0.003, b_dim=0.0028,
    )
    # cells_per_pocket_phi = 360 * 0.0028 / (π * 0.05) ≈ 6.42 → ok.
    assert diag["cells_per_pocket_phi"] >= 4.0
    assert diag["resolution_status"] in ("ok", "marginal")
    # With ~6.4 cells we expect 'ok' specifically.
    assert diag["resolution_status"] == "ok"
    # Axial: cells_per_pocket_z = 2 * 160 * 0.003 / 0.055 ≈ 17.5.
    assert diag["cells_per_pocket_z"] > 4.0


# ─── 4. CLI flag plumbing for anisotropic grid ─────────────────────

def _build_cli_parser():
    """Reflectively rebuild the script's argparse parser so we don't
    have to actually run it. ``run_diesel_thd_transient.main`` uses
    ``argparse.ArgumentParser.parse_args(argv)`` internally — we
    intercept that and capture the namespace."""
    import scripts.run_diesel_thd_transient as runscript

    captured = {}
    orig_parse = argparse.ArgumentParser.parse_args

    def _capture(self, argv=None, namespace=None):
        ns = orig_parse(self, argv, namespace)
        captured["ns"] = ns
        # Short-circuit so main() does not actually run anything.
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(SystemExit):
            runscript.main([
                "--n-phi", "360",
                "--n-z", "160",
                "--peak-lo-deg", "340.0",
                "--peak-hi-deg", "470.0",
            ])
    finally:
        argparse.ArgumentParser.parse_args = orig_parse
    return captured["ns"]


def test_anisotropic_cli_resolves_n_phi_and_n_z():
    """``--n-phi`` / ``--n-z`` / ``--peak-lo-deg`` / ``--peak-hi-deg``
    all parse and land on the namespace with the expected values."""
    ns = _build_cli_parser()
    assert ns.n_phi == 360
    assert ns.n_z == 160
    assert ns.peak_lo_deg == pytest.approx(340.0)
    assert ns.peak_hi_deg == pytest.approx(470.0)


def test_anisotropic_falls_back_to_n_grid_when_unspecified():
    """When neither ``--n-phi`` nor ``--n-z`` is given the namespace
    leaves them None so ``run_transient`` falls back to ``n_grid``
    (legacy isotropic behaviour preserved)."""
    import scripts.run_diesel_thd_transient as runscript

    captured = {}
    orig_parse = argparse.ArgumentParser.parse_args

    def _capture(self, argv=None, namespace=None):
        ns = orig_parse(self, argv, namespace)
        captured["ns"] = ns
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(SystemExit):
            runscript.main(["--n-grid", "200"])
    finally:
        argparse.ArgumentParser.parse_args = orig_parse
    ns = captured["ns"]
    assert ns.n_grid == 200
    assert ns.n_phi is None
    assert ns.n_z is None
    # Defaults for the peak window must match the new contract.
    assert ns.peak_lo_deg == pytest.approx(330.0)
    assert ns.peak_hi_deg == pytest.approx(480.0)
