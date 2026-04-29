"""Stage J fu-2 fixup-1 — texture-resolution diagnostic contract.

Pins the ``texture_kind``-aware behaviour of
:func:`models.diesel_transient.texture_resolution_diagnostic`:

* the dimple path (legacy, default) uses
  ``a_dim`` / ``b_dim`` semi-axes from ``DieselParams``;
* the groove path uses the angular branch width
  ``w_branch_angle_rad`` from
  :func:`config.diesel_groove_presets.resolve_groove_preset`;
* the recommended ``N_phi`` for the diesel ``g4_same_depth_safe``
  preset on the production R=100 mm bearing is close to 170, NOT
  the legacy 449 (which came from the dimple-pocket formula
  applied to the groove geometry by mistake).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import diesel_params as params
from config.diesel_groove_presets import resolve_groove_preset
from models.diesel_transient import texture_resolution_diagnostic


# ─── 1. Groove path — explicit angular width ───────────────────────


def test_groove_diagnostic_uses_angular_width():
    """The groove path computes
    ``cells_per_groove_width_phi = N_phi * w_rad / (2π)`` and
    recommends the smallest N_phi giving 4 cells across one branch."""
    w_rad = 14.8e-3 / 0.100  # ≈ 0.148 rad — g4_same_depth_safe on R=100 mm
    diag = texture_resolution_diagnostic(
        N_phi=160, N_z=60,
        R=0.100, L=0.140,
        texture_kind="groove",
        w_branch_angle_rad=w_rad,
    )
    expected_cells = 160.0 * w_rad / (2.0 * np.pi)
    assert diag["texture_kind"] == "groove"
    assert diag["cells_per_groove_width_phi"] == pytest.approx(
        expected_cells, rel=1e-12)
    # 160 × 0.148 / 2π ≈ 3.77 cells → insufficient.
    assert diag["resolution_status"] == "insufficient"
    # Recommended N_phi gives ≥4 cells across one branch:
    # ⌈4 · 2π / 0.148⌉ = ⌈169.8⌉ = 170.
    assert diag["recommended_n_phi_min"] == 170
    # Sanity: applying the recommendation produces a valid
    # 4-cells-across-one-branch resolution.
    diag_at_min = texture_resolution_diagnostic(
        N_phi=170, N_z=60,
        R=0.100, L=0.140,
        texture_kind="groove",
        w_branch_angle_rad=w_rad,
    )
    assert diag_at_min["cells_per_groove_width_phi"] >= 4.0
    assert diag_at_min["resolution_status"] in ("marginal", "ok")


def test_groove_diagnostic_marginal_band():
    """The 'marginal' band is [4, 6) cells across the branch."""
    w_rad = 14.8e-3 / 0.100
    # 200 cells × 0.148 / 2π ≈ 4.71 — marginal.
    diag = texture_resolution_diagnostic(
        N_phi=200, N_z=60,
        R=0.100, L=0.140,
        texture_kind="groove",
        w_branch_angle_rad=w_rad,
    )
    assert diag["resolution_status"] == "marginal"
    # 320 cells × 0.148 / 2π ≈ 7.55 — ok.
    diag_ok = texture_resolution_diagnostic(
        N_phi=320, N_z=60,
        R=0.100, L=0.140,
        texture_kind="groove",
        w_branch_angle_rad=w_rad,
    )
    assert diag_ok["resolution_status"] == "ok"


def test_groove_diagnostic_rejects_dimple_inputs():
    """The groove path errors out without ``w_branch_angle_rad``;
    a dimple-style ``a_dim`` / ``b_dim`` does not silently substitute."""
    with pytest.raises(ValueError):
        texture_resolution_diagnostic(
            N_phi=160, N_z=60,
            R=0.100, L=0.140,
            texture_kind="groove",
            a_dim=2e-4, b_dim=1.5e-3,
        )
    with pytest.raises(ValueError):
        texture_resolution_diagnostic(
            N_phi=160, N_z=60,
            R=0.100, L=0.140,
            texture_kind="groove",
            w_branch_angle_rad=0.0,
        )


# ─── 2. Dimple path — legacy contract preserved ────────────────────


def test_dimple_diagnostic_legacy_contract():
    """The dimple path preserves the pre-fixup behaviour bit-for-bit
    (now with explicit ``texture_kind="dimple"``)."""
    diag = texture_resolution_diagnostic(
        N_phi=80, N_z=30,
        R=params.R, L=params.L,
        texture_kind="dimple",
        a_dim=params.a_dim, b_dim=params.b_dim,
    )
    expected_cpp_phi = 80.0 * params.b_dim / (np.pi * params.R)
    expected_cpp_z = 2.0 * 30.0 * params.a_dim / params.L
    assert diag["texture_kind"] == "dimple"
    assert diag["cells_per_pocket_phi"] == pytest.approx(
        expected_cpp_phi, rel=1e-12)
    assert diag["cells_per_pocket_z"] == pytest.approx(
        expected_cpp_z, rel=1e-12)
    expected_rec = int(np.ceil(4.0 * np.pi * params.R / params.b_dim))
    assert diag["recommended_n_phi_min"] == expected_rec


def test_dimple_diagnostic_default_kind():
    """``texture_kind`` defaults to dimple — the legacy call site
    that omits the argument continues to work."""
    diag = texture_resolution_diagnostic(
        N_phi=80, N_z=30,
        R=params.R, L=params.L,
        a_dim=params.a_dim, b_dim=params.b_dim,
    )
    assert diag["texture_kind"] == "dimple"
    assert "cells_per_pocket_phi" in diag


# ─── 3. End-to-end with the resolved preset ────────────────────────


def test_diesel_g4_same_depth_safe_recommends_170_not_449():
    """Regression guard against the legacy bug: the diesel
    ``g4_same_depth_safe`` preset on the R=100 mm bearing must
    recommend ``N_phi`` close to 170 (groove branch ≈ 0.148 rad
    needs 4 cells), NOT the legacy ≈ 449 that came from running
    the dimple-pocket formula against a bogus ``w_branch_nondim``
    that was π× too large."""
    pr = resolve_groove_preset(
        "g4_same_depth_safe",
        R_m=0.100, L_m=params.L, c_m=120e-6)
    diag = texture_resolution_diagnostic(
        N_phi=160, N_z=60,
        R=0.100, L=params.L,
        texture_kind="groove",
        w_branch_angle_rad=float(pr["w_branch_angle_rad"]),
    )
    # Recommended N_phi for 4 cells across one branch = ⌈4·2π / 0.148⌉.
    expected = int(math.ceil(4.0 * 2.0 * math.pi
                              / float(pr["w_branch_angle_rad"])))
    assert diag["recommended_n_phi_min"] == expected
    # The headline number is ~170, far from the pre-fixup 449.
    assert 165 <= diag["recommended_n_phi_min"] <= 175
    assert diag["recommended_n_phi_min"] != 449
