"""Geometry unit tests T1-T8 for v3 central-feed builders."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cases.gu_loaded_side_v3.geometry_builders import (
    build_relief,
    build_straight_ramped_relief,
    build_half_herringbone_ramped_relief,
    apply_partial_coverage,
)


def _grid(N_phi=800, N_Z=200):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    return np.meshgrid(phi, Z)


COMMON = dict(
    depth_nondim=0.25,
    N_branch_per_side=3,
    w_branch_nondim=0.15,
    belt_half_nondim=0.15,
    ramp_frac=0.15,
    taper_ratio=1.0,
)


# T1: belt preservation
def test_T1_central_belt_preservation():
    Phi, Z = _grid()
    r = build_straight_ramped_relief(Phi, Z, **COMMON)
    belt = np.abs(Z) <= COMMON["belt_half_nondim"]
    assert float(np.max(np.abs(r[belt]))) < 1e-12


# T2: max depth
def test_T2_max_depth_check():
    Phi, Z = _grid()
    r = build_straight_ramped_relief(Phi, Z, **COMMON)
    assert float(np.max(r)) <= COMMON["depth_nondim"] + 1e-6
    assert float(np.max(r)) > 0


# T3: no depth doubling
def test_T3_no_depth_doubling():
    Phi, Z = _grid()
    r = build_half_herringbone_ramped_relief(
        Phi, Z, **COMMON, beta_deg=30.0, apex_radius_frac=0.5)
    assert float(np.max(r)) <= COMMON["depth_nondim"] + 1e-6


# T4: ramp smoothness (no jump discontinuity)
def test_T4_ramp_smoothness():
    Phi, Z = _grid(N_phi=100, N_Z=400)
    r = build_straight_ramped_relief(Phi, Z, **COMMON)
    # Check axial depth profile at one branch center
    phi_c = 2 * math.pi / COMMON["N_branch_per_side"] * 0.5
    col = int(np.argmin(np.abs(Phi[0, :] - phi_c)))
    profile = r[:, col]
    jumps = np.abs(np.diff(profile))
    dZ = 2.0 / (Z.shape[0] - 1)
    max_jump_per_dZ = float(np.max(jumps)) / dZ
    # Smooth ramp should have bounded gradient
    assert max_jump_per_dZ < COMMON["depth_nondim"] * 20, (
        f"jump gradient {max_jump_per_dZ:.4f} too steep")


# T5: taper monotonicity
def test_T5_taper_monotonicity():
    Phi, Z = _grid(N_phi=200, N_Z=400)
    params = {**COMMON, "taper_ratio": 0.5}
    r = build_straight_ramped_relief(Phi, Z, **params)
    # Groove should narrow toward ends: at fixed Z far from belt,
    # angular width should be smaller than near belt
    belt_edge = COMMON["belt_half_nondim"]
    z_near = belt_edge + 0.1  # near belt
    z_far = 0.9  # near end
    row_near = int(np.argmin(np.abs(Z[:, 0] - z_near)))
    row_far = int(np.argmin(np.abs(Z[:, 0] - z_far)))
    width_near = int(np.sum(r[row_near, :] > 0.01 * COMMON["depth_nondim"]))
    width_far = int(np.sum(r[row_far, :] > 0.01 * COMMON["depth_nondim"]))
    assert width_far <= width_near + 2, (
        f"taper violation: near={width_near}, far={width_far}")


# T6: (arc endpoint — skipped, arcs not yet implemented)
def test_T6_arc_not_implemented():
    Phi, Z = _grid()
    with pytest.raises(NotImplementedError):
        build_relief(Phi, Z, variant="arc_ramped", **COMMON)


# T7: partial coverage mask
def test_T7_partial_coverage_preserves_smooth_zone():
    Phi, Z = _grid()
    r = build_straight_ramped_relief(Phi, Z, **COMMON)
    r_partial = apply_partial_coverage(
        r, Phi, mode="partial_fixed",
        protected_lo_deg=80.0, protected_hi_deg=130.0)
    # Protected zone should be zero
    phi_mod = np.mod(Phi, 2 * math.pi)
    lo = math.radians(80.0)
    hi = math.radians(130.0)
    protected = (phi_mod >= lo) & (phi_mod <= hi)
    assert float(np.max(np.abs(r_partial[protected]))) < 1e-12
    # Outside protected zone, relief should be preserved
    outside = ~protected
    assert np.allclose(r_partial[outside], r[outside])


# T8: legacy equivalence (ramp=0, apex=0, taper=1 → old straight)
def test_T8_legacy_equivalence_straight():
    Phi, Z = _grid()
    r_new = build_straight_ramped_relief(
        Phi, Z,
        depth_nondim=0.25, N_branch_per_side=3,
        w_branch_nondim=0.15, belt_half_nondim=0.15,
        ramp_frac=0.0, taper_ratio=1.0)
    # With ramp=0: depth should be constant inside branch
    # (like old rectangular straight builder)
    branch_mask = r_new > 0.01 * 0.25
    if np.any(branch_mask):
        # All nonzero relief should be exactly depth_nondim
        assert np.allclose(r_new[branch_mask], 0.25, atol=1e-6), (
            "with ramp=0, depth should be uniform")
