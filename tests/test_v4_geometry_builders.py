"""Geometry tests for v4 rescue gate builders."""
import math, os, sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cases.gu_loaded_side_v4.geometry_builders import (
    build_relief, build_straight_ramped_relief, build_arc_ramped_relief,
    build_half_herringbone_ramped_relief, apply_partial_coverage,
    get_branch_centers, get_removed_branches,
)
from cases.gu_loaded_side_v4.schema import PROTECTED_LO_DEG, PROTECTED_HI_DEG

def _grid(Np=400, Nz=100):
    phi = np.linspace(0, 2*math.pi, Np, endpoint=False)
    Z = np.linspace(-1, 1, Nz)
    return np.meshgrid(phi, Z)

C = dict(depth_nondim=0.25, w_branch_nondim=0.15, belt_half_nondim=0.15,
         ramp_frac=0.15, taper_ratio=1.0)

def test_T1_belt_preservation():
    Phi, Z = _grid()
    r = build_straight_ramped_relief(Phi, Z, N_branch_per_side=8, **C)
    assert float(np.max(np.abs(r[np.abs(Z) <= C["belt_half_nondim"]]))) < 1e-12

def test_T2_max_depth_arc():
    Phi, Z = _grid()
    r = build_arc_ramped_relief(Phi, Z, N_branch_per_side=6,
                                 curvature_k=0.3, **C)
    assert float(np.max(r)) <= C["depth_nondim"] + 1e-6
    assert float(np.max(r)) > 0

def test_T3_no_doubling_herr():
    Phi, Z = _grid()
    r = build_half_herringbone_ramped_relief(
        Phi, Z, N_branch_per_side=6, beta_deg=30, apex_radius_frac=0.5, **C)
    assert float(np.max(r)) <= C["depth_nondim"] + 1e-6

def test_T4_ramp_smoothness():
    Phi, Z = _grid(Np=100, Nz=400)
    r = build_straight_ramped_relief(Phi, Z, N_branch_per_side=5, **C)
    col = r.shape[1] // (2 * 5)
    profile = r[:, col]
    jumps = np.abs(np.diff(profile))
    dZ = 2.0 / (Z.shape[0] - 1)
    assert float(np.max(jumps)) / dZ < C["depth_nondim"] * 20

def test_T5_arc_k0_equals_straight():
    Phi, Z = _grid()
    r_str = build_straight_ramped_relief(Phi, Z, N_branch_per_side=5, **C)
    r_arc = build_arc_ramped_relief(Phi, Z, N_branch_per_side=5,
                                     curvature_k=0.0, **C)
    assert np.allclose(r_str, r_arc, atol=1e-10)

def test_T6_partial_removes_branches():
    removed = get_removed_branches(
        [float(math.degrees(c)) for c in get_branch_centers(8)],
        PROTECTED_LO_DEG, PROTECTED_HI_DEG)
    assert len(removed) >= 1

def test_T7_partial_less_area():
    Phi, Z = _grid()
    r_full = build_straight_ramped_relief(Phi, Z, N_branch_per_side=8, **C)
    r_part = apply_partial_coverage(r_full, Phi, "protect_loaded_union",
                                     PROTECTED_LO_DEG, PROTECTED_HI_DEG)
    assert float(np.sum(r_part > 0)) < float(np.sum(r_full > 0))

def test_T8_dg_zero_returns_zero():
    Phi, Z = _grid()
    r = build_relief(Phi, Z, "straight_ramped", depth_nondim=0.0,
                      N_branch_per_side=8, w_branch_nondim=0.15,
                      belt_half_nondim=0.15)
    assert np.array_equal(r, np.zeros_like(r))
