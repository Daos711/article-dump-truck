"""Contract tests for feed-consistent geometry builder."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.feed_geometry import (
    create_H_with_central_feed_branches,
    feed_window_metadata,
)


def _grid(N_phi=400, N_Z=100):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    return np.meshgrid(phi, Z)


def test_dg_zero_reproduces_smooth():
    Phi, Zm = _grid()
    H0 = 1.0 + 0.5 * np.cos(Phi)
    H = create_H_with_central_feed_branches(
        H0, 0.0, Phi, Zm, 3, 0.15, 0.15, 0.0, "straight")
    assert np.array_equal(H, H0)


def test_left_right_branch_symmetry():
    Phi, Zm = _grid()
    H0 = np.ones_like(Phi)
    H = create_H_with_central_feed_branches(
        H0, 0.5, Phi, Zm, 3, 0.15, 0.15, 0.0, "straight")
    relief = H - H0
    relief_flip = relief[::-1, :]
    corr = np.corrcoef(relief.ravel(), relief_flip.ravel())[0, 1]
    assert corr > 0.99, f"Z-symmetry correlation = {corr:.4f}"


def test_no_double_depth():
    Phi, Zm = _grid(800, 200)
    H0 = np.ones_like(Phi)
    depth = 0.5
    H = create_H_with_central_feed_branches(
        H0, depth, Phi, Zm, 5, 0.2, 0.10, 20.0, "half_herringbone")
    max_relief = float(np.max(H - H0))
    assert max_relief <= depth + 1e-10, (
        f"max relief {max_relief} > depth {depth}")


def test_belt_zone_groove_free():
    Phi, Zm = _grid()
    H0 = np.ones_like(Phi)
    belt = 0.20
    H = create_H_with_central_feed_branches(
        H0, 0.3, Phi, Zm, 3, 0.15, belt, 0.0, "straight")
    belt_mask = np.abs(Zm) <= belt
    belt_relief = (H - H0)[belt_mask]
    assert float(np.max(np.abs(belt_relief))) < 1e-12


def test_feed_window_outside_pressure_peak():
    meta = feed_window_metadata(phi_feed_deg=300.0)
    assert meta["phi_feed_deg"] == 300.0
    assert 290.0 < meta["phi_feed_lo_deg"] < 310.0


# ─── Stage H feed mask tests ──────────────────────────────────────

from models.feed_geometry import (
    build_feed_mask, build_feed_mask_variant, p_supply_to_g_bc,
)


def test_belt_wide_nonempty():
    Phi, Zm = _grid()
    mask, meta = build_feed_mask_variant(Phi, Zm, "belt_wide",
                                          z_belt_half=0.15)
    assert np.sum(mask) > 0
    assert meta["variant"] == "belt_wide"


def test_belt_wide_inside_belt():
    Phi, Zm = _grid()
    mask, _ = build_feed_mask_variant(Phi, Zm, "belt_wide",
                                       z_belt_half=0.15)
    assert np.all(np.abs(Zm[mask]) <= 0.15 + 1e-10)


def test_point_unloaded_inside_belt():
    Phi, Zm = _grid()
    mask, meta = build_feed_mask_variant(
        Phi, Zm, "point_unloaded",
        phi_loaded_deg=143.4, phi_feed_half_deg=5, z_belt_half=0.15)
    assert np.sum(mask) > 0
    assert np.all(np.abs(Zm[mask]) <= 0.15 + 1e-10)


def test_point_loaded_center():
    Phi, Zm = _grid()
    _, meta = build_feed_mask_variant(
        Phi, Zm, "point_loaded",
        phi_loaded_deg=143.4, phi_feed_half_deg=5, z_belt_half=0.15)
    assert abs(meta["phi_center_deg"] - 143.4) < 0.1


def test_wrap_around():
    Phi, Zm = _grid()
    mask, _ = build_feed_mask_variant(
        Phi, Zm, "point_unloaded",
        phi_loaded_deg=178.0, phi_feed_half_deg=5, z_belt_half=0.15)
    assert np.sum(mask) > 0


def test_empty_mask_raises():
    Phi, Zm = _grid(N_phi=10, N_Z=5)
    with pytest.raises(ValueError, match="empty"):
        build_feed_mask_variant(
            Phi, Zm, "point_loaded",
            phi_loaded_deg=143.4, phi_feed_half_deg=0.001,
            z_belt_half=0.001)


def test_p_supply_to_g_bc_value():
    g = p_supply_to_g_bc(2e5, 0.018, 2*math.pi*2000/60, 0.027, 40e-6)
    assert abs(g - 0.0194) < 0.001


def test_p_supply_zero():
    g = p_supply_to_g_bc(0.0, 0.018, 2*math.pi*2000/60, 0.027, 40e-6)
    assert g == 0.0
