"""Contract tests for groove_sector_magnet (per-magnet vector sum model)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.groove_sector_magnet import (
    EmbeddedGrooveSectorMagnetModel,
    make_sector_magnet_model,
)
from models.groove_magnet_force import make_groove_magnet_model
from models.texture_geometry import get_herringbone_cell_centers

# ── shared fixtures ──────────────────────────────────────────────────

N_G = 10
C_M = 40e-6
D_SUB_M = 50e-6

# 3-cell loaded-side subset centred near phi=pi (bottom)
_LOADED_CENTERS = np.array([
    math.pi - 2 * math.pi / N_G,
    math.pi,
    math.pi + 2 * math.pi / N_G,
])


# ── T1: B=0 gives zero force ────────────────────────────────────────

def test_bref_zero_gives_zero():
    m = make_sector_magnet_model(
        _LOADED_CENTERS, c_m=C_M, d_sub_m=D_SUB_M, B_ref_T=0.0)
    Fx, Fy = m.force(0.1, -0.3)
    assert abs(Fx) < 1e-15, f"Fx={Fx}"
    assert abs(Fy) < 1e-15, f"Fy={Fy}"


# ── T2: |F| increases with B_ref ────────────────────────────────────

def test_bref_monotone():
    X, Y = 0.1, -0.3
    magnitudes = []
    for b in [0.3, 0.5, 0.8, 1.0]:
        m = make_sector_magnet_model(
            _LOADED_CENTERS, c_m=C_M, d_sub_m=D_SUB_M, B_ref_T=b)
        Fx, Fy = m.force(X, Y)
        magnitudes.append(math.sqrt(Fx**2 + Fy**2))
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] < magnitudes[i + 1], (
            f"|F| not monotone: {magnitudes}")


# ── T3: sector vector sum != isotropic radial surrogate ─────────────

def test_sector_differs_from_radial():
    """For an asymmetric 3-cell subset the per-magnet vector sum must
    differ from the isotropic radial surrogate (groove_magnet_force)."""
    m_sector = make_sector_magnet_model(
        _LOADED_CENTERS, c_m=C_M, d_sub_m=D_SUB_M, B_ref_T=0.5)
    m_radial = make_groove_magnet_model(
        N_g=N_G, c_m=C_M, d_g_m=D_SUB_M, B_ref_T=0.5)
    X, Y = 0.1, -0.3
    Fs_x, Fs_y = m_sector.force(X, Y)
    Fr_x, Fr_y = m_radial.force(X, Y)
    assert abs(Fr_x - Fs_x) > 1e-6 or abs(Fr_y - Fs_y) > 1e-6, (
        "sector and radial surrogate should differ for asymmetric subset")


# ── T4: conv_mag and groove_mag use identical phi_centers ────────────

def test_conv_mag_groove_mag_same_centers():
    """For the same pattern_id (active_cells list), both the sector
    magnet model and the radial magnet model derive centers from the
    same get_herringbone_cell_centers grid."""
    active_cells = [3, 4, 5]
    all_centers = get_herringbone_cell_centers(N_G)
    subset_centers = all_centers[active_cells]

    m_sector = make_sector_magnet_model(
        subset_centers, c_m=C_M, d_sub_m=D_SUB_M, B_ref_T=0.5)

    # The radial model factory uses compute_magnet_centers_from_groove_cells
    # which generates centers via the same formula as get_herringbone_cell_centers.
    full_centers_radial = make_groove_magnet_model(N_g=N_G, B_ref_T=0.5).phi_centers

    np.testing.assert_allclose(
        all_centers, full_centers_radial, atol=1e-14,
        err_msg="herringbone cell centers != magnet force centers")

    # Subset centres used by sector model match the expected slice
    np.testing.assert_allclose(
        m_sector.phi_centers, subset_centers, atol=1e-14,
        err_msg="sector model centres don't match expected subset")
