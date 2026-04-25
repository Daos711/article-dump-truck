"""Contract tests for partial (subset) groove masks used in gu_loaded_side."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.texture_geometry import (
    get_herringbone_cell_centers,
    create_H_with_herringbone_grooves_subset,
)

# ── helpers ──────────────────────────────────────────────────────────

N_G = 10  # total cells around circumference


def _active_cells(N_active: int, center_cell: int, shift: int = 0) -> list:
    """Return wrapped active-cell indices: a contiguous window of N_active
    cells centred on *center_cell*, then shifted by *shift* cells."""
    half = N_active // 2
    start = center_cell - half + shift
    return [(start + i) % N_G for i in range(N_active)]


def _grid(N_phi=400, N_Z=100):
    phi = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return Phi, Zm


# ── T1: contiguous window, no wrap ──────────────────────────────────

def test_subset_cells_contiguous():
    """N_active=5, center=3, shift=0 -> [1,2,3,4,5]."""
    cells = _active_cells(5, center_cell=3, shift=0)
    assert cells == [1, 2, 3, 4, 5], f"got {cells}"


# ── T2: wrap-around window ──────────────────────────────────────────

def test_subset_wraps_around():
    """N_active=5, center=9 (of 10 total) -> [7,8,9,0,1]."""
    cells = _active_cells(5, center_cell=9, shift=0)
    assert cells == [7, 8, 9, 0, 1], f"got {cells}"


# ── T3: shift moves window ─────────────────────────────────────────

def test_shift_changes_window():
    """shift=+1 moves the window one cell clockwise."""
    base = _active_cells(5, center_cell=3, shift=0)
    shifted = _active_cells(5, center_cell=3, shift=1)
    assert shifted != base, "shift should change the window"
    # Each cell in shifted should be (base cell + 1) mod N_G
    for b, s in zip(base, shifted):
        assert s == (b + 1) % N_G, f"expected {(b+1)%N_G}, got {s}"


# ── T4: subset builder preserves depth (no doubling) ───────────────

def test_subset_builder_preserves_depth():
    """create_H_with_herringbone_grooves_subset with active_cells=[0,1,2]
    must have max relief == depth_nondim, not doubled."""
    Phi, Zm = _grid(800, 200)
    H0 = np.ones_like(Phi)
    depth = 0.5
    H = create_H_with_herringbone_grooves_subset(
        H0, depth, Phi, Zm,
        N_g=N_G, w_g_nondim=0.15, L_g_nondim=1.0,
        beta_deg=30.0, active_cells=[0, 1, 2],
    )
    max_relief = float(np.max(H - H0))
    assert max_relief <= depth + 1e-12, (
        f"max relief {max_relief:.6f} > depth {depth} -- overlap doubles!")
    assert max_relief > 0.0, "no groove detected at all"


# ── T5: active cell centers match expected positions ────────────────

def test_subset_active_centers_on_cell_grid():
    """get_herringbone_cell_centers at subset indices should match the
    expected angular positions phi_k = (2pi/N_g)*(k+0.5)."""
    centers = get_herringbone_cell_centers(N_G)
    assert len(centers) == N_G
    active = [0, 4, 9]
    cell_span = 2.0 * math.pi / N_G
    for k in active:
        expected = cell_span * (k + 0.5)
        assert abs(centers[k] - expected) < 1e-12, (
            f"cell {k}: center={centers[k]:.6f}, expected={expected:.6f}")
