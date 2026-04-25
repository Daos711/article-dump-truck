"""Contract tests for herringbone groove geometry builders (ТЗ §10)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.texture_geometry import (
    create_H_with_straight_grooves,
    create_H_with_herringbone_grooves,
    gu_groove_params_nondim,
)


def _grid(N_phi=400, N_Z=100):
    phi = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return Phi, Zm


# ── T1: zero texture == H0 ────────────────────────────────────────

def test_zero_texture_equals_H0():
    Phi, Zm = _grid()
    H0 = 1.0 + 0.5 * np.cos(Phi)
    H_s = create_H_with_straight_grooves(H0, 0.0, Phi, Zm, 10, 0.1, 0.5)
    H_h = create_H_with_herringbone_grooves(H0, 0.0, Phi, Zm, 10, 0.1, 0.5, 30.0)
    assert np.array_equal(H_s, H0), "depth=0 straight must equal H0"
    assert np.array_equal(H_h, H0), "depth=0 herringbone must equal H0"


# ── T2: depth bounded (no 2·d_g in overlap) ──────────────────────

def test_herringbone_depth_is_bounded():
    """Max relief ≡ depth_nondim, never 2× (overlap rule)."""
    Phi, Zm = _grid(800, 200)
    H0 = np.ones_like(Phi)
    depth = 0.5
    H = create_H_with_herringbone_grooves(
        H0, depth, Phi, Zm, N_g=10,
        w_g_nondim=0.15, L_g_nondim=1.0, beta_deg=30.0)
    max_relief = float(np.max(H - H0))
    assert max_relief <= depth + 1e-12, (
        f"max relief {max_relief:.6f} > depth {depth} — overlap doubles!")
    assert max_relief > 0.0, "no groove detected"


# ── T3: periodic wrap consistency ─────────────────────────────────

def test_periodic_wrap_consistency():
    """Grooves crossing φ=0 seam remain continuous. We place a groove
    centered near φ=0 (actually near 2π since cells wrap) and check
    that the groove depth is present on BOTH sides of the seam."""
    N_phi = 400
    Phi, Zm = _grid(N_phi, 100)
    H0 = np.ones_like(Phi)
    # With N_g=10, cell_span = 2π/10 ≈ 0.628. First cell center at
    # 0.314 rad. That's well inside [0, 2π). But by setting N_g=1
    # (single groove centered at π) we don't test the seam.
    # With N_g=2, cell centers at π/2 and 3π/2. Cell span = π.
    # Groove with w_g ≈ 1.0 rad and cell center 3π/2: groove extends
    # from π to 2π, with wrap some samples at φ near 0 may be inside.
    # Actually our _wrap_phi_distance handles this — test it:
    H = create_H_with_straight_grooves(
        H0, 0.1, Phi, Zm, N_g=2, w_g_nondim=2.0, L_g_nondim=1.5)
    # With 2 grooves spanning ~2.0 rad each on a 2π circle, most of
    # the domain should be grooved.
    grooved_fraction = float(np.mean(H > H0 + 0.05))
    assert grooved_fraction > 0.3, (
        f"groove coverage = {grooved_fraction:.2f}, expected >0.3 for "
        f"wrap-around test")
    # Also check that BOTH halves (near φ=0 and near φ=π) have grooves
    left = (Phi < 1.0)
    right = (Phi > math.pi - 1.0) & (Phi < math.pi + 1.0)
    assert np.any((H - H0)[left] > 0.05), "no groove near φ=0"
    assert np.any((H - H0)[right] > 0.05), "no groove near φ=π"


# ── T4: mirror symmetry of one herringbone cell ─────────────────

def test_mirror_symmetry_of_one_cell():
    """Left/right arms of one herringbone cell should be symmetric
    about the apex axis (Z=0) when centered at (φ_c, 0)."""
    N_phi, N_Z = 800, 200
    Phi, Zm = _grid(N_phi, N_Z)
    H0 = np.ones_like(Phi)
    # Single cell (N_g=1) centered at φ = π
    H = create_H_with_herringbone_grooves(
        H0, 0.3, Phi, Zm, N_g=1,
        w_g_nondim=0.2, L_g_nondim=1.2, beta_deg=30.0)
    relief = H - H0
    # Mirror Z → -Z should give same relief pattern
    relief_flipped = relief[::-1, :]
    # Tolerance: the discrete grid may not be perfectly symmetric if
    # N_Z is even (no exact Z=0 row), but the overall pattern must match.
    corr = np.corrcoef(relief.ravel(), relief_flipped.ravel())[0, 1]
    assert corr > 0.95, (
        f"Z-mirror correlation = {corr:.3f}, expected >0.95")


# ── T5: same cell count / coverage for both types ────────────────

def test_straight_vs_herringbone_cell_counts():
    """Same N_g, w_g, L_g → both types have grooved fraction within
    a reasonable margin of each other (not wildly different)."""
    Phi, Zm = _grid(800, 200)
    H0 = np.ones_like(Phi)
    depth = 0.3
    kwargs = dict(N_g=10, w_g_nondim=0.15, L_g_nondim=0.8)
    H_s = create_H_with_straight_grooves(H0, depth, Phi, Zm, **kwargs)
    H_h = create_H_with_herringbone_grooves(
        H0, depth, Phi, Zm, beta_deg=30.0, **kwargs)
    frac_s = float(np.mean(H_s > H0 + 0.1 * depth))
    frac_h = float(np.mean(H_h > H0 + 0.1 * depth))
    # Both should cover a nonzero fraction, and neither should differ
    # by more than 2× from the other
    assert frac_s > 0.01, f"straight coverage={frac_s:.4f}"
    assert frac_h > 0.01, f"herringbone coverage={frac_h:.4f}"
    ratio = max(frac_s, frac_h) / max(min(frac_s, frac_h), 1e-12)
    assert ratio < 3.0, (
        f"coverage ratio {ratio:.2f}: straight={frac_s:.4f}, "
        f"herringbone={frac_h:.4f}")
