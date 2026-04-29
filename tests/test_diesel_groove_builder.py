"""Stage J — diesel groove builder contract tests.

Five contract tests on the Stage J groove relief façade
(:func:`models.groove_geometry.build_herringbone_groove_relief`)
plus the preset registry. None of these tests touches the GPU
solver — they are pure geometry checks on the additive Δh/c relief
returned by the builder.
"""
from __future__ import annotations

import numpy as np
import pytest

from config.diesel_groove_presets import (
    GROOVE_PRESETS,
    resolve_groove_preset,
)
from models.bearing_model import setup_grid
from models.groove_geometry import (
    build_herringbone_groove_relief,
    relief_stats,
)
from models.diesel_transient import build_H_2d
from config import diesel_params as params


def _diesel_grid(N_phi: int = 80, N_z: int = 40):
    phi_1D, Z_1D, Phi, Z, _, _ = setup_grid(N_phi, N_z)
    return Phi, Z


# ─── 1. Builder invariants ─────────────────────────────────────────


def test_groove_relief_nonnegative_and_bounded():
    """Stage J contract: relief is non-negative, never exceeds the
    declared depth, never has NaN/inf, and the apex does not double
    in depth where two arms meet (the v4 builder uses np.maximum to
    fuse arms — the test guards that union semantics)."""
    Phi, Z = _diesel_grid()
    pr = resolve_groove_preset(
        "g4_scaled_ratio",
        R_m=params.R, L_m=params.L, c_m=params.c)
    relief = build_herringbone_groove_relief(
        Phi, Z,
        depth_nondim=pr["depth_nondim"],
        N_branch_per_side=pr["N_branch_per_side"],
        w_branch_nondim=pr["w_branch_nondim"],
        belt_half_nondim=pr["belt_half_nondim"],
        beta_deg=pr["beta_deg"],
        ramp_frac=pr["ramp_frac"],
        taper_ratio=pr["taper_ratio"],
        apex_radius_frac=pr["apex_radius_frac"],
        chirality=pr["chirality"],
        coverage_mode=pr["coverage_mode"],
        protected_lo_deg=pr["protected_lo_deg"],
        protected_hi_deg=pr["protected_hi_deg"],
    )
    stats = relief_stats(relief)
    assert stats["has_nan"] is False
    assert stats["has_inf"] is False
    assert stats["relief_min"] >= 0.0
    # The builder uses np.maximum to fuse arms — the resulting depth
    # cannot exceed the declared depth_nondim by more than rounding.
    assert stats["relief_max"] <= pr["depth_nondim"] + 1e-12
    # Sanity: the relief must actually be present somewhere (the
    # default G4 preset does not protect the entire shell).
    assert stats["relief_nonzero_frac"] > 0.0


# ─── 2. Periodic wrap on phi seam ──────────────────────────────────


def test_groove_periodic_wrap():
    """Reliefs near φ=0 / φ=2π must agree across the seam — the
    diesel grid is endpoint-free so a groove crossing the seam is
    represented as a column at index 0 and one at index N_phi - 1
    that wrap to each other. The builder's _wrap_phi math is the
    contract here."""
    Phi, Z = _diesel_grid(N_phi=120, N_z=40)
    pr = resolve_groove_preset(
        "g4_scaled_ratio",
        R_m=params.R, L_m=params.L, c_m=params.c)
    relief = build_herringbone_groove_relief(
        Phi, Z,
        depth_nondim=pr["depth_nondim"],
        N_branch_per_side=pr["N_branch_per_side"],
        w_branch_nondim=pr["w_branch_nondim"],
        belt_half_nondim=pr["belt_half_nondim"],
        beta_deg=pr["beta_deg"],
        ramp_frac=pr["ramp_frac"],
        taper_ratio=pr["taper_ratio"],
        apex_radius_frac=pr["apex_radius_frac"],
        chirality=pr["chirality"],
        # full_360 to make the seam test deterministic — protected
        # sectors can otherwise mask one side of the seam by accident.
        coverage_mode="full_360",
    )
    # Build the relief shifted by one full period on phi (Phi + 2π
    # on the same Z) — the result must be identical because the
    # builder's _wrap_phi maps both versions onto the same canonical
    # representation.
    relief_shifted = build_herringbone_groove_relief(
        Phi + 2.0 * np.pi, Z,
        depth_nondim=pr["depth_nondim"],
        N_branch_per_side=pr["N_branch_per_side"],
        w_branch_nondim=pr["w_branch_nondim"],
        belt_half_nondim=pr["belt_half_nondim"],
        beta_deg=pr["beta_deg"],
        ramp_frac=pr["ramp_frac"],
        taper_ratio=pr["taper_ratio"],
        apex_radius_frac=pr["apex_radius_frac"],
        chirality=pr["chirality"],
        coverage_mode="full_360",
    )
    assert np.allclose(relief, relief_shifted, atol=1e-12)


# ─── 3. G4 scaled-ratio preset conversion ──────────────────────────


def test_g4_scaled_ratio_conversion():
    """The ``g4_scaled_ratio`` preset must preserve the validated G4
    ratios — d_g/c = 15/40 = 0.375, w_g/D = 4/54 ≈ 0.0741, plus
    N=10, β=45°, taper=0.6, chirality=pump_to_edge."""
    pr = resolve_groove_preset(
        "g4_scaled_ratio",
        R_m=params.R, L_m=params.L, c_m=params.c)
    # d_g/c at c=120 µm with d_g=45 µm.
    assert pr["d_g_over_c"] == pytest.approx(45.0 / 120.0, rel=1e-12)
    assert pr["d_g_over_c"] == pytest.approx(0.375, rel=1e-12)
    # w_g/D: original ratio 4/54 ≈ 0.07407; the rounded preset
    # value 14.8 mm on D=200 mm gives 14.8/200 = 0.074. The 0.1%
    # rounding matches the tolerance documented in the preset file.
    assert pr["w_g_over_D"] == pytest.approx(14.8 / 200.0, rel=1e-12)
    assert pr["w_g_over_D"] == pytest.approx(4.0 / 54.0, rel=2e-3)
    # G4 design family invariants.
    assert pr["N_branch_per_side"] == 10
    assert pr["beta_deg"] == pytest.approx(45.0)
    assert pr["taper_ratio"] == pytest.approx(0.6)
    assert pr["chirality"] == "pump_to_edge"


# ─── 3a. Angular branch-width contract (Stage J fu-2 fixup-1) ──────


def test_groove_angular_branch_width_in_radians():
    """The preset must hand the builder an *angular* branch width
    (radians on the Phi grid), not a width-over-diameter ratio.

    The builder ``build_half_herringbone_ramped_relief`` compares
    ``|Phi - phi_c| <= 0.5 * w_local`` where ``Phi`` is in radians
    and ``w_local`` is derived from ``w_branch_nondim``. The
    physical mapping arc_length = angle × R requires
    ``w_branch_nondim = w_g / R``.

    This test pins both the conversion and the new diagnostic field
    ``w_branch_angle_rad`` so a regression to the old
    ``(w_g / D) * 2π`` formula (which produced widths π× too large)
    cannot pass silently.
    """
    pr = resolve_groove_preset(
        "g4_same_depth_safe",
        R_m=params.R, L_m=params.L, c_m=params.c)
    # Diagnostic-only ratio still expressed against D.
    assert pr["w_g_over_D"] == pytest.approx(
        14.8e-3 / (2.0 * params.R), rel=1e-12)
    # The actual builder argument is an angle in radians = w_g / R.
    assert pr["w_branch_nondim"] == pytest.approx(
        14.8e-3 / params.R, rel=1e-12)
    # Explicit echo field — same value, different name, exists so
    # callers reading the preset don't have to re-derive the unit.
    assert pr["w_branch_angle_rad"] == pytest.approx(
        14.8e-3 / params.R, rel=1e-12)
    assert pr["w_branch_angle_rad"] == pr["w_branch_nondim"]
    # The bug we're guarding against: the old formula
    # (w_g/D) * 2π = π × (w_g/R) inflated the angle by π.
    assert pr["w_branch_nondim"] != pytest.approx(
        (14.8e-3 / (2.0 * params.R)) * 2.0 * np.pi, rel=1e-9)


def test_diesel_g4_same_depth_safe_geometry_echo():
    """The summary-writer-facing diagnostic fields for
    ``g4_same_depth_safe`` on the diesel R=100 mm bearing match
    the physical preset declaration:
        w_g = 14.8 mm  →  w_g/D ≈ 0.074, w_branch_angle_rad ≈ 0.148.
    """
    pr = resolve_groove_preset(
        "g4_same_depth_safe",
        R_m=0.100, L_m=params.L, c_m=120e-6)
    assert pr["w_g_mm"] == pytest.approx(14.8, rel=1e-12)
    assert pr["w_g_over_D"] == pytest.approx(14.8 / 200.0, rel=1e-12)
    assert pr["w_branch_angle_rad"] == pytest.approx(
        14.8e-3 / 0.100, rel=1e-12)
    # Sanity on absolute magnitude — 0.148 rad ≈ 8.5°, well below
    # the cell span of 2π/10 ≈ 0.628 rad ≈ 36°, so 10 branches
    # don't overlap.
    assert pr["w_branch_angle_rad"] < (2.0 * np.pi
                                        / pr["N_branch_per_side"])


# ─── 4. Geometry safety at the eccentricity cap ────────────────────


def test_groove_H_safe_at_eps_cap():
    """At the ε = 0.95 mechanical clamp, the textured film must
    remain strictly positive everywhere (ε_max keeps min(H_base) at
    0.05; the groove relief only adds, never removes, so
    H_total = H_base + relief stays ≥ 0.05 — well above zero)."""
    Phi, Z = _diesel_grid(N_phi=120, N_z=40)
    pr = resolve_groove_preset(
        "g4_scaled_ratio",
        R_m=params.R, L_m=params.L, c_m=params.c)
    relief = build_herringbone_groove_relief(
        Phi, Z,
        depth_nondim=pr["depth_nondim"],
        N_branch_per_side=pr["N_branch_per_side"],
        w_branch_nondim=pr["w_branch_nondim"],
        belt_half_nondim=pr["belt_half_nondim"],
        beta_deg=pr["beta_deg"],
        ramp_frac=pr["ramp_frac"],
        taper_ratio=pr["taper_ratio"],
        apex_radius_frac=pr["apex_radius_frac"],
        chirality=pr["chirality"],
        coverage_mode=pr["coverage_mode"],
        protected_lo_deg=pr["protected_lo_deg"],
        protected_hi_deg=pr["protected_hi_deg"],
    )
    eps = 0.95   # mechanical clamp
    eps_x = eps
    eps_y = 0.0
    H = build_H_2d(eps_x, eps_y, Phi, Z, params,
                   textured=True,
                   texture_kind="groove",
                   groove_relief=relief)
    assert np.all(np.isfinite(H))
    # Without the groove (smooth), min(H) ≈ 1 - ε = 0.05 (modulo
    # the σ/c regularisation). With the additive groove relief the
    # minimum can only rise.
    H_smooth = build_H_2d(eps_x, eps_y, Phi, Z, params,
                            textured=False)
    assert float(np.min(H)) >= float(np.min(H_smooth)) - 1e-12
    assert float(np.min(H)) > 0.0


# ─── 5. Default dimple path is bit-for-bit unchanged ───────────────


def test_default_dimple_path_unchanged():
    """build_H_2d(texture_kind='dimple', ...) must produce exactly
    the legacy ellipsoidal-dimple film — no Stage J refactor must
    perturb it. We compare ``texture_kind='dimple'`` (the default)
    against the historical default behaviour by passing the same
    eps and the same setup_texture-derived dimple centres."""
    from models.bearing_model import setup_texture
    Phi, Z = _diesel_grid(N_phi=80, N_z=30)
    phi_c, Z_c = setup_texture(params)
    eps_x = 0.30
    eps_y = -0.10
    # Legacy default — no texture_kind override.
    H_default = build_H_2d(eps_x, eps_y, Phi, Z, params,
                              textured=True,
                              phi_c_flat=phi_c, Z_c_flat=Z_c)
    # Explicit dimple path.
    H_explicit = build_H_2d(eps_x, eps_y, Phi, Z, params,
                               textured=True,
                               phi_c_flat=phi_c, Z_c_flat=Z_c,
                               texture_kind="dimple")
    assert np.array_equal(H_default, H_explicit)
    # And smooth (textured=False) is identical to texture_kind='none'.
    H_smooth = build_H_2d(eps_x, eps_y, Phi, Z, params,
                             textured=False)
    H_none = build_H_2d(eps_x, eps_y, Phi, Z, params,
                           textured=True, texture_kind="none")
    assert np.array_equal(H_smooth, H_none)


def test_groove_preset_registry_has_three_documented():
    """Sanity: the registry exposes at least the three TZ presets."""
    assert "g4_scaled_ratio" in GROOVE_PRESETS
    assert "g4_same_depth_safe" in GROOVE_PRESETS
    assert "diesel_divergent_180_360" in GROOVE_PRESETS
