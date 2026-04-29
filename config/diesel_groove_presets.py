"""Stage J — diesel groove preset registry.

Three named presets, each fully specified so the runner does not
need to know how the G4 anchor maps onto the diesel geometry:

* ``g4_scaled_ratio`` — preserves dg/c = 15/40 = 0.375 from the G4
  pump anchor; on the diesel bearing (c = 120 µm) this becomes
  dg = 45 µm. Default acceptance preset.
* ``g4_same_depth_safe`` — preserves the absolute G4 depth
  dg = 15 µm. A safer-but-shallower diagnostic.
* ``diesel_divergent_180_360`` — diagnostic placement on the
  180°-360° divergent zone (phi_window coverage) instead of the
  G4 protect-loaded-union mask.

All presets share the validated G4 design family: N=10 branches per
side, β=45°, taper=0.6, chirality=pump_to_edge, apex_radius_frac=0.5.
``resolve_groove_preset(name, params)`` returns a fully numeric
dict (depth_nondim, w_branch_nondim) ready for
``models.groove_geometry.build_herringbone_groove_relief``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


GROOVE_PRESETS: Dict[str, Dict[str, Any]] = {
    "g4_scaled_ratio": {
        "variant": "half_herringbone_ramped",
        "N_branch_per_side": 10,
        "beta_deg": 45.0,
        # 15 µm / 40 µm clearance ratio preserved — at diesel c = 120 µm
        # this gives dg = 45 µm.
        "d_g_um": 45.0,
        # 4 mm groove width on D = 54 mm pump → ratio preserved on
        # D = 200 mm gives w_g = 14.815 mm. We round to 14.8 mm —
        # the 0.1% slip is well below the integration grid.
        "w_g_mm": 14.8,
        "belt_half_nondim": 0.15,
        "taper_ratio": 0.6,
        "ramp_frac": 0.15,
        "apex_radius_frac": 0.5,
        "chirality": "pump_to_edge",
        "coverage_mode": "protect_loaded_union",
        "protected_lo_deg": 105.0,
        "protected_hi_deg": 175.0,
    },
    "g4_same_depth_safe": {
        "variant": "half_herringbone_ramped",
        "N_branch_per_side": 10,
        "beta_deg": 45.0,
        # Absolute source depth preserved (safer than dg/c-scaled).
        "d_g_um": 15.0,
        "w_g_mm": 14.8,
        "belt_half_nondim": 0.15,
        "taper_ratio": 0.6,
        "ramp_frac": 0.15,
        "apex_radius_frac": 0.5,
        "chirality": "pump_to_edge",
        "coverage_mode": "protect_loaded_union",
        "protected_lo_deg": 105.0,
        "protected_hi_deg": 175.0,
    },
    "diesel_divergent_180_360": {
        "variant": "half_herringbone_ramped",
        "N_branch_per_side": 10,
        "beta_deg": 45.0,
        "d_g_um": 45.0,
        "w_g_mm": 14.8,
        "belt_half_nondim": 0.15,
        "taper_ratio": 0.6,
        "ramp_frac": 0.15,
        "apex_radius_frac": 0.5,
        "chirality": "pump_to_edge",
        # Diagnostic: keep relief only inside the divergent
        # 180°-360° window.
        "coverage_mode": "phi_window",
        "protected_lo_deg": 180.0,
        "protected_hi_deg": 360.0,
    },
}


def resolve_groove_preset(
    name: str,
    *,
    R_m: float,
    L_m: float,
    c_m: float,
) -> Dict[str, Any]:
    """Resolve a named preset against the diesel bearing geometry.

    Returns a dict that ``build_herringbone_groove_relief`` accepts
    directly — depth_nondim = dg/c, w_branch_nondim = wg / R (the
    angular branch width in radians on the Phi grid).

    The diagnostic ratios ``d_g_over_c``, ``w_g_over_D``, and
    ``w_branch_angle_rad`` are included so the summary writer can
    echo them back. ``w_branch_angle_rad`` is numerically equal to
    ``w_branch_nondim`` and exists to make the unit (radians)
    obvious to the reader. The original metric inputs
    (``d_g_um``, ``w_g_mm``) are echoed verbatim so the user can
    check them at a glance.
    """
    if name not in GROOVE_PRESETS:
        raise KeyError(
            f"unknown groove preset {name!r}; valid names: "
            f"{sorted(GROOVE_PRESETS)}"
        )
    src = dict(GROOVE_PRESETS[name])
    d_g_m = float(src["d_g_um"]) * 1e-6
    w_g_m = float(src["w_g_mm"]) * 1e-3
    D_m = 2.0 * float(R_m)
    depth_nondim = d_g_m / float(c_m)
    # Stage J fu-2 — geometry-fixup-1 (expert review).
    # ``w_branch_nondim`` is the **circumferential angular width**
    # of one groove branch, in radians, on the Phi grid that the
    # builder operates on (φ ∈ [0, 2π)). The builder compares
    # ``|Phi - phi_c| <= 0.5 * w_local`` directly against ``Phi``,
    # so the unit must be radians.
    #
    # Physical mapping: arc_length = angle × R. So for a groove of
    # physical circumferential width ``w_g`` on a journal of radius
    # ``R``, the angular width in radians is ``w_g / R``.
    #
    # The previous formula ``(w_g / D) * 2π = π * w_g / R`` produced
    # a width π× too large (e.g. on the diesel R=100 mm bearing,
    # ``w_g=14.8 mm`` came out as 0.465 rad ≈ 26.6° instead of the
    # honest 0.148 rad ≈ 8.5°). With every branch swallowing ~3×
    # the protected sector, the textured smoke at F=0.3 aborted on
    # the firing peak — a geometry bug, not a physics result.
    w_branch_nondim = w_g_m / float(R_m)
    out = dict(
        # Pass-through builder kwargs.
        variant=str(src["variant"]),
        depth_nondim=depth_nondim,
        N_branch_per_side=int(src["N_branch_per_side"]),
        w_branch_nondim=w_branch_nondim,
        belt_half_nondim=float(src["belt_half_nondim"]),
        beta_deg=float(src["beta_deg"]),
        ramp_frac=float(src["ramp_frac"]),
        taper_ratio=float(src["taper_ratio"]),
        apex_radius_frac=float(src["apex_radius_frac"]),
        chirality=str(src["chirality"]),
        coverage_mode=str(src["coverage_mode"]),
        protected_lo_deg=float(src["protected_lo_deg"]),
        protected_hi_deg=float(src["protected_hi_deg"]),
        # Diagnostic / echo fields.
        d_g_um=float(src["d_g_um"]),
        w_g_mm=float(src["w_g_mm"]),
        d_g_over_c=depth_nondim,
        w_g_over_D=w_g_m / D_m,
        # Stage J fu-2 — geometry-fixup-1: explicit angular-width
        # echo so the diagnostic / summary writer can quote the
        # physical groove footprint without re-deriving from
        # w_branch_nondim. Numerically equal to w_branch_nondim,
        # named differently to make the unit (radians) obvious to
        # the reader.
        w_branch_angle_rad=w_branch_nondim,
        preset_name=name,
    )
    return out


def list_presets() -> Dict[str, Dict[str, Any]]:
    """Return a deep-copy snapshot of the preset registry."""
    return {k: dict(v) for k, v in GROOVE_PRESETS.items()}
