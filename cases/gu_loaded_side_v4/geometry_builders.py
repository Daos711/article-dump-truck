"""Stage J back-compat shim — historical path for v4 case scripts.

The ramped/tapered groove relief builders moved to
``models/groove_geometry.py`` so the diesel transient runner and
the v4 case scripts share a single source of truth (Stage J,
"groove builder duplication audit"). This module is now a thin
re-export so existing imports keep working unchanged:

    from cases.gu_loaded_side_v4.geometry_builders import build_relief

still binds the canonical implementation in
``models.groove_geometry``. New code should import from
``models.groove_geometry`` directly.
"""
from __future__ import annotations

from models.groove_geometry import (
    _raised_cosine_ramp,
    _taper_width,
    _wrap_phi,
    apply_partial_coverage,
    build_arc_ramped_relief,
    build_half_herringbone_ramped_relief,
    build_relief,
    build_straight_ramped_relief,
    get_branch_centers,
    get_removed_branches,
)

__all__ = [
    "_raised_cosine_ramp",
    "_taper_width",
    "_wrap_phi",
    "apply_partial_coverage",
    "build_arc_ramped_relief",
    "build_half_herringbone_ramped_relief",
    "build_relief",
    "build_straight_ramped_relief",
    "get_branch_centers",
    "get_removed_branches",
]
