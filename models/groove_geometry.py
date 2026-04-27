"""Stage J — canonical home for ramped/tapered herringbone groove
relief builders.

Boundary with ``models/texture_geometry.py``
--------------------------------------------
``models/texture_geometry.py`` keeps its existing API:

* binary-mask groove builders
  (``create_H_with_straight_grooves``,
   ``create_H_with_herringbone_grooves``,
   ``create_H_with_herringbone_grooves_subset``);
* dimensional helpers
  (``gu_groove_params_nondim``, ``transfer_groove_params``,
   ``get_herringbone_cell_centers``).

These are used by 11 historical case scripts (``gu_2020``,
``gu_loaded_side``, ``pump_herringbone``) and 4 tests, none of which
need ramped/tapered relief. The binary-mask family is a different
texture model (full-H output, sharp edges) and remains the source
of truth for those experiments.

This module — ``models/groove_geometry.py`` — is the **single
source of truth** for the ramped/tapered groove family used by:

* the diesel transient runner (Stage J);
* ``cases/gu_loaded_side_v4/*`` (via a thin back-compat shim
  in ``cases/gu_loaded_side_v4/geometry_builders.py``).

The v4 case-script-side builders ``build_straight_ramped_relief``,
``build_half_herringbone_ramped_relief``, ``build_arc_ramped_relief``,
``apply_partial_coverage``, ``build_relief``, ``get_branch_centers``,
``get_removed_branches`` now live here and are re-exported by the
v4 shim so existing imports remain valid.

Coordinate conventions
----------------------
* ``Phi`` is the circumferential angle in radians, monotonically
  increasing on [0, 2π) (matches diesel ``setup_grid``).
* ``Z`` is the non-dimensional axial coordinate on [-1, +1].
* The returned ``relief`` is non-dimensional Δh/c (≥ 0); add to the
  base film H0 to obtain the textured film thickness.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


# ─── Low-level geometry primitives (moved from v4 case dir) ────────


def _raised_cosine_ramp(s: np.ndarray) -> np.ndarray:
    out = np.zeros_like(s, dtype=float)
    mask = (s > 0) & (s < 1)
    out[mask] = 0.5 * (1.0 - np.cos(np.pi * s[mask]))
    out[s >= 1] = 1.0
    return out


def _taper_width(s, w_in, taper_ratio):
    return w_in * (1.0 - (1.0 - taper_ratio) * np.clip(s, 0, 1))


def _wrap_phi(Phi, phi_c):
    d = Phi - phi_c
    return d - 2.0 * math.pi * np.round(d / (2.0 * math.pi))


# ─── Ramped relief builders (moved from v4 case dir) ───────────────


def build_straight_ramped_relief(
        Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
        belt_half_nondim, ramp_frac=0.15, taper_ratio=1.0):
    relief = np.zeros_like(Phi, dtype=float)
    if depth_nondim <= 0 or N_branch_per_side <= 0:
        return relief
    cell_span = 2.0 * math.pi / N_branch_per_side
    for k in range(N_branch_per_side):
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi(Phi, phi_c)
        for sign in [-1, 1]:
            z_start = sign * belt_half_nondim
            z_end = sign * 1.0
            z_lo, z_hi = min(z_start, z_end), max(z_start, z_end)
            L_branch = z_hi - z_lo
            if L_branch < 1e-12:
                continue
            s = np.clip(np.abs(Z - z_start) / L_branch, 0, 1)
            in_axial = (Z >= z_lo) & (Z <= z_hi)
            w_local = _taper_width(s, w_branch_nondim, taper_ratio)
            in_width = np.abs(dp) <= 0.5 * w_local
            ramp_len = ramp_frac * L_branch
            s_ramp = np.clip(np.abs(Z - z_start) / max(ramp_len, 1e-12),
                              0, 1)
            depth_profile = depth_nondim * _raised_cosine_ramp(s_ramp)
            mask = in_axial & in_width
            relief[mask] = np.maximum(relief[mask], depth_profile[mask])
    return relief


def build_half_herringbone_ramped_relief(
        Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
        belt_half_nondim, beta_deg=20.0, ramp_frac=0.15,
        taper_ratio=1.0, apex_radius_frac=0.5,
        chirality="pump_to_belt"):
    """Angled branches with chirality control.

    chirality:
      pump_to_belt — left arm angled to push flow toward Z=0 (standard)
      pump_to_edge — reversed angle, flow pushed toward ends
    """
    relief = np.zeros_like(Phi, dtype=float)
    if depth_nondim <= 0 or N_branch_per_side <= 0:
        return relief
    beta = math.radians(float(beta_deg))
    if chirality == "pump_to_edge":
        beta = -beta
    cell_span = 2.0 * math.pi / N_branch_per_side
    for k in range(N_branch_per_side):
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi(Phi, phi_c)
        for sign in [-1, 1]:
            z_start = sign * belt_half_nondim
            z_end = sign * 1.0
            L_branch = abs(z_end - z_start)
            if L_branch < 1e-12:
                continue
            z_mid = 0.5 * (z_start + z_end)
            arm_beta = sign * beta
            st_a, ct_a = math.sin(arm_beta), math.cos(arm_beta)
            dz = Z - z_mid
            t = dp * st_a + dz * ct_a
            n = dp * ct_a - dz * st_a
            s = np.clip((t / (0.5 * L_branch) + 1) * 0.5, 0, 1)
            if sign < 0:
                s = 1.0 - s
            w_local = _taper_width(s, w_branch_nondim, taper_ratio)
            in_arm = (np.abs(t) <= 0.5 * L_branch) & (np.abs(n) <= 0.5 * w_local)
            dist_from_belt = np.abs(Z - z_start)
            ramp_len = ramp_frac * L_branch
            s_ramp = np.clip(dist_from_belt / max(ramp_len, 1e-12), 0, 1)
            depth_profile = depth_nondim * _raised_cosine_ramp(s_ramp)
            if apex_radius_frac > 0:
                r_apex = apex_radius_frac * w_branch_nondim
                dist_apex = np.sqrt(dp ** 2 + (Z - z_start) ** 2)
                apex_mask = dist_apex <= r_apex
                apex_depth = depth_nondim * _raised_cosine_ramp(
                    dist_apex / max(r_apex, 1e-12))
                relief[apex_mask] = np.maximum(relief[apex_mask],
                                                apex_depth[apex_mask])
            relief[in_arm] = np.maximum(relief[in_arm],
                                         depth_profile[in_arm])
    return relief


def build_arc_ramped_relief(
        Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
        belt_half_nondim, curvature_k=0.15, ramp_frac=0.15,
        taper_ratio=1.0):
    """Curved-arc branches from belt to ends.

    Centerline: phi_center(s) = phi0 + side*curvature_k*cell_span*sin(pi*s)
    where s ∈ [0,1] is normalized distance from belt edge to bearing end.
    """
    relief = np.zeros_like(Phi, dtype=float)
    if depth_nondim <= 0 or N_branch_per_side <= 0:
        return relief
    cell_span = 2.0 * math.pi / N_branch_per_side
    for k in range(N_branch_per_side):
        phi_c = cell_span * (k + 0.5)
        for sign in [-1, 1]:
            z_start = sign * belt_half_nondim
            z_end = sign * 1.0
            z_lo, z_hi = min(z_start, z_end), max(z_start, z_end)
            L_branch = z_hi - z_lo
            if L_branch < 1e-12:
                continue
            s = np.clip(np.abs(Z - z_start) / L_branch, 0, 1)
            in_axial = (Z >= z_lo) & (Z <= z_hi)
            phi_offset = sign * curvature_k * cell_span * np.sin(
                np.pi * s)
            dp = _wrap_phi(Phi, phi_c) - phi_offset
            w_local = _taper_width(s, w_branch_nondim, taper_ratio)
            in_width = np.abs(dp) <= 0.5 * w_local
            ramp_len = ramp_frac * L_branch
            s_ramp = np.clip(np.abs(Z - z_start) / max(ramp_len, 1e-12),
                              0, 1)
            depth_profile = depth_nondim * _raised_cosine_ramp(s_ramp)
            mask = in_axial & in_width
            relief[mask] = np.maximum(relief[mask], depth_profile[mask])
    return relief


def apply_partial_coverage(
        relief, Phi,
        mode="full_360",
        protected_lo_deg=105.0,
        protected_hi_deg=175.0,
        phi_loaded_deg=140.0,
        adaptive_width_deg=50.0):
    if mode == "full_360":
        return relief
    out = relief.copy()
    if mode in ("partial_fixed", "protect_loaded_union"):
        lo = math.radians(protected_lo_deg)
        hi = math.radians(protected_hi_deg)
    elif mode == "partial_adaptive":
        center = math.radians(phi_loaded_deg)
        half = math.radians(adaptive_width_deg / 2.0)
        lo, hi = center - half, center + half
    else:
        raise ValueError(f"unknown coverage mode: {mode!r}")
    phi_mod = np.mod(Phi, 2 * math.pi)
    lo_mod = lo % (2 * math.pi)
    hi_mod = hi % (2 * math.pi)
    if lo_mod <= hi_mod:
        protected = (phi_mod >= lo_mod) & (phi_mod <= hi_mod)
    else:
        protected = (phi_mod >= lo_mod) | (phi_mod <= hi_mod)
    out[protected] = 0.0
    return out


def get_branch_centers(N_branch_per_side):
    """Return phi centers of all branches (rad)."""
    cell_span = 2.0 * math.pi / N_branch_per_side
    return np.array([cell_span * (k + 0.5)
                     for k in range(N_branch_per_side)])


def get_removed_branches(all_centers_deg, protected_lo_deg, protected_hi_deg):
    """Return indices and centers of branches removed by partial coverage."""
    removed = []
    for i, phi_deg in enumerate(all_centers_deg):
        phi_mod = phi_deg % 360.0
        lo = protected_lo_deg % 360.0
        hi = protected_hi_deg % 360.0
        if lo <= hi:
            if lo <= phi_mod <= hi:
                removed.append(i)
        else:
            if phi_mod >= lo or phi_mod <= hi:
                removed.append(i)
    return removed


def build_relief(
        Phi, Z, variant, depth_nondim, N_branch_per_side,
        w_branch_nondim, belt_half_nondim,
        beta_deg=20.0, ramp_frac=0.15, taper_ratio=1.0,
        apex_radius_frac=0.5, curvature_k=0.15,
        chirality="pump_to_belt",
        coverage_mode="full_360",
        protected_lo_deg=105.0, protected_hi_deg=175.0,
        phi_loaded_deg=140.0, adaptive_width_deg=50.0):
    if variant == "straight_ramped":
        relief = build_straight_ramped_relief(
            Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
            belt_half_nondim, ramp_frac, taper_ratio)
    elif variant == "half_herringbone_ramped":
        relief = build_half_herringbone_ramped_relief(
            Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
            belt_half_nondim, beta_deg, ramp_frac, taper_ratio,
            apex_radius_frac, chirality)
    elif variant == "arc_ramped":
        relief = build_arc_ramped_relief(
            Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
            belt_half_nondim, curvature_k, ramp_frac, taper_ratio)
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    return apply_partial_coverage(
        relief, Phi, coverage_mode,
        protected_lo_deg, protected_hi_deg,
        phi_loaded_deg, adaptive_width_deg)


# ─── Stage J façade ────────────────────────────────────────────────


def build_herringbone_groove_relief(
    Phi: np.ndarray,
    Z: np.ndarray,
    *,
    depth_nondim: float,
    N_branch_per_side: int,
    w_branch_nondim: float,
    belt_half_nondim: float,
    beta_deg: float = 45.0,
    ramp_frac: float = 0.15,
    taper_ratio: float = 0.6,
    apex_radius_frac: float = 0.5,
    chirality: str = "pump_to_edge",
    coverage_mode: str = "protect_loaded_union",
    protected_lo_deg: float = 105.0,
    protected_hi_deg: float = 175.0,
    phi_loaded_deg: float = 140.0,
    adaptive_width_deg: float = 50.0,
    variant: str = "half_herringbone_ramped",
) -> np.ndarray:
    """Stage J groove builder façade.

    Returns non-negative non-dimensional relief Δh/c shaped like
    ``Phi``. ``coverage_mode`` controls where the relief is masked
    out:

    * ``protect_loaded_union`` / ``partial_fixed`` — protect the
      [protected_lo_deg, protected_hi_deg] sector (G4 anchor uses
      105°-175° on the loaded side).
    * ``partial_adaptive`` — protect a window of width
      ``adaptive_width_deg`` centred on ``phi_loaded_deg``.
    * ``full_360`` — no protection.
    * ``phi_window`` — the inverse of the protected-sector idea:
      keep relief only inside [protected_lo_deg, protected_hi_deg]
      (used for the diagnostic 180°-360° divergent placement).

    The ``variant`` argument is forwarded to :func:`build_relief` so
    future work can plug in straight or arc reliefs without changing
    the diesel runner.
    """
    if coverage_mode == "phi_window":
        full = build_relief(
            Phi, Z, variant=variant,
            depth_nondim=float(depth_nondim),
            N_branch_per_side=int(N_branch_per_side),
            w_branch_nondim=float(w_branch_nondim),
            belt_half_nondim=float(belt_half_nondim),
            beta_deg=float(beta_deg),
            ramp_frac=float(ramp_frac),
            taper_ratio=float(taper_ratio),
            apex_radius_frac=float(apex_radius_frac),
            chirality=str(chirality),
            coverage_mode="full_360",
        )
        lo = np.deg2rad(float(protected_lo_deg))
        hi = np.deg2rad(float(protected_hi_deg))
        phi_mod = np.mod(Phi, 2.0 * np.pi)
        lo_mod = lo % (2.0 * np.pi)
        hi_mod = hi % (2.0 * np.pi)
        if lo_mod <= hi_mod:
            keep = (phi_mod >= lo_mod) & (phi_mod <= hi_mod)
        else:
            keep = (phi_mod >= lo_mod) | (phi_mod <= hi_mod)
        out = np.zeros_like(full)
        out[keep] = full[keep]
        return out
    return build_relief(
        Phi, Z, variant=variant,
        depth_nondim=float(depth_nondim),
        N_branch_per_side=int(N_branch_per_side),
        w_branch_nondim=float(w_branch_nondim),
        belt_half_nondim=float(belt_half_nondim),
        beta_deg=float(beta_deg),
        ramp_frac=float(ramp_frac),
        taper_ratio=float(taper_ratio),
        apex_radius_frac=float(apex_radius_frac),
        chirality=str(chirality),
        coverage_mode=str(coverage_mode),
        protected_lo_deg=float(protected_lo_deg),
        protected_hi_deg=float(protected_hi_deg),
        phi_loaded_deg=float(phi_loaded_deg),
        adaptive_width_deg=float(adaptive_width_deg),
    )


def relief_stats(relief: np.ndarray) -> dict:
    """Quick descriptive stats for the builder output — used by the
    summary writer + the geometry-safety contract tests."""
    a = np.asarray(relief, dtype=float)
    return dict(
        relief_min=float(np.min(a)) if a.size else 0.0,
        relief_max=float(np.max(a)) if a.size else 0.0,
        relief_mean=float(np.mean(a)) if a.size else 0.0,
        relief_nonzero_frac=float(np.mean(a > 0.0)) if a.size else 0.0,
        has_nan=bool(np.any(np.isnan(a))),
        has_inf=bool(np.any(np.isinf(a))),
    )


__all__ = [
    "build_straight_ramped_relief",
    "build_half_herringbone_ramped_relief",
    "build_arc_ramped_relief",
    "apply_partial_coverage",
    "build_relief",
    "get_branch_centers",
    "get_removed_branches",
    "build_herringbone_groove_relief",
    "relief_stats",
]
