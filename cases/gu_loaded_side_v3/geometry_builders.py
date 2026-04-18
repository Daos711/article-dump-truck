"""Feed-consistent geometry builders for v3 central-feed branch.

Priority builders (Pass 1):
  * straight_ramped — straight axial branches with C1-smooth depth ramp
  * half_herringbone_ramped — angled branches with rounded apex + ramp

Future (Pass 1 fail → implement):
  * arc_ramped
  * arc_herringbone_ramped

All builders return ΔH relief array (same shape as Phi/Z meshgrids).
Call: H = H0 + build_relief(Phi, Z, params)

Partial coverage wrapper masks grooves outside active circumferential
windows, preserving a smooth protected pressure-peak zone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Ramp / taper / apex helpers ───────────────────────────────────

def _raised_cosine_ramp(s: np.ndarray) -> np.ndarray:
    """C1-smooth ramp: 0 at s=0, 1 at s>=1. Raised-cosine profile."""
    out = np.zeros_like(s, dtype=float)
    mask = (s > 0) & (s < 1)
    out[mask] = 0.5 * (1.0 - np.cos(np.pi * s[mask]))
    out[s >= 1] = 1.0
    return out


def _taper_width(s: np.ndarray, w_in: float, taper_ratio: float) -> np.ndarray:
    """Width along branch: w_in at s=0, w_in*taper_ratio at s=1."""
    return w_in * (1.0 - (1.0 - taper_ratio) * s)


def _wrap_phi(Phi: np.ndarray, phi_c: float) -> np.ndarray:
    d = Phi - phi_c
    return d - 2.0 * math.pi * np.round(d / (2.0 * math.pi))


# ─── Straight ramped builder ──────────────────────────────────────

def build_straight_ramped_relief(
        Phi: np.ndarray, Z: np.ndarray,
        depth_nondim: float,
        N_branch_per_side: int,
        w_branch_nondim: float,
        belt_half_nondim: float,
        ramp_frac: float = 0.15,
        taper_ratio: float = 1.0,
) -> np.ndarray:
    """Straight axial branches from belt to bearing ends.

    Depth ramp: C1-smooth raised-cosine over first ramp_frac of branch.
    Taper: width decreases from w_in at belt to w_in*taper_ratio at end.
    """
    relief = np.zeros_like(Phi, dtype=float)
    if depth_nondim <= 0 or N_branch_per_side <= 0:
        return relief

    cell_span = 2.0 * math.pi / N_branch_per_side

    for k in range(N_branch_per_side):
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi(Phi, phi_c)

        for sign in [-1, 1]:
            # Branch from belt edge to bearing end
            z_start = sign * belt_half_nondim
            z_end = sign * 1.0
            z_lo = min(z_start, z_end)
            z_hi = max(z_start, z_end)
            L_branch = z_hi - z_lo
            if L_branch < 1e-12:
                continue

            # Normalized coordinate along branch: 0=belt, 1=end
            s = np.abs(Z - z_start) / L_branch
            in_axial = (Z >= z_lo) & (Z <= z_hi) if sign > 0 else \
                       (Z <= z_hi) & (Z >= z_lo)

            # Local width with taper
            w_local = _taper_width(s, w_branch_nondim, taper_ratio)
            in_width = np.abs(dp) <= 0.5 * w_local

            # Depth with ramp
            ramp_len = ramp_frac * L_branch
            s_ramp = np.clip((np.abs(Z - z_start)) / max(ramp_len, 1e-12),
                              0.0, 1.0)
            depth_profile = depth_nondim * _raised_cosine_ramp(s_ramp)

            mask = in_axial & in_width
            relief[mask] = np.maximum(relief[mask], depth_profile[mask])

    return relief


# ─── Half-herringbone ramped builder ──────────────────────────────

def build_half_herringbone_ramped_relief(
        Phi: np.ndarray, Z: np.ndarray,
        depth_nondim: float,
        N_branch_per_side: int,
        w_branch_nondim: float,
        belt_half_nondim: float,
        beta_deg: float = 20.0,
        ramp_frac: float = 0.15,
        taper_ratio: float = 1.0,
        apex_radius_frac: float = 0.5,
) -> np.ndarray:
    """Angled branches from belt to ends with rounded apex + C1 ramp.

    Each cell has two arms (left/right of belt) at angles +/-beta.
    Apex region (where arms meet at belt edge) is smoothed with a
    circular blend of radius apex_radius_frac * w_branch.
    """
    relief = np.zeros_like(Phi, dtype=float)
    if depth_nondim <= 0 or N_branch_per_side <= 0:
        return relief

    beta = math.radians(float(beta_deg))
    cell_span = 2.0 * math.pi / N_branch_per_side
    st = math.sin(beta)
    ct = math.cos(beta)

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
            arm_beta = sign * beta  # mirror for left/right side

            # Parallelogram mask (rotated by arm_beta)
            st_a = math.sin(arm_beta)
            ct_a = math.cos(arm_beta)
            dz = Z - z_mid
            t = dp * st_a + dz * ct_a  # along arm axis
            n = dp * ct_a - dz * st_a  # perpendicular

            # Normalized progress along arm: 0=belt, 1=end
            s = np.clip((t / (0.5 * L_branch) + 1.0) * 0.5, 0.0, 1.0)
            if sign < 0:
                s = 1.0 - s

            # Taper width
            w_local = _taper_width(s, w_branch_nondim, taper_ratio)
            in_arm = (np.abs(t) <= 0.5 * L_branch) & (np.abs(n) <= 0.5 * w_local)

            # Depth ramp from belt edge
            dist_from_belt = np.abs(Z - z_start)
            ramp_len = ramp_frac * L_branch
            s_ramp = np.clip(dist_from_belt / max(ramp_len, 1e-12), 0.0, 1.0)
            depth_profile = depth_nondim * _raised_cosine_ramp(s_ramp)

            # Apex rounding: smooth blend near belt edge
            if apex_radius_frac > 0:
                r_apex = apex_radius_frac * w_branch_nondim
                dist_apex = np.sqrt(dp ** 2 + (Z - z_start) ** 2)
                apex_mask = dist_apex <= r_apex
                # In apex region, apply circular blend
                apex_depth = depth_nondim * _raised_cosine_ramp(
                    dist_apex / max(r_apex, 1e-12))
                relief[apex_mask] = np.maximum(relief[apex_mask],
                                                apex_depth[apex_mask])

            relief[in_arm] = np.maximum(relief[in_arm],
                                         depth_profile[in_arm])

    return relief


# ─── Partial coverage wrapper ────────────────────────────────────

def apply_partial_coverage(
        relief: np.ndarray,
        Phi: np.ndarray,
        mode: str = "full_360",
        protected_lo_deg: float = 80.0,
        protected_hi_deg: float = 130.0,
        phi_loaded_deg: float = 140.0,
        adaptive_width_deg: float = 50.0,
) -> np.ndarray:
    """Zero out relief inside protected smooth zone.

    Modes:
      full_360 — no masking
      partial_fixed — smooth zone [protected_lo_deg, protected_hi_deg]
      partial_adaptive — smooth zone centered on phi_loaded ± width/2
    """
    if mode == "full_360":
        return relief

    out = relief.copy()

    if mode == "partial_fixed":
        lo = math.radians(protected_lo_deg)
        hi = math.radians(protected_hi_deg)
    elif mode == "partial_adaptive":
        center = math.radians(phi_loaded_deg)
        half = math.radians(adaptive_width_deg / 2.0)
        lo = center - half
        hi = center + half
    else:
        raise ValueError(f"unknown coverage mode: {mode!r}")

    # Handle wrap-around
    phi_mod = np.mod(Phi, 2 * math.pi)
    lo_mod = lo % (2 * math.pi)
    hi_mod = hi % (2 * math.pi)

    if lo_mod <= hi_mod:
        protected = (phi_mod >= lo_mod) & (phi_mod <= hi_mod)
    else:
        protected = (phi_mod >= lo_mod) | (phi_mod <= hi_mod)

    out[protected] = 0.0
    return out


# ─── Unified builder interface ───────────────────────────────────

def build_relief(
        Phi: np.ndarray, Z: np.ndarray,
        variant: str,
        depth_nondim: float,
        N_branch_per_side: int,
        w_branch_nondim: float,
        belt_half_nondim: float,
        beta_deg: float = 20.0,
        ramp_frac: float = 0.15,
        taper_ratio: float = 1.0,
        apex_radius_frac: float = 0.5,
        curvature_k: float = 0.0,
        coverage_mode: str = "full_360",
        protected_lo_deg: float = 80.0,
        protected_hi_deg: float = 130.0,
        phi_loaded_deg: float = 140.0,
        adaptive_width_deg: float = 50.0,
) -> np.ndarray:
    """Unified entry point for all geometry builders."""
    if variant == "straight_ramped":
        relief = build_straight_ramped_relief(
            Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
            belt_half_nondim, ramp_frac, taper_ratio)
    elif variant == "half_herringbone_ramped":
        relief = build_half_herringbone_ramped_relief(
            Phi, Z, depth_nondim, N_branch_per_side, w_branch_nondim,
            belt_half_nondim, beta_deg, ramp_frac, taper_ratio,
            apex_radius_frac)
    elif variant in ("arc_ramped", "arc_herringbone_ramped"):
        raise NotImplementedError(
            f"{variant} not yet implemented; implement only if "
            f"straight_ramped and half_herringbone_ramped fail Pass 1")
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    return apply_partial_coverage(
        relief, Phi, coverage_mode,
        protected_lo_deg, protected_hi_deg,
        phi_loaded_deg, adaptive_width_deg)


__all__ = [
    "build_relief",
    "build_straight_ramped_relief",
    "build_half_herringbone_ramped_relief",
    "apply_partial_coverage",
]
