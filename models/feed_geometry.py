"""Feed-consistent shallow groove geometry (gu_loaded_side_v2).

Central smooth belt + symmetric straight (or half-herringbone) branches
extending from belt edges to bearing ends. Feed window metadata for
supply-pressure BC.

All coordinates in nondimensional (phi_rad, Z ∈ [-1,1]).
Depth in nondimensional h/c.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def create_H_with_central_feed_branches(
        H0: np.ndarray,
        depth_nondim: float,
        Phi: np.ndarray, Z: np.ndarray,
        N_branch_per_side: int,
        w_branch_nondim: float,
        belt_half_nondim: float,
        beta_deg: float = 0.0,
        variant: str = "straight",
) -> np.ndarray:
    """Add symmetric branches from central belt to bearing ends.

    Parameters
    ----------
    H0 : base film thickness (2D, shape NZ×Nphi)
    depth_nondim : branch groove depth in h/c
    Phi, Z : meshgrid (2D)
    N_branch_per_side : number of branches on each side (total = 2×N)
    w_branch_nondim : branch width perpendicular to branch axis (in phi rad)
    belt_half_nondim : half-width of smooth central belt (in Z units,
                       i.e. fraction of L/2)
    beta_deg : branch angle to Z-axis (0 = straight axial).
               Used only for variant="half_herringbone".
    variant : "straight" or "half_herringbone"

    Returns
    -------
    H : modified film thickness (branches increase H by depth_nondim)
    """
    H = np.array(H0, dtype=float)
    if depth_nondim <= 0 or N_branch_per_side <= 0:
        return H

    cell_span = 2.0 * math.pi / N_branch_per_side
    beta = math.radians(float(beta_deg)) if variant == "half_herringbone" else 0.0

    for k in range(N_branch_per_side):
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi(Phi, phi_c)

        # Left side: Z from -1 to -belt_half
        _add_branch(H, dp, Z, phi_c, w_branch_nondim,
                    z_start=-1.0, z_end=-belt_half_nondim,
                    beta=beta, depth=depth_nondim)

        # Right side: Z from +belt_half to +1
        _add_branch(H, dp, Z, phi_c, w_branch_nondim,
                    z_start=belt_half_nondim, z_end=1.0,
                    beta=-beta, depth=depth_nondim)

    return H


def _wrap_phi(Phi: np.ndarray, phi_c: float) -> np.ndarray:
    d = Phi - phi_c
    return d - 2.0 * math.pi * np.round(d / (2.0 * math.pi))


def _add_branch(H, dp, Z, phi_c, w, z_start, z_end, beta, depth):
    """Add one branch strip (possibly angled) to H."""
    z_lo = min(z_start, z_end)
    z_hi = max(z_start, z_end)
    L_branch = z_hi - z_lo
    if L_branch < 1e-12:
        return
    z_mid = 0.5 * (z_lo + z_hi)

    if abs(beta) < 1e-8:
        # Straight branch: simple rectangle
        mask = (np.abs(dp) <= 0.5 * w) & (Z >= z_lo) & (Z <= z_hi)
    else:
        # Angled branch: parallelogram
        st = math.sin(beta)
        ct = math.cos(beta)
        dz = Z - z_mid
        t = dp * st + dz * ct
        n = dp * ct - dz * st
        mask = (np.abs(t) <= 0.5 * L_branch) & (np.abs(n) <= 0.5 * w)

    H[mask] += depth


def feed_window_metadata(phi_feed_deg: float = 300.0,
                          phi_feed_half_deg: float = 5.0,
                          z_belt_half: float = 0.15,
                          ) -> Dict[str, Any]:
    """Return metadata dict describing the feed window location.

    The feed window is where supply pressure BC would be applied.
    It sits in the central belt at phi_feed, within the belt zone.
    """
    return dict(
        phi_feed_deg=float(phi_feed_deg),
        phi_feed_half_deg=float(phi_feed_half_deg),
        phi_feed_lo_deg=float(phi_feed_deg - phi_feed_half_deg),
        phi_feed_hi_deg=float(phi_feed_deg + phi_feed_half_deg),
        z_belt_half=float(z_belt_half),
    )


def feed_geometry_params(d_g_m: float, c_m: float,
                          N_branch: int = 3,
                          w_g_m: float = 0.004,
                          R_m: float = 0.027,
                          belt_half_frac: float = 0.15,
                          beta_deg: float = 0.0,
                          variant: str = "straight",
                          ) -> Dict[str, Any]:
    """Build nondimensional parameter dict for the feed geometry."""
    return dict(
        depth_nondim=float(d_g_m / c_m),
        N_branch_per_side=int(N_branch),
        w_branch_nondim=float(w_g_m / R_m),
        belt_half_nondim=float(belt_half_frac),
        beta_deg=float(beta_deg),
        variant=str(variant),
        d_g_um=float(d_g_m * 1e6),
        belt_pct=float(belt_half_frac * 200),
    )


__all__ = [
    "create_H_with_central_feed_branches",
    "feed_window_metadata",
    "feed_geometry_params",
    "build_feed_mask",
    "build_feed_mask_variant",
    "p_supply_to_g_bc",
]


# ─── Stage H: explicit supply pressure ────────────────────────────

def build_feed_mask(
        Phi: np.ndarray, Z: np.ndarray,
        phi_feed_deg: float = 300.0,
        phi_feed_half_deg: float = 5.0,
        z_belt_half: float = 0.15,
) -> np.ndarray:
    """Build bool Dirichlet mask for central feed window.

    The feed window is a rectangle in (phi, Z) inside the central belt.
    Wrap-safe: works correctly for windows crossing phi=0/360°.

    Returns contiguous bool array, same shape as Phi.
    """
    phi_c = math.radians(float(phi_feed_deg) % 360.0)
    half = math.radians(float(phi_feed_half_deg))
    dphi = Phi - phi_c
    dphi = dphi - 2.0 * math.pi * np.round(dphi / (2.0 * math.pi))
    mask_phi = np.abs(dphi) <= half
    mask_z = np.abs(Z) <= float(z_belt_half)
    mask = mask_phi & mask_z
    return np.ascontiguousarray(mask)


def p_supply_to_g_bc(
        p_supply_Pa: float,
        mu: float, omega: float, R: float, c: float,
) -> float:
    """Convert dimensional supply pressure to nondimensional g_bc.

    g_bc = p_supply_Pa / p_scale
    p_scale = 6 * mu * omega * (R/c)^2
    """
    p_scale = 6.0 * float(mu) * float(omega) * (float(R) / float(c)) ** 2
    return float(p_supply_Pa) / p_scale


def build_feed_mask_variant(
        Phi: np.ndarray, Z: np.ndarray,
        variant: str,
        *,
        phi_loaded_deg: float = None,
        phi_feed_deg: float = None,
        phi_feed_half_deg: float = 5.0,
        z_belt_half: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build feed mask by variant name.

    Variants:
      belt_wide       — full axial belt |Z|<=z_belt_half, all φ
      point_loaded    — rectangular window at phi_loaded
      point_unloaded  — rectangular window at phi_loaded + 180°

    Returns (mask, meta_dict).
    Raises ValueError if mask is empty.
    """
    meta = dict(variant=variant, z_belt_half=z_belt_half)

    if variant == "belt_wide":
        mask = np.abs(Z) <= float(z_belt_half)
        meta["phi_center_deg"] = None
        meta["phi_half_deg"] = None
    elif variant in ("point_loaded", "point_unloaded"):
        if phi_feed_deg is not None:
            phi_c = float(phi_feed_deg)
        elif phi_loaded_deg is not None:
            if variant == "point_loaded":
                phi_c = float(phi_loaded_deg)
            else:
                phi_c = (float(phi_loaded_deg) + 180.0) % 360.0
        else:
            raise ValueError(
                f"variant={variant!r} requires phi_loaded_deg or "
                f"phi_feed_deg")
        mask = build_feed_mask(Phi, Z, phi_c, phi_feed_half_deg,
                                z_belt_half)
        meta["phi_center_deg"] = phi_c
        meta["phi_half_deg"] = phi_feed_half_deg
    else:
        raise ValueError(f"unknown feed variant: {variant!r}")

    mask = np.ascontiguousarray(mask)
    if not np.any(mask):
        raise ValueError(
            f"feed mask is empty for variant={variant!r} on this grid")
    meta["n_cells"] = int(np.sum(mask))
    meta["frac"] = float(np.mean(mask))
    return mask, meta
