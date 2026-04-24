"""Stage I — surrogate heavy-duty load cycle + quasi-static cycle runner.

Load cycle: one dominant firing lobe (Wiebe-like gas) + inertia
(slider-crank 1st+2nd order) + preload floor. Vector output (Wx, Wy).

Cycle runner: for each crank angle, solve stationary equilibrium with
warm-start from previous angle. No squeeze term.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Surrogate load cycle ──────────────────────────────────────────

def build_surrogate_heavyduty_v1(
        n_points: int = 72,
        W_gas_peak_N: float = 25000.0,
        W_inertia_amp_N: float = 5000.0,
        W_preload_N: float = 500.0,
        firing_center_deg: float = 370.0,
        firing_width_deg: float = 60.0,
) -> Dict[str, Any]:
    """Build single-cylinder surrogate load cycle.

    Returns dict with phi_crank_deg, Wx_N, Wy_N arrays.
    Load is in bearing frame: positive Wy = downward on journal.
    """
    phi = np.linspace(0, 720, n_points, endpoint=False)
    phi_rad = np.deg2rad(phi)

    # Gas force: Wiebe-like envelope centered at firing_center
    sigma_gas = np.deg2rad(firing_width_deg) / 2.5
    gas_env = np.exp(-0.5 * ((np.deg2rad(phi) - np.deg2rad(firing_center_deg))
                              / sigma_gas) ** 2)
    W_gas = W_gas_peak_N * gas_env

    # Inertia: 1st + 2nd order slider-crank (simplified)
    omega_half = np.deg2rad(phi) / 2.0  # half-speed for 4-stroke
    W_inertia = W_inertia_amp_N * (
        np.cos(omega_half) + 0.25 * np.cos(2 * omega_half))

    # Total vertical load (Wy positive = downward)
    Wy = W_gas + W_inertia + W_preload_N

    # Small lateral component from conrod angle
    Wx = 0.15 * W_inertia_amp_N * np.sin(omega_half)

    return dict(
        phi_crank_deg=phi.tolist(),
        Wx_N=Wx.tolist(),
        Wy_N=Wy.tolist(),
        name="surrogate_heavyduty_v1",
        n_points=int(n_points),
    )


def get_load_at_angle(cycle: Dict[str, Any],
                       phi_deg: float) -> Tuple[float, float]:
    """Interpolate (Wx, Wy) at arbitrary crank angle."""
    phi_arr = np.array(cycle["phi_crank_deg"])
    Wx_arr = np.array(cycle["Wx_N"])
    Wy_arr = np.array(cycle["Wy_N"])
    phi_mod = float(phi_deg) % 720.0
    Wx = float(np.interp(phi_mod, phi_arr, Wx_arr))
    Wy = float(np.interp(phi_mod, phi_arr, Wy_arr))
    return Wx, Wy


def get_placement_from_reference(cycle: Dict[str, Any],
                                  phi_ref_deg: float = 90.0
                                  ) -> Dict[str, float]:
    """Shell-fixed placement: determine loaded/unloaded azimuth
    from load direction at reference crank angle."""
    Wx, Wy = get_load_at_angle(cycle, phi_ref_deg)
    W_angle = math.degrees(math.atan2(Wy, Wx)) % 360.0
    phi_loaded = W_angle
    phi_unloaded = (W_angle + 180.0) % 360.0
    return dict(
        phi_ref_deg=phi_ref_deg,
        phi_loaded_deg=phi_loaded,
        phi_unloaded_deg=phi_unloaded,
        Wx_ref=Wx, Wy_ref=Wy,
    )


# ─── Quasi-static validity diagnostic ─────────────────────────────

def quasistatic_ratio(eps_arr, phi_arr_deg, omega_rad_s, R, c):
    """R_sq ≈ |dε/dt| * c / (ω * R * ε) — ratio of squeeze to wedge.

    Small R_sq (<< 1) means quasi-static is reasonable.
    """
    phi_rad = np.deg2rad(np.array(phi_arr_deg))
    eps = np.array(eps_arr)
    # dt between consecutive angles
    dphi = np.diff(phi_rad)
    dphi[dphi == 0] = 1e-12
    dt = dphi / omega_rad_s
    deps = np.diff(eps)
    deps_dt = deps / dt
    eps_mid = 0.5 * (eps[:-1] + eps[1:])
    U = omega_rad_s * R
    R_sq = np.abs(deps_dt) * c / (U * np.maximum(eps_mid, 1e-6))
    return R_sq.tolist()
