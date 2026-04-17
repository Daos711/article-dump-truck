"""Loaded-side sector magnet model (gu_loaded_side_v1).

Per-magnet VECTOR SUM (not radial surrogate). Each active magnet at
phi_i contributes F_i in the -e_r(phi_i) direction. The net force is
the vector sum — asymmetric subsets produce directional unloading.

Critical difference from groove_magnet_force.EmbeddedGrooveMagnetModel:
that model sums magnitudes and points toward center (isotropic surrogate).
This model preserves per-magnet angular direction.

Force law identical: F_i = F_ref * ((g_ref+g_reg)/(g_i+g_reg))^n_mag
Direction: u_i = (-cos phi_i, -sin phi_i) per magnet.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MU_0 = 4.0 * math.pi * 1e-7


def compute_Fref_from_Bref(B_ref_T: float, A_mag_m2: float) -> float:
    return float(A_mag_m2) * float(B_ref_T) ** 2 / (2.0 * MU_0)


class EmbeddedGrooveSectorMagnetModel:
    """Per-magnet vector sum loaded-side magnet model.

    Compatible with find_equilibrium via .force(X, Y) and .scale property.
    """

    def __init__(self, *,
                 phi_centers: np.ndarray,
                 c_m: float,
                 d_sub_m: float,
                 t_cover_m: float = 0.0,
                 B_ref_T: float = 0.0,
                 A_mag_m2: float = 24e-6,
                 n_mag: float = 3.0,
                 g_reg_m: float = 10e-6):
        self.phi_centers = np.asarray(phi_centers, dtype=float)
        self.c = float(c_m)
        self.d_sub = float(d_sub_m)
        self.t_cover = float(t_cover_m)
        self.A_mag = float(A_mag_m2)
        self.B_ref = float(B_ref_T)
        self.n_mag = float(n_mag)
        self.g_reg = float(g_reg_m)

        self._F_ref = compute_Fref_from_Bref(self.B_ref, self.A_mag)
        self._g_ref = self.c + self.d_sub + self.t_cover

    @property
    def scale(self) -> float:
        return self._F_ref

    @scale.setter
    def scale(self, value: float):
        self._F_ref = float(value)

    @property
    def F_ref(self) -> float:
        return self._F_ref

    def _local_gap(self, phi_i: float, X: float, Y: float) -> float:
        H_land = 1.0 + X * math.cos(phi_i) + Y * math.sin(phi_i)
        return self.c * max(H_land, 0.01) + self.d_sub + self.t_cover

    def _force_one(self, g_i: float) -> float:
        if self._F_ref == 0.0:
            return 0.0
        return self._F_ref * (
            (self._g_ref + self.g_reg) / (g_i + self.g_reg)
        ) ** self.n_mag

    def force(self, X: float, Y: float) -> Tuple[float, float]:
        """Per-magnet attractive vector sum.

        Each magnet at phi_i attracts the journal TOWARD that wall:
            u_i = (+cos phi_i, +sin phi_i)

        For unloaded-side magnets this pulls the journal away from the
        loaded wall → unloading effect. For loaded-side magnets this
        pulls INTO the load → adverse (expected, diagnostic only).
        """
        X, Y = float(X), float(Y)
        Fx, Fy = 0.0, 0.0
        for phi_i in self.phi_centers:
            g_i = self._local_gap(phi_i, X, Y)
            F_i = self._force_one(g_i)
            Fx += F_i * math.cos(phi_i)
            Fy += F_i * math.sin(phi_i)
        return Fx, Fy

    def force_breakdown(self, X: float, Y: float) -> List[Dict[str, float]]:
        X, Y = float(X), float(Y)
        out = []
        for phi_i in self.phi_centers:
            g_i = self._local_gap(phi_i, X, Y)
            F_i = self._force_one(g_i)
            out.append(dict(
                phi_deg=float(math.degrees(phi_i)),
                g_um=float(g_i * 1e6),
                F_N=float(F_i),
                Fx=F_i * math.cos(phi_i),
                Fy=F_i * math.sin(phi_i),
            ))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            model_type="sector_vector_sum",
            N_active=int(len(self.phi_centers)),
            phi_centers_deg=[float(math.degrees(p))
                             for p in self.phi_centers],
            c_um=float(self.c * 1e6),
            d_sub_um=float(self.d_sub * 1e6),
            t_cover_um=float(self.t_cover * 1e6),
            B_ref_T=float(self.B_ref),
            A_mag_mm2=float(self.A_mag * 1e6),
            n_mag=float(self.n_mag),
            g_reg_um=float(self.g_reg * 1e6),
            F_ref_N=float(self._F_ref),
            g_ref_um=float(self._g_ref * 1e6),
        )


def make_sector_magnet_model(
        active_phi_centers: np.ndarray,
        c_m: float, d_sub_m: float,
        B_ref_T: float = 0.0,
        t_cover_m: float = 0.0,
        A_mag_m2: float = 24e-6,
        n_mag: float = 3.0,
        g_reg_m: float = 10e-6,
) -> EmbeddedGrooveSectorMagnetModel:
    return EmbeddedGrooveSectorMagnetModel(
        phi_centers=active_phi_centers,
        c_m=c_m, d_sub_m=d_sub_m, t_cover_m=t_cover_m,
        B_ref_T=B_ref_T, A_mag_m2=A_mag_m2,
        n_mag=n_mag, g_reg_m=g_reg_m)


def sanity_checks_sector_model(verbose: bool = False) -> Tuple[bool, List[str]]:
    msgs = []
    ok = True

    # Use 3-cell UNLOADED-side subset. For Y<0 anchor the loaded side
    # is at phi≈pi/2 (where H is min); unloaded is at phi≈3pi/2.
    cell_span = 2 * math.pi / 10
    phi_unloaded = 3 * math.pi / 2  # opposite of loaded side
    centers = np.array([
        phi_unloaded - cell_span,
        phi_unloaded,
        phi_unloaded + cell_span,
    ])

    # 1. B=0 → zero
    m0 = make_sector_magnet_model(centers, 40e-6, 50e-6, B_ref_T=0.0)
    Fx, Fy = m0.force(0.1, -0.3)
    c1 = abs(Fx) < 1e-15 and abs(Fy) < 1e-15
    msgs.append(f"  [{'✓' if c1 else '✗'}] B=0 → F=0")
    ok = ok and c1

    # 2. monotone with B_ref
    Fs = []
    for b in [0.3, 0.5, 0.8, 1.0]:
        m = make_sector_magnet_model(centers, 40e-6, 50e-6, B_ref_T=b)
        F = math.sqrt(sum(x ** 2 for x in m.force(0.1, -0.3)))
        Fs.append(F)
    c2 = all(Fs[i] < Fs[i + 1] for i in range(len(Fs) - 1))
    msgs.append(f"  [{'✓' if c2 else '✗'}] B_ref monotone: "
                f"{[f'{f:.3f}' for f in Fs]}")
    ok = ok and c2

    # 3. Unloaded-side attractive magnets should pull journal TOWARD
    # unloaded wall (phi≈3pi/2 for Y<0 anchor). With attractive model
    # u_i = (cos phi_i, sin phi_i), magnets near 3pi/2 → Fy < 0.
    # For load in -Y → e_resist = (0, +1). But the unloaded-side pull
    # is in -Y too. This actually reduces the needed hydro support:
    # the equilibrium equation is F_h + F_mag = W_applied. If F_mag
    # pulls in -Y (same as W_applied), then F_h needs to compensate
    # LESS → lower eps → unloading effect.
    # Check: Fy_mag < 0 for unloaded subset at Y<0 anchor (expected).
    m3 = make_sector_magnet_model(centers, 40e-6, 50e-6, B_ref_T=0.5)
    Fx3, Fy3 = m3.force(0.0, -0.5)
    c3 = Fy3 < 0
    msgs.append(f"  [{'✓' if c3 else '✗'}] unloaded-side pull: "
                f"Fy={Fy3:.4f} < 0 (pulls toward unloaded wall)")
    ok = ok and c3

    # 4. vector sum ≠ radial surrogate for asymmetric subset
    from models.groove_magnet_force import make_groove_magnet_model
    m_radial = make_groove_magnet_model(
        N_g=10, c_m=40e-6, d_g_m=50e-6, B_ref_T=0.5)
    Fr_x, Fr_y = m_radial.force(0.1, -0.3)
    Fs_x, Fs_y = m3.force(0.1, -0.3)
    c4 = (abs(Fr_x - Fs_x) > 1e-6 or abs(Fr_y - Fs_y) > 1e-6)
    msgs.append(f"  [{'✓' if c4 else '✗'}] sector ≠ radial surrogate")
    ok = ok and c4

    if verbose:
        print("Sanity checks groove_sector_magnet:")
        for m_ in msgs:
            print(m_)
    return ok, msgs


def build_active_subset(all_phi_centers: np.ndarray,
                         phi_center_deg: float,
                         N_active: int) -> np.ndarray:
    """Select N_active contiguous cells nearest to phi_center_deg.

    Returns array of phi values (rad) for the active subset.
    """
    N_g = len(all_phi_centers)
    phi_c = math.radians(float(phi_center_deg))
    dists = np.abs(np.mod(all_phi_centers - phi_c + math.pi,
                          2 * math.pi) - math.pi)
    center_idx = int(np.argmin(dists))
    half = N_active // 2
    indices = [(center_idx + k) % N_g
               for k in range(-half, N_active - half)]
    return all_phi_centers[indices]


def get_placement_center_deg(phi_loaded_deg: float,
                              mode: str) -> float:
    """Return magnet center angle for given placement mode.

    * 'unloaded' → phi_loaded + 180°
    * 'loaded'   → phi_loaded (diagnostic only)
    """
    if mode == "unloaded":
        return (float(phi_loaded_deg) + 180.0) % 360.0
    if mode == "loaded":
        return float(phi_loaded_deg)
    raise ValueError(f"unknown placement mode: {mode!r}")


__all__ = [
    "EmbeddedGrooveSectorMagnetModel",
    "compute_Fref_from_Bref",
    "make_sector_magnet_model",
    "build_active_subset",
    "get_placement_center_deg",
    "sanity_checks_sector_model",
]
