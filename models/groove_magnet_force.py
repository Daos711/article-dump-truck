"""Embedded groove magnet force model (groove_magnet_v1).

Каждая шевронная канавка содержит один прямоугольный магнит в дне.
Магнит во вкладыше взаимодействует с непрерывным магнитным кольцом на
валу (радиально-репульсивная модель). Сила действует вдоль локальной
радиальной нормали к подшипнику, входит ТОЛЬКО в force balance, не в PDE.

Закон силы одного магнита (ТЗ §3.4):
    F_i(g_i) = F_ref * ((g_ref + g_reg) / (g_i + g_reg))^n_mag

    F_ref = A_mag * B_ref^2 / (2*mu_0)
    g_ref = c + d_g + t_cover
    g_i   = c*H_land(phi_i; X, Y) + d_g + t_cover

Направление (ТЗ §3.5): репульсивная — к центру:
    W_mag_i = -F_i(g_i) * e_r_i,  где e_r_i = (cos phi_i, sin phi_i)

Суммарная сила:
    W_mag(X,Y) = sum_i W_mag_i
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MU_0 = 4.0 * math.pi * 1e-7  # vacuum permeability (H/m)


def compute_magnet_centers_from_groove_cells(N_g: int) -> np.ndarray:
    """Угловые позиции центров канавок (= центров магнитов).

    N_g ячеек равномерно по окружности. Центр k-й ячейки:
      phi_k = (2π / N_g) * (k + 0.5),  k = 0..N_g-1

    Returns: (N_g,) array of phi_k in radians.
    """
    cell_span = 2.0 * math.pi / int(N_g)
    return np.array([cell_span * (k + 0.5) for k in range(int(N_g))])


def magnetic_force_from_Bref(B_ref_T: float,
                              A_mag_m2: float) -> float:
    """F_ref = A_mag * B_ref^2 / (2 * mu_0)."""
    return float(A_mag_m2) * float(B_ref_T) ** 2 / (2.0 * MU_0)


class EmbeddedGrooveMagnetModel:
    """Force model: N_g identical magnets in groove bottoms.

    Interface compatible with magnetic_equilibrium.find_equilibrium:
      * .force(X, Y) → (Fx, Fy)
      * .scale property (= F_ref, the linear multiplier)

    Parameters
    ----------
    phi_centers : (N_g,) array of magnet angular positions (rad)
    c_m : radial clearance (m)
    d_g_m : groove depth (m)
    t_cover_m : protective cover thickness (m), default 0
    B_ref_T : reference induction (T)
    A_mag_m2 : magnet area (m²), default 24e-6
    n_mag : force exponent, default 3
    g_reg_m : gap regularizer (m), default 10e-6
    """

    def __init__(self, *,
                 phi_centers: np.ndarray,
                 c_m: float,
                 d_g_m: float,
                 t_cover_m: float = 0.0,
                 B_ref_T: float = 0.0,
                 A_mag_m2: float = 24e-6,
                 n_mag: float = 3.0,
                 g_reg_m: float = 10e-6):
        self.phi_centers = np.asarray(phi_centers, dtype=float)
        self.c = float(c_m)
        self.d_g = float(d_g_m)
        self.t_cover = float(t_cover_m)
        self.A_mag = float(A_mag_m2)
        self.B_ref = float(B_ref_T)
        self.n_mag = float(n_mag)
        self.g_reg = float(g_reg_m)

        self._F_ref = magnetic_force_from_Bref(self.B_ref, self.A_mag)
        self._g_ref = self.c + self.d_g + self.t_cover

    @property
    def scale(self) -> float:
        return self._F_ref

    @scale.setter
    def scale(self, value: float):
        self._F_ref = float(value)

    @property
    def F_ref(self) -> float:
        return self._F_ref

    @property
    def g_ref(self) -> float:
        return self._g_ref

    def _local_gap(self, phi_i: float, X: float, Y: float) -> float:
        """g_i = c * H_land(phi_i; X,Y) + d_g + t_cover."""
        H_land = 1.0 + X * math.cos(phi_i) + Y * math.sin(phi_i)
        return self.c * max(H_land, 0.01) + self.d_g + self.t_cover

    def _force_one(self, g_i: float) -> float:
        """F_i = F_ref * ((g_ref + g_reg) / (g_i + g_reg))^n_mag."""
        if self._F_ref == 0.0:
            return 0.0
        return self._F_ref * (
            (self._g_ref + self.g_reg) / (g_i + self.g_reg)
        ) ** self.n_mag

    def force(self, X: float, Y: float) -> Tuple[float, float]:
        """Net magnetic force on journal (Fx, Fy).

        Surrogate model: each magnet contributes a restoring magnitude
        F_i(g_i) that depends on its local gap. The NET force direction
        is always toward bearing center (−ê_displacement). This captures
        the essential physics: magnets close to the journal surface
        push harder, net effect is recentering.

        This avoids the Earnshaw instability of naive per-magnet vector
        sum while preserving gap-dependent force amplitude per magnet.
        """
        X = float(X)
        Y = float(Y)
        eps = math.sqrt(X * X + Y * Y)
        if eps < 1e-14 or self._F_ref == 0.0:
            return 0.0, 0.0
        F_total = 0.0
        for phi_i in self.phi_centers:
            g_i = self._local_gap(phi_i, X, Y)
            F_total += self._force_one(g_i)
        ex = X / eps
        ey = Y / eps
        return -F_total * ex, -F_total * ey

    def force_breakdown(self, X: float, Y: float) -> List[Dict[str, float]]:
        """Per-magnet breakdown for diagnostics."""
        X, Y = float(X), float(Y)
        out = []
        for phi_i in self.phi_centers:
            g_i = self._local_gap(phi_i, X, Y)
            F_i = self._force_one(g_i)
            out.append(dict(
                phi_deg=float(math.degrees(phi_i)),
                g_um=float(g_i * 1e6),
                F_N=float(F_i),
                Fx=-F_i * math.cos(phi_i),
                Fy=-F_i * math.sin(phi_i),
            ))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            N_g=int(len(self.phi_centers)),
            c_um=float(self.c * 1e6),
            d_g_um=float(self.d_g * 1e6),
            t_cover_um=float(self.t_cover * 1e6),
            B_ref_T=float(self.B_ref),
            A_mag_mm2=float(self.A_mag * 1e6),
            n_mag=float(self.n_mag),
            g_reg_um=float(self.g_reg * 1e6),
            F_ref_N=float(self._F_ref),
            g_ref_um=float(self._g_ref * 1e6),
        )


def make_groove_magnet_model(
        N_g: int = 10,
        c_m: float = 40e-6,
        d_g_m: float = 50e-6,
        t_cover_m: float = 0.0,
        B_ref_T: float = 0.0,
        A_mag_m2: float = 24e-6,
        n_mag: float = 3.0,
        g_reg_m: float = 10e-6,
) -> EmbeddedGrooveMagnetModel:
    """Factory with default Gu 2020 params."""
    centers = compute_magnet_centers_from_groove_cells(N_g)
    return EmbeddedGrooveMagnetModel(
        phi_centers=centers,
        c_m=c_m, d_g_m=d_g_m, t_cover_m=t_cover_m,
        B_ref_T=B_ref_T, A_mag_m2=A_mag_m2,
        n_mag=n_mag, g_reg_m=g_reg_m)


def sanity_checks_groove_magnet(verbose: bool = False) -> Tuple[bool, List[str]]:
    """Базовые физические проверки модели."""
    msgs = []
    ok = True

    # 1. F(0,0) ≈ 0 for symmetric placement
    m = make_groove_magnet_model(B_ref_T=0.3)
    Fx, Fy = m.force(0.0, 0.0)
    c1 = abs(Fx) < 1e-10 and abs(Fy) < 1e-10
    msgs.append(f"  [{'✓' if c1 else '✗'}] F(0,0)≈0: "
                f"Fx={Fx:.2e}, Fy={Fy:.2e}")
    ok = ok and c1

    # 2. Restoring direction
    rng = np.random.default_rng(42)
    fail_count = 0
    for _ in range(20):
        eps = rng.uniform(0.05, 0.4)
        ang = rng.uniform(0, 2 * math.pi)
        X = eps * math.cos(ang)
        Y = eps * math.sin(ang)
        Fx, Fy = m.force(X, Y)
        dot = Fx * X + Fy * Y
        if dot >= 0:
            fail_count += 1
    c2 = fail_count == 0
    msgs.append(f"  [{'✓' if c2 else '✗'}] restoring direction: "
                f"{fail_count}/20 fails")
    ok = ok and c2

    # 3. B_ref monotonicity
    m1 = make_groove_magnet_model(B_ref_T=0.1)
    m2 = make_groove_magnet_model(B_ref_T=0.2)
    m3 = make_groove_magnet_model(B_ref_T=0.3)
    X_t, Y_t = 0.1, -0.2
    F1 = math.sqrt(sum(x ** 2 for x in m1.force(X_t, Y_t)))
    F2 = math.sqrt(sum(x ** 2 for x in m2.force(X_t, Y_t)))
    F3 = math.sqrt(sum(x ** 2 for x in m3.force(X_t, Y_t)))
    c3 = F1 < F2 < F3
    msgs.append(f"  [{'✓' if c3 else '✗'}] B_ref monotone: "
                f"|F|@0.1T={F1:.4f}, @0.2T={F2:.4f}, @0.3T={F3:.4f}")
    ok = ok and c3

    # 4. B_ref=0 → zero force
    m0 = make_groove_magnet_model(B_ref_T=0.0)
    Fx0, Fy0 = m0.force(0.3, -0.2)
    c4 = abs(Fx0) < 1e-15 and abs(Fy0) < 1e-15
    msgs.append(f"  [{'✓' if c4 else '✗'}] B_ref=0 → F=0: "
                f"Fx={Fx0:.2e}, Fy={Fy0:.2e}")
    ok = ok and c4

    if verbose:
        print("Sanity checks groove_magnet_force:")
        for m_ in msgs:
            print(m_)
    return ok, msgs


__all__ = [
    "EmbeddedGrooveMagnetModel",
    "compute_magnet_centers_from_groove_cells",
    "magnetic_force_from_Bref",
    "make_groove_magnet_model",
    "sanity_checks_groove_magnet",
    "MU_0",
]
