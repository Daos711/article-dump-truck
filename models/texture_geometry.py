"""Flat-bottom groove geometry builders for herringbone study (Gu 2020).

Два типа:
  * straight_grooves  — прямоугольные канавки вдоль оси Z
  * herringbone_grooves — V-образные (шевронные) канавки

Все работают в безразмерных координатах (φ_rad, Z_nondim ∈ [-1,1]).
Глубина задаётся в безразмерных единицах h/c.

Конвенция одного плеча (clarification 1 из ТЗ):
  Плечо — параллелограмм в (φ, Z) с центром (φ_c, Z_c), шириной w
  (перпендикулярно оси плеча), длиной L_arm (вдоль оси плеча) и углом
  ±β к оси Z. Точка (φ, Z) внутри плеча, если её проекция на ось
  плеча ≤ L_arm/2, а перпендикулярное расстояние ≤ w/2.

Overlap rule (ТЗ §4.5.2): при пересечении двух плеч в зоне apex
используется max(relief_left, relief_right), т.е. глубина ≡ d_g,
а не 2·d_g.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _parallelogram_mask(phi: np.ndarray, Z: np.ndarray,
                         phi_c: float, Z_c: float,
                         L_arm: float, w: float,
                         beta_rad: float) -> np.ndarray:
    """Маска одного плеча (параллелограмм в unfolded domain).

    Ось плеча наклонена на ±β к оси Z.
    Координаты повёрнуты так, что ось плеча → локальная ось t,
    перпендикулярная → ось n.

    Parameters
    ----------
    phi, Z : 2D mesh arrays (same shape)
    phi_c, Z_c : центр плеча
    L_arm : длина вдоль оси плеча
    w : ширина перпендикулярно оси плеча
    beta_rad : угол наклона к оси Z (положительный = clockwise
               при стандартном unfolded mapping)

    Returns
    -------
    bool mask (same shape)
    """
    dp = phi - phi_c
    dz = Z - Z_c
    # Единичные вектора оси плеча (t) и нормали (n).
    # Ось плеча: направление (sin β, cos β) в (φ, Z)-пространстве.
    st = math.sin(beta_rad)
    ct = math.cos(beta_rad)
    # Проекция на ось плеча
    t = dp * st + dz * ct
    # Проекция на нормаль
    n = dp * ct - dz * st
    return (np.abs(t) <= 0.5 * L_arm) & (np.abs(n) <= 0.5 * w)


def _wrap_phi_distance(phi: np.ndarray, phi_c: float) -> np.ndarray:
    """Разность φ − φ_c с учётом циклической обёртки [0, 2π)."""
    d = phi - phi_c
    return d - 2.0 * math.pi * np.round(d / (2.0 * math.pi))


def create_H_with_straight_grooves(
        H0: np.ndarray,
        depth_nondim: float,
        Phi: np.ndarray, Z: np.ndarray,
        N_g: int,
        w_g_nondim: float,
        L_g_nondim: float,
) -> np.ndarray:
    """Добавить N_g прямых (осевых) канавок.

    Канавки равномерно распределены по окружности с полным покрытием.
    Каждая канавка — прямоугольник в (φ, Z) шириной w_g_nondim (по φ)
    и длиной L_g_nondim (по Z).

    Parameters
    ----------
    H0 : base film thickness (2D array, shape NZ×Nphi)
    depth_nondim : groove depth in h/c units (positive = groove cut into surface)
    Phi, Z : meshgrid координаты (2D)
    N_g : число канавок по окружности
    w_g_nondim : ширина канавки по φ (rad)
    L_g_nondim : длина канавки по Z (nondimensional)

    Returns
    -------
    H : modified film thickness (grooves make H larger)
    """
    H = np.array(H0, dtype=float)
    cell_span = 2.0 * math.pi / N_g
    for k in range(N_g):
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi_distance(Phi, phi_c)
        mask = (np.abs(dp) <= 0.5 * w_g_nondim) & (np.abs(Z) <= 0.5 * L_g_nondim)
        H[mask] += depth_nondim
    return H


def create_H_with_herringbone_grooves(
        H0: np.ndarray,
        depth_nondim: float,
        Phi: np.ndarray, Z: np.ndarray,
        N_g: int,
        w_g_nondim: float,
        L_g_nondim: float,
        beta_deg: float,
) -> np.ndarray:
    """Добавить N_g шевронных (V-shaped) канавок.

    Каждая ячейка: две зеркальные плечи с углами +β и -β, apex на
    осевой середине (Z=0 для centered groove).

    Overlap rule: max(left, right) — no double-depth.

    Parameters
    ----------
    H0 : base film thickness (2D)
    depth_nondim : groove depth (positive)
    Phi, Z : meshgrid (2D)
    N_g : число ячеек по окружности
    w_g_nondim : ширина канавки (перпендикулярно оси плеча) (φ-rad)
    L_g_nondim : полная длина V-grooves по Z (nondimensional)
    beta_deg : угол плеча к оси Z (градусы)

    Returns
    -------
    H : modified film thickness
    """
    H = np.array(H0, dtype=float)
    beta = math.radians(float(beta_deg))
    cell_span = 2.0 * math.pi / N_g
    L_arm = L_g_nondim / 2.0

    for k in range(N_g):
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi_distance(Phi, phi_c)

        # Left arm: apex at Z=0, extends toward Z = -L_arm/2, angle +β
        Z_c_left = -L_arm / 2.0
        mask_left = _parallelogram_mask(
            dp, Z, 0.0, Z_c_left, L_arm, w_g_nondim, +beta)

        # Right arm: extends toward Z = +L_arm/2, angle -β
        Z_c_right = +L_arm / 2.0
        mask_right = _parallelogram_mask(
            dp, Z, 0.0, Z_c_right, L_arm, w_g_nondim, -beta)

        # Overlap: take max (union), not sum
        combined = mask_left | mask_right
        H[combined] += depth_nondim
    return H


# ─── Parameter converters ──────────────────────────────────────────

def gu_groove_params_nondim(D_m: float, L_m: float, c_m: float,
                             R_m: float,
                             w_g_m: float, L_g_m: float, d_g_m: float,
                             beta_deg: float, N_g: int
                             ) -> Dict[str, Any]:
    """Convert Gu 2020 dimensional groove params to nondimensional.

    Returns dict ready for create_H_with_{straight,herringbone}_grooves.
    """
    return dict(
        w_g_nondim=float(w_g_m / R_m),           # groove width in rad
        L_g_nondim=float(L_g_m / (L_m / 2.0)),   # groove length in Z-units
        depth_nondim=float(d_g_m / c_m),          # groove depth in h/c
        beta_deg=float(beta_deg),
        N_g=int(N_g),
        # Ratios for transfer
        w_g_over_D=float(w_g_m / D_m),
        L_g_over_D=float(L_g_m / D_m),
        d_g_over_c=float(d_g_m / c_m),
    )


def transfer_groove_params(source_ratios: Dict[str, float],
                            D_target: float, L_target: float,
                            c_target: float, R_target: float,
                            N_g: int,
                            beta_deg: float,
                            mode: str = "scaled_nondim",
                            ) -> Dict[str, Any]:
    """Transfer groove design to different bearing geometry.

    Modes:
      * "scaled_nondim" — preserve w_g/D, L_g/D, d_g/c
      * "same_mm" — preserve absolute mm dimensions

    N_g held fixed at 10 for both modes (ТЗ clarification 2).
    """
    if mode == "scaled_nondim":
        w_g_m = source_ratios["w_g_over_D"] * D_target
        L_g_m = source_ratios["L_g_over_D"] * D_target
        d_g_m = source_ratios["d_g_over_c"] * c_target
    elif mode == "same_mm":
        w_g_m = source_ratios["w_g_over_D"] * (source_ratios.get("D_source_m") or D_target)
        L_g_m = source_ratios["L_g_over_D"] * (source_ratios.get("D_source_m") or D_target)
        d_g_m = source_ratios["d_g_over_c"] * (source_ratios.get("c_source_m") or c_target)
    else:
        raise ValueError(f"unknown transfer mode: {mode!r}")
    return gu_groove_params_nondim(
        D_target, L_target, c_target, R_target,
        w_g_m, L_g_m, d_g_m, beta_deg, N_g)


__all__ = [
    "create_H_with_straight_grooves",
    "create_H_with_herringbone_grooves",
    "get_herringbone_cell_centers",
    "create_H_with_herringbone_grooves_subset",
    "gu_groove_params_nondim",
    "transfer_groove_params",
]


# ─── Cell center helper + subset builder ───────────────────────────

def get_herringbone_cell_centers(N_g: int) -> np.ndarray:
    """Angular centers of groove cells (rad). Shape (N_g,)."""
    cell_span = 2.0 * math.pi / int(N_g)
    return np.array([cell_span * (k + 0.5) for k in range(int(N_g))])


def create_H_with_herringbone_grooves_subset(
        H0: np.ndarray,
        depth_nondim: float,
        Phi: np.ndarray, Z: np.ndarray,
        N_g: int,
        w_g_nondim: float,
        L_g_nondim: float,
        beta_deg: float,
        active_cells: list,
) -> np.ndarray:
    """Like create_H_with_herringbone_grooves but only activates cells
    whose indices are in `active_cells`.

    active_cells: list of int indices in [0, N_g). Wrap-around is the
    caller's responsibility (pass already-wrapped indices).
    """
    H = np.array(H0, dtype=float)
    beta = math.radians(float(beta_deg))
    cell_span = 2.0 * math.pi / N_g
    L_arm = L_g_nondim / 2.0

    for k in active_cells:
        k = int(k) % int(N_g)
        phi_c = cell_span * (k + 0.5)
        dp = _wrap_phi_distance(Phi, phi_c)
        Z_c_left = -L_arm / 2.0
        mask_left = _parallelogram_mask(
            dp, Z, 0.0, Z_c_left, L_arm, w_g_nondim, +beta)
        Z_c_right = +L_arm / 2.0
        mask_right = _parallelogram_mask(
            dp, Z, 0.0, Z_c_right, L_arm, w_g_nondim, -beta)
        combined = mask_left | mask_right
        H[combined] += depth_nondim
    return H
