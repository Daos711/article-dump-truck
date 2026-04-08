#!/usr/bin/env python3
"""Изолированный тест grid convergence + сравнение omega=1.5 vs auto.

Гладкий подшипник, без текстуры, без PV, ε=0.6, минеральное масло.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from reynolds_solver import solve_reynolds
from models.bearing_model import setup_grid, make_H
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

eta = MINERAL_OIL["eta_pump"]
eps = 0.6
omega_shaft = 2 * np.pi * params.n / 60.0
p_scale = 6.0 * eta * omega_shaft * (params.R / params.c) ** 2

GRIDS = [(800, 200), (2000, 200), (5000, 400), (8000, 1000)]


def compute_W(N_phi, N_Z, omega_sor=None):
    """Решить Reynolds и вернуть W, n_iter, converged, delta."""
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = setup_grid(N_phi, N_Z)
    H = make_H(eps, Phi, Zm, params, textured=False)

    kw = dict(
        closure="laminar",
        cavitation="half_sommerfeld",
        return_converged=True,
    )
    if omega_sor is not None:
        kw["omega"] = omega_sor

    P, delta, n_iter, converged = solve_reynolds(
        H, d_phi, d_Z, params.R, params.L, **kw)

    P_dim = P * p_scale
    Fx = -np.trapezoid(np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
                       Z_1D) * params.R * params.L / 2
    Fy = -np.trapezoid(np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
                       Z_1D) * params.R * params.L / 2
    W = np.sqrt(Fx**2 + Fy**2)
    return W, n_iter, converged, delta


def main():
    # === Шаг 2: Grid convergence (omega=auto) ===
    print("=" * 70)
    print("ШАГ 2: Grid convergence (omega=auto, гладкий, без PV)")
    print(f"ε = {eps}, η = {eta} Па·с, R = {params.R*1e3:.0f} мм")
    print("=" * 70)

    print(f"\n{'Сетка':>12} {'W_smooth':>10} {'n_iter':>8} {'conv':>6} {'delta':>10}")
    print("-" * 52)

    W_prev = None
    for N_phi, N_Z in GRIDS:
        W, n_iter, conv, delta = compute_W(N_phi, N_Z)
        diff = ""
        if W_prev is not None and W_prev > 0:
            pct = abs(W - W_prev) / W_prev * 100
            diff = f"  Δ={pct:.1f}%"
        print(f"{N_phi}x{N_Z:>4}: W={W:>9.0f}  iter={n_iter:>6}  "
              f"conv={str(conv):>5}  delta={delta:.2e}{diff}")
        W_prev = W

    # === Шаг 4: omega=1.5 vs auto (на 5000×200) ===
    print(f"\n{'=' * 70}")
    print("ШАГ 4: omega=1.5 vs auto (5000×200)")
    print("=" * 70)

    N_phi_cmp, N_Z_cmp = 5000, 200

    W_old, n_old, conv_old, d_old = compute_W(N_phi_cmp, N_Z_cmp, omega_sor=1.5)
    W_new, n_new, conv_new, d_new = compute_W(N_phi_cmp, N_Z_cmp, omega_sor=None)

    print(f"\n{'':>15} {'W':>10} {'n_iter':>8} {'conv':>6} {'delta':>10}")
    print("-" * 55)
    print(f"{'omega=1.5':>15}: W={W_old:>9.0f}  iter={n_old:>6}  "
          f"conv={str(conv_old):>5}  delta={d_old:.2e}")
    print(f"{'omega=auto':>15}: W={W_new:>9.0f}  iter={n_new:>6}  "
          f"conv={str(conv_new):>5}  delta={d_new:.2e}")

    if conv_old and conv_new:
        pct = abs(W_old - W_new) / W_new * 100
        print(f"\nОба сошлись. ΔW = {pct:.1f}%")
    elif not conv_old:
        print(f"\nomega=1.5 НЕ СОШЁЛСЯ → ложная сходимость подтверждена")
    else:
        print(f"\nomega=auto не сошёлся — проблема в солвере")


if __name__ == "__main__":
    main()
