#!/usr/bin/env python3
"""Параметрический поиск МИКРО-текстуры насоса.

Малые лунки (0.3–1.0 мм), много штук, зона 0°–90° (дивергентная).
Сетка 5000×200 для разрешения мелких лунок.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import (
    setup_grid, make_H, solve_and_compute,
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

# ─── Настройки ─────────────────────────────────────────────────────
N_PHI = 5000
N_Z = 200
EPS_TARGET = 0.60
PROFILE = "sqrt"

# Варианты: (имя, phi_start°, phi_end°, a_mm, b_mm, h_p_um, nphi_tex, nz_tex)
VARIANTS = [
    # --- Размер b=1.0мм (диам. 2мм), ~45 узлов/лунку ---
    ("1 b1.0 h5  0-90",     0,  90,  1.0, 1.0,   5,  18, 18),
    ("2 b1.0 h10 0-90",     0,  90,  1.0, 1.0,  10,  18, 18),
    ("3 b1.0 h3  0-90",     0,  90,  1.0, 1.0,   3,  18, 18),
    ("4 b1.0 h5  330-30",   330, 30,  1.0, 1.0,   5,  10, 18),

    # --- Размер b=0.5мм (диам. 1мм), ~23 узла/лунку ---
    ("5 b0.5 h5  0-90",     0,  90,  0.5, 0.5,   5,  36, 37),
    ("6 b0.5 h10 0-90",     0,  90,  0.5, 0.5,  10,  36, 37),
    ("7 b0.5 h3  0-90",     0,  90,  0.5, 0.5,   3,  36, 37),
    ("8 b0.5 h5  330-30",   330, 30,  0.5, 0.5,   5,  18, 37),

    # --- Размер b=0.3мм (диам. 0.6мм), ~14 узлов/лунку ---
    ("9  b0.3 h5  0-90",    0,  90,  0.3, 0.3,   5,  61, 62),
    ("10 b0.3 h3  0-90",    0,  90,  0.3, 0.3,   3,  61, 62),

    # --- Зона 90-180 (дивергентная верхняя) ---
    ("11 b1.0 h5  90-180",  90, 180,  1.0, 1.0,   5,  18, 18),
    ("12 b0.5 h5  90-180",  90, 180,  0.5, 0.5,   5,  36, 37),

    # --- Зона 0-180 (вся дивергентная) ---
    ("13 b1.0 h5  0-180",   0,  180,  1.0, 1.0,   5,  36, 18),
    ("14 b0.5 h5  0-180",   0,  180,  0.5, 0.5,   5,  72, 37),

    # --- Глубокие микро ---
    ("15 b0.5 h15 0-90",    0,  90,  0.5, 0.5,  15,  36, 37),
]


def setup_texture_custom(p):
    """Расставить центры лунок с учётом зоны через 0° и произвольных размеров."""
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R

    phi_s = np.deg2rad(p.phi_start_deg)
    phi_e = np.deg2rad(p.phi_end_deg)

    # Зона через 0° (например 330°→30°)
    if phi_s > phi_e:
        phi_span = (2 * np.pi - phi_s) + phi_e
    else:
        phi_span = phi_e - phi_s

    N_phi_t = p.N_phi_tex
    N_Z_t = p.N_Z_tex

    # Центры по φ
    if N_phi_t == 1:
        phi_centers = np.array([phi_s + phi_span / 2])
    else:
        total_dimple_span = N_phi_t * 2 * B
        if total_dimple_span > phi_span:
            # Слишком много лунок — уменьшим
            N_phi_t = max(1, int(phi_span / (2.5 * B)))
            total_dimple_span = N_phi_t * 2 * B
        gap = (phi_span - total_dimple_span) / max(N_phi_t - 1, 1)
        step = 2 * B + gap
        phi_centers = phi_s + B + step * np.arange(N_phi_t)

    # Нормализация по 2π
    phi_centers = phi_centers % (2 * np.pi)

    # Центры по Z
    if N_Z_t == 1:
        Z_centers = np.array([0.0])
    else:
        total_Z_span = N_Z_t * 2 * A
        if total_Z_span > 2.0:
            N_Z_t = max(1, int(2.0 / (2.5 * A)))
            total_Z_span = N_Z_t * 2 * A
        gap_Z = (2.0 - total_Z_span) / max(N_Z_t - 1, 1)
        step_Z = 2 * A + gap_Z
        Z_centers = -1.0 + A + step_Z * np.arange(N_Z_t)

    phi_c_grid, Z_c_grid = np.meshgrid(phi_centers, Z_centers)
    return phi_c_grid.flatten(), Z_c_grid.flatten()


def run_single(phi_start_deg, phi_end_deg, a_mm, b_mm, h_p_um, nphi_tex, nz_tex):
    """Прогнать 1 вариант."""
    p = types.SimpleNamespace(**{k: getattr(params, k)
                                 for k in dir(params) if not k.startswith('_')})
    p.h_p = h_p_um * 1e-6
    p.a_dim = a_mm * 1e-3
    p.b_dim = b_mm * 1e-3
    p.phi_start_deg = phi_start_deg
    p.phi_end_deg = phi_end_deg
    p.N_phi_tex = nphi_tex
    p.N_Z_tex = nz_tex

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)
    phi_c, Z_c = setup_texture_custom(p)

    eta = MINERAL_OIL["eta_pump"]
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c

    # Гладкий
    H_s = make_H(EPS_TARGET, Phi_mesh, Z_mesh, p, textured=False)
    _, W_s, f_s, _, _, pmax_s, _, _ = solve_and_compute(
        H_s, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh)

    # Текстурированный
    H0 = make_H(EPS_TARGET, Phi_mesh, Z_mesh, p, textured=False)
    H_t = create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A, B, profile=PROFILE)

    _, W_t, f_t, _, _, pmax_t, _, _ = solve_and_compute(
        H_t, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh)

    g = lambda a, b: a / b if b > 0 else 0.0
    return {
        "W_s": W_s, "W_t": W_t, "gain_W": g(W_t, W_s),
        "f_s": f_s, "f_t": f_t, "gain_f": g(f_t, f_s),
        "pmax_s": pmax_s, "pmax_t": pmax_t, "gain_pmax": g(pmax_t, pmax_s),
        "n_dimples": len(phi_c),
    }


def main():
    print("=" * 90)
    print("ПАРАМЕТРИЧЕСКИЙ ПОИСК МИКРО-ТЕКСТУРЫ")
    print(f"Сетка: {N_PHI}×{N_Z}, ε = {EPS_TARGET}, профиль = {PROFILE}, без PV")
    print("=" * 90)

    results = []
    t0_all = time.time()

    for i, (name, ps, pe, a, b, hp, npt, nzt) in enumerate(VARIANTS):
        t0 = time.time()
        sys.stdout.write(f"  [{i+1:2d}/{len(VARIANTS)}] {name:25s} ... ")
        sys.stdout.flush()
        r = run_single(ps, pe, a, b, hp, npt, nzt)
        dt = time.time() - t0
        marker = " <<<" if r["gain_W"] > 1.02 else ""
        print(f"gain_W={r['gain_W']:.4f}  N_dimples={r['n_dimples']:>4}  ({dt:.1f} с){marker}")
        results.append((name, ps, pe, a, b, hp, npt, nzt, r))

    dt_all = time.time() - t0_all
    print(f"\nОбщее время: {dt_all:.0f} с")

    # Сортировка
    results.sort(key=lambda x: x[8]["gain_W"], reverse=True)

    print(f"\n{'=' * 110}")
    print("СВОДНАЯ ТАБЛИЦА (отсортировано по gain_W)")
    print(f"{'=' * 110}")
    print(f"{'Вариант':<25} {'Зона':>10} {'a,b мм':>7} {'h_p':>4} "
          f"{'Nлун':>5} {'W_smooth':>9} {'W_tex':>9} {'gain_W':>7} "
          f"{'gain_f':>7} {'gain_pm':>7}")
    print("-" * 110)

    for name, ps, pe, a, b, hp, npt, nzt, r in results:
        zone = f"{ps}-{pe}°"
        ab = f"{a:.1f}/{b:.1f}"
        marker = " <<<" if r["gain_W"] > 1.02 else ""
        print(f"{name:<25} {zone:>10} {ab:>7} {hp:>4} "
              f"{r['n_dimples']:>5} {r['W_s']:>9.0f} {r['W_t']:>9.0f} "
              f"{r['gain_W']:>7.4f} {r['gain_f']:>7.4f} {r['gain_pmax']:>7.4f}{marker}")

    best = results[0]
    print(f"\nЛучший: {best[0]} — gain_W = {best[8]['gain_W']:.4f}")
    if best[8]["gain_W"] > 1.05:
        print("  >>> Текстура помогает (>5%)!")
    elif best[8]["gain_W"] > 1.01:
        print("  Слабый положительный эффект.")
    else:
        print("  Текстура не помогает.")

    # Сохранить
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "micro_texture_sweep.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Микро-текстура: {N_PHI}×{N_Z}, ε={EPS_TARGET}, {PROFILE}, без PV\n")
        f.write(f"Время: {dt_all:.0f} с\n\n")
        for name, ps, pe, a, b, hp, npt, nzt, r in results:
            f.write(f"{name:<25} {ps}-{pe}° a/b={a}/{b}мм h_p={hp}мкм "
                    f"N={r['n_dimples']} gain_W={r['gain_W']:.4f}\n")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
