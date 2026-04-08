#!/usr/bin/env python3
"""Параметрический поиск текстуры насоса.

Автоматически прогоняет набор вариантов (зона, глубина, число лунок),
печатает одну сводную таблицу, отсортированную по gain_W.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
    DEFAULT_CLOSURE, DEFAULT_CAVITATION,
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL
import types

# ─── Настройки sweep ───────────────────────────────────────────────
N_PHI = 3000
N_Z = 200
EPS_TARGET = 0.60
PROFILE = "sqrt"

VARIANTS = [
    # (имя, phi_start, phi_end, h_p_um, nphi_tex, nz_tex)
    ("1 base 90-270",       90,  270, 10, 8, 9),
    ("2 diverg 0-90",        0,   90, 10, 8, 9),
    ("3 wide 270-90",      270,   90, 10, 8, 9),
    ("4 cavit 330-30",     330,   30, 10, 8, 9),
    ("5 converg 180-270",  180,  270, 10, 8, 9),
    ("6 diverg h5",          0,   90,  5, 8, 9),
    ("7 diverg h20",         0,   90, 20, 8, 9),
    ("8 diverg h3",          0,   90,  3, 8, 9),
    ("9 diverg Nphi12",      0,   90, 10, 12, 9),
    ("10 diverg Nphi4",      0,   90, 10, 4, 9),
]


def run_single(phi_start_deg, phi_end_deg, h_p_um, nphi_tex, nz_tex):
    """Прогнать 1 вариант: гладкий + текстурированный при ε = EPS_TARGET.

    Returns
    -------
    dict с W_smooth, W_tex, gain_W, f_smooth, f_tex, gain_f,
         pmax_smooth, pmax_tex, gain_pmax
    """
    # Параметры подшипника с переопределениями
    p = types.SimpleNamespace(**{k: getattr(params, k)
                                 for k in dir(params) if not k.startswith('_')})
    p.h_p = h_p_um * 1e-6
    p.phi_start_deg = phi_start_deg
    p.phi_end_deg = phi_end_deg
    p.N_phi_tex = nphi_tex
    p.N_Z_tex = nz_tex

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)
    phi_c, Z_c = setup_texture(p)

    eta = MINERAL_OIL["eta_pump"]
    eps = EPS_TARGET

    # --- Гладкий ---
    H_s = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
    _, W_s, f_s, _, _, pmax_s, _, _ = solve_and_compute(
        H_s, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh)

    # --- Текстурированный ---
    H_t = make_H(eps, Phi_mesh, Z_mesh, p, textured=True,
                 phi_c_flat=phi_c, Z_c_flat=Z_c, profile=PROFILE)
    H_smooth = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
    tex_params = {
        "phi_c": phi_c, "Z_c": Z_c,
        "A": 2 * p.a_dim / p.L,
        "B": p.b_dim / p.R,
        "H_p": p.h_p / p.c,
        "profile": PROFILE,
    }
    _, W_t, f_t, _, _, pmax_t, _, _ = solve_and_compute(
        H_t, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh,
        subcell_quad=True, H_smooth=H_smooth, texture_params=tex_params)

    def g(a, b):
        return a / b if b > 0 else 0.0

    return {
        "W_smooth": W_s, "W_tex": W_t, "gain_W": g(W_t, W_s),
        "f_smooth": f_s, "f_tex": f_t, "gain_f": g(f_t, f_s),
        "pmax_smooth": pmax_s, "pmax_tex": pmax_t, "gain_pmax": g(pmax_t, pmax_s),
    }


def main():
    print("=" * 80)
    print("ПАРАМЕТРИЧЕСКИЙ ПОИСК ТЕКСТУРЫ НАСОСА")
    print(f"Сетка: {N_PHI}×{N_Z}, ε = {EPS_TARGET}, профиль = {PROFILE}, без PV")
    print(f"Масло: минеральное (η = {MINERAL_OIL['eta_pump']} Па·с)")
    print("=" * 80)

    results = []
    t0_total = time.time()

    for i, (name, ps, pe, hp, npt, nzt) in enumerate(VARIANTS):
        t0 = time.time()
        sys.stdout.write(f"  [{i+1:2d}/{len(VARIANTS)}] {name:25s} ... ")
        sys.stdout.flush()
        r = run_single(ps, pe, hp, npt, nzt)
        dt = time.time() - t0
        print(f"gain_W={r['gain_W']:.3f}  ({dt:.1f} с)")
        results.append((name, ps, pe, hp, npt, nzt, r))

    dt_total = time.time() - t0_total
    print(f"\nОбщее время: {dt_total:.0f} с")

    # Сортировка по gain_W (убывание)
    results.sort(key=lambda x: x[6]["gain_W"], reverse=True)

    # Сводная таблица
    print(f"\n{'=' * 100}")
    print(f"СВОДНАЯ ТАБЛИЦА (отсортировано по gain_W)")
    print(f"{'=' * 100}")
    print(f"{'#':<3} {'Вариант':<25} {'Зона':>10} {'h_p':>5} {'Nphi':>5} "
          f"{'W_smooth':>9} {'W_tex':>9} {'gain_W':>7} "
          f"{'gain_f':>7} {'gain_pmax':>9}")
    print("-" * 100)

    for name, ps, pe, hp, npt, nzt, r in results:
        zone = f"{ps}°-{pe}°"
        marker = " <<<" if r["gain_W"] > 1.05 else ""
        print(f"    {'':3s}{name:<25} {zone:>10} {hp:>5} {npt:>5} "
              f"{r['W_smooth']:>9.0f} {r['W_tex']:>9.0f} {r['gain_W']:>7.3f} "
              f"{r['gain_f']:>7.3f} {r['gain_pmax']:>7.3f}{marker}")

    # Лучший вариант
    best = results[0]
    print(f"\nЛучший: {best[0]} — gain_W = {best[6]['gain_W']:.3f}")
    if best[6]["gain_W"] > 1.05:
        print("  >>> Текстура помогает! Кандидат для дальнейшего исследования.")
    elif best[6]["gain_W"] > 1.0:
        print("  Текстура даёт слабый положительный эффект.")
    else:
        print("  Текстура не помогает ни в одном варианте.")

    # Сохранить таблицу
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "texture_sweep.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Параметрический поиск текстуры насоса\n")
        f.write(f"Сетка: {N_PHI}×{N_Z}, ε = {EPS_TARGET}, "
                f"профиль = {PROFILE}, без PV\n")
        f.write(f"Время: {dt_total:.0f} с\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Вариант':<25} {'Зона':>10} {'h_p':>5} {'Nphi':>5} "
                f"{'W_smooth':>9} {'W_tex':>9} {'gain_W':>7} "
                f"{'gain_f':>7} {'gain_pmax':>9}\n")
        f.write("-" * 100 + "\n")
        for name, ps, pe, hp, npt, nzt, r in results:
            zone = f"{ps}°-{pe}°"
            f.write(f"{name:<25} {zone:>10} {hp:>5} {npt:>5} "
                    f"{r['W_smooth']:>9.0f} {r['W_tex']:>9.0f} "
                    f"{r['gain_W']:>7.3f} {r['gain_f']:>7.3f} "
                    f"{r['gain_pmax']:>7.3f}\n")
    print(f"\nТаблица сохранена: {out_path}")


if __name__ == "__main__":
    main()
