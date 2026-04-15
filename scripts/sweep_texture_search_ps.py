#!/usr/bin/env python3
"""Параметрический поиск текстуры с Payvar-Salant.

Прогоняет ~50 вариантов текстуры при ε=0.6, минеральное масло,
сетка 1000×400. Выводит сводную таблицу по gain_W.
"""
import sys
import os
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import (
    setup_grid, make_H, solve_and_compute,
)
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL

# ─── Фиксированные параметры ──────────────────────────────────────
N_PHI = 1000
N_Z = 400
EPS = 0.6
ETA = MINERAL_OIL["eta_pump"]
PROFILE = "smoothcap"
CAVITATION = "payvar_salant"

# ─── Варьируемые параметры ─────────────────────────────────────────
HP_VALUES = [5e-6, 10e-6, 15e-6, 20e-6, 30e-6]

ZONES = [
    (90, 180),
    (120, 200),
    (90, 270),
    (180, 360),
    (0, 360),
]

DIMPLE_SIZES = [
    (1.0e-3, 0.8e-3),    # a=1.0мм, b=0.8мм
    (2.0e-3, 1.5e-3),    # a=2.0мм, b=1.5мм
]


def compute_N_tex(zone_deg, a_dim, b_dim, R, L):
    """Максимальное число лунок без перекрытия (запас 25%)."""
    phi_s, phi_e = zone_deg
    if phi_s < phi_e:
        phi_span = np.deg2rad(phi_e - phi_s)
    else:
        phi_span = np.deg2rad((360 - phi_s) + phi_e)

    B = b_dim / R                # полуось по φ (рад)
    A = 2 * a_dim / L            # полуось по Z (безразм.)

    N_phi_tex = max(1, int(phi_span / (2.5 * 2 * B)))
    N_Z_tex = max(1, int(2.0 / (2.5 * 2 * A)))
    return N_phi_tex, N_Z_tex


def setup_texture_custom(p):
    """Расставить центры лунок с отступом от краёв зоны.

    Для зоны 0-360: отступ ≥ B от шва 0/2π.
    """
    B = p.b_dim / p.R
    A = 2 * p.a_dim / p.L

    phi_s = np.deg2rad(p.phi_start_deg)
    phi_e = np.deg2rad(p.phi_end_deg)

    if p.phi_start_deg < p.phi_end_deg:
        phi_span = phi_e - phi_s
    else:
        phi_span = (2 * np.pi - phi_s) + phi_e

    # Отступ от краёв: ≥ B (полуось), чтобы лунка не вылезала за зону
    margin = B * 1.1
    usable_span = phi_span - 2 * margin

    N_phi_t = p.N_phi_tex
    if usable_span <= 0 or N_phi_t < 1:
        return np.array([]), np.array([])

    if N_phi_t == 1:
        phi_centers = np.array([phi_s + phi_span / 2])
    else:
        phi_centers = phi_s + margin + np.linspace(0, usable_span, N_phi_t)

    phi_centers = phi_centers % (2 * np.pi)

    # Z-центры с отступом
    margin_Z = A * 1.1
    usable_Z = 2.0 - 2 * margin_Z
    N_Z_t = p.N_Z_tex
    if usable_Z <= 0 or N_Z_t < 1:
        return np.array([]), np.array([])

    if N_Z_t == 1:
        Z_centers = np.array([0.0])
    else:
        Z_centers = -1.0 + margin_Z + np.linspace(0, usable_Z, N_Z_t)

    phi_g, Z_g = np.meshgrid(phi_centers, Z_centers)
    return phi_g.ravel(), Z_g.ravel()


def make_params(hp, zone, a_dim, b_dim, N_phi_tex, N_Z_tex):
    """Создать namespace с переопределёнными параметрами."""
    p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                 for k in dir(base_params)
                                 if not k.startswith('_')})
    p.h_p = hp
    p.a_dim = a_dim
    p.b_dim = b_dim
    p.phi_start_deg = zone[0]
    p.phi_end_deg = zone[1]
    p.N_phi_tex = N_phi_tex
    p.N_Z_tex = N_Z_tex
    return p


def main():
    print("=" * 100)
    print("ПАРАМЕТРИЧЕСКИЙ ПОИСК ТЕКСТУРЫ — PAYVAR-SALANT")
    print(f"ε = {EPS}, η = {ETA} Па·с, сетка {N_PHI}×{N_Z}, профиль = {PROFILE}")
    print("=" * 100)

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)

    # --- Гладкий (один раз) ---
    p_base = types.SimpleNamespace(**{k: getattr(base_params, k)
                                      for k in dir(base_params)
                                      if not k.startswith('_')})
    H_smooth = make_H(EPS, Phi_mesh, Z_mesh, p_base, textured=False)

    print("\n  Гладкий подшипник...", end="", flush=True)
    t0 = time.time()
    _, W_s, f_s, _, _, pmax_s, _, _, _, cf_s = solve_and_compute(
        H_smooth, d_phi, d_Z, p_base.R, p_base.L, ETA, p_base.n, p_base.c,
        phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)
    dt_s = time.time() - t0
    print(f" W={W_s:.0f} Н, f={f_s:.4f}, cav={cf_s:.1%} ({dt_s:.1f} с)")

    # --- Построить варианты ---
    variants = []
    for hp in HP_VALUES:
        for zone in ZONES:
            for (a, b) in DIMPLE_SIZES:
                N_phi_tex, N_Z_tex = compute_N_tex(zone, a, b,
                                                    base_params.R, base_params.L)
                if N_phi_tex >= 1 and N_Z_tex >= 1:
                    variants.append({
                        "hp": hp, "zone": zone, "a": a, "b": b,
                        "N_phi_tex": N_phi_tex, "N_Z_tex": N_Z_tex,
                    })

    print(f"\n  Всего вариантов: {len(variants)}")
    print(f"  Оценка времени: ~{len(variants) * 2:.0f} с")
    print()

    # --- Прогнать варианты ---
    results = []
    for i, v in enumerate(variants):
        p = make_params(v["hp"], v["zone"], v["a"], v["b"],
                        v["N_phi_tex"], v["N_Z_tex"])
        phi_c, Z_c = setup_texture_custom(p)

        if len(phi_c) == 0:
            continue

        n_dimples = len(phi_c)
        A = 2 * p.a_dim / p.L
        B = p.b_dim / p.R
        H_p = p.h_p / p.c

        H0 = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
        H_tex = create_H_with_ellipsoidal_depressions(
            H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A, B,
            profile=PROFILE)

        t0 = time.time()
        _, W_t, f_t, _, _, pmax_t, _, n_out, _, cf_t = solve_and_compute(
            H_tex, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
            phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)
        dt = time.time() - t0

        gain_W = W_t / W_s if W_s > 0 else 0
        gain_f = f_t / f_s if f_s > 0 else 0
        gain_pmax = pmax_t / pmax_s if pmax_s > 0 else 0

        zone_str = f"{v['zone'][0]}-{v['zone'][1]}°"
        marker = " <<<" if gain_W > 1.05 else ""
        print(f"  [{i+1:2d}/{len(variants)}] hp={v['hp']*1e6:4.0f}мкм "
              f"zone={zone_str:>8s} a/b={v['a']*1e3:.1f}/{v['b']*1e3:.1f} "
              f"N={n_dimples:>4d} "
              f"gain_W={gain_W:.4f} gain_f={gain_f:.4f} "
              f"({dt:.1f}с){marker}")

        results.append({
            "hp_um": v["hp"] * 1e6,
            "zone": zone_str,
            "a_mm": v["a"] * 1e3,
            "b_mm": v["b"] * 1e3,
            "N_phi_tex": v["N_phi_tex"],
            "N_Z_tex": v["N_Z_tex"],
            "n_dimples": n_dimples,
            "W_tex": W_t,
            "f_tex": f_t,
            "pmax_tex": pmax_t,
            "cav_tex": cf_t,
            "gain_W": gain_W,
            "gain_f": gain_f,
            "gain_pmax": gain_pmax,
            "n_iter": n_out,
            "time": dt,
        })

    # --- Сортировка по gain_W ---
    results.sort(key=lambda x: x["gain_W"], reverse=True)

    # --- Сводная таблица ---
    print(f"\n{'=' * 110}")
    print("СВОДНАЯ ТАБЛИЦА (отсортировано по gain_W)")
    print(f"W_smooth = {W_s:.0f} Н, f_smooth = {f_s:.4f}, "
          f"pmax_smooth = {pmax_s/1e6:.1f} МПа")
    print(f"{'=' * 110}")
    print(f"{'#':>3} {'hp,мкм':>7} {'зона':>9} {'a,мм':>5} {'b,мм':>5} "
          f"{'N_лун':>6} {'gain_W':>8} {'gain_f':>8} {'gain_pm':>8} "
          f"{'cav,%':>6} {'n_iter':>7} {'time':>5}")
    print("-" * 110)

    for i, r in enumerate(results):
        marker = " <<<" if r["gain_W"] > 1.05 else ""
        print(f"{i+1:3d} {r['hp_um']:7.0f} {r['zone']:>9s} "
              f"{r['a_mm']:5.1f} {r['b_mm']:5.1f} "
              f"{r['n_dimples']:6d} {r['gain_W']:8.4f} {r['gain_f']:8.4f} "
              f"{r['gain_pmax']:8.4f} {r['cav_tex']*100:6.1f} "
              f"{r['n_iter']:7d} {r['time']:5.1f}{marker}")

    # --- Кандидаты ---
    candidates = [r for r in results
                  if r["gain_W"] > 1.05 and r["gain_f"] < 0.95
                  and r["gain_pmax"] < 3.0]

    print(f"\n{'=' * 110}")
    if candidates:
        print(f"КАНДИДАТЫ НА ТОНКУЮ СЕТКУ ({len(candidates)} шт):")
        for r in candidates[:5]:
            print(f"  hp={r['hp_um']:.0f}мкм, zone={r['zone']}, "
                  f"a/b={r['a_mm']:.1f}/{r['b_mm']:.1f} — "
                  f"gain_W={r['gain_W']:.4f}, gain_f={r['gain_f']:.4f}")
    else:
        best = results[0] if results else None
        if best:
            print(f"Ни один вариант не дал gain_W > 1.05.")
            print(f"Лучший: hp={best['hp_um']:.0f}мкм, zone={best['zone']}, "
                  f"gain_W={best['gain_W']:.4f}")
        else:
            print("Нет результатов.")
    print(f"{'=' * 110}")

    # --- CSV ---
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump_ps")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "texture_search.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hp_um", "zone", "a_mm", "b_mm", "N_phi_tex", "N_Z_tex",
                     "n_dimples", "W_tex", "gain_W", "f_tex", "gain_f",
                     "pmax_tex_MPa", "gain_pmax", "cav_tex", "n_iter", "time_s"])
        for r in results:
            w.writerow([
                f"{r['hp_um']:.0f}", r["zone"],
                f"{r['a_mm']:.1f}", f"{r['b_mm']:.1f}",
                r["N_phi_tex"], r["N_Z_tex"], r["n_dimples"],
                f"{r['W_tex']:.1f}", f"{r['gain_W']:.4f}",
                f"{r['f_tex']:.5f}", f"{r['gain_f']:.4f}",
                f"{r['pmax_tex']/1e6:.2f}", f"{r['gain_pmax']:.4f}",
                f"{r['cav_tex']:.3f}", r["n_iter"], f"{r['time']:.1f}",
            ])
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
