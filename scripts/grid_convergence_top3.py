#!/usr/bin/env python3
"""Сеточная сходимость для топ-3 кандидатов текстуры.

Проверка: сохраняется ли gain_W > 1 на тонких сетках.
Без PV, ε=0.3, минеральное масло.
"""
import sys
import os
import time
import csv
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import setup_grid, make_H, solve_and_compute
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"
EPS = 0.3
ETA = MINERAL_OIL["eta_pump"]

GRIDS = [
    (800, 200),
    (1600, 400),
    (2400, 600),
    (3200, 800),
]

CANDIDATES = [
    {"label": "1", "hp": 30e-6, "a": 1.5e-3, "b": 1.2e-3,
     "zone": (0, 90), "sf": 1.5},
    {"label": "2", "hp": 20e-6, "a": 1.5e-3, "b": 1.2e-3,
     "zone": (0, 90), "sf": 1.5},
    {"label": "3", "hp": 15e-6, "a": 2.0e-3, "b": 1.5e-3,
     "zone": (0, 90), "sf": 1.5},
]


def compute_N_tex(zone, a, b, R, L, sf):
    phi_span = np.deg2rad(zone[1] - zone[0])
    B = b / R
    A = 2 * a / L
    return max(1, int(phi_span / (sf * 2 * B))), max(1, int(2.0 / (sf * 2 * A)))


def setup_texture_custom(p):
    B = p.b_dim / p.R
    A = 2 * p.a_dim / p.L
    phi_s = np.deg2rad(p.phi_start_deg)
    phi_e = np.deg2rad(p.phi_end_deg)
    phi_span = phi_e - phi_s
    margin = B * 1.1
    usable = phi_span - 2 * margin
    if usable <= 0:
        return np.array([]), np.array([])
    if p.N_phi_tex == 1:
        phi_c = np.array([phi_s + phi_span / 2])
    else:
        phi_c = phi_s + margin + np.linspace(0, usable, p.N_phi_tex)
    phi_c = phi_c % (2 * np.pi)
    margin_Z = A * 1.1
    usable_Z = 2.0 - 2 * margin_Z
    if usable_Z <= 0:
        return np.array([]), np.array([])
    if p.N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_c = -1.0 + margin_Z + np.linspace(0, usable_Z, p.N_Z_tex)
    pg, zg = np.meshgrid(phi_c, Z_c)
    return pg.ravel(), zg.ravel()


def solve_one(H, d_phi, d_Z, phi_1D, Z_1D, Phi_mesh):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        _, W, f, _, _, pmax, _, n_out, _, cav = solve_and_compute(
            H, d_phi, d_Z, base_params.R, base_params.L, ETA,
            base_params.n, base_params.c,
            phi_1D, Z_1D, Phi_mesh,
            cavitation=CAVITATION, alpha_pv=None)
        dt = time.time() - t0
        wup = not any("warmup" in str(w.message) for w in caught)
    return W, wup, dt


def main():
    print("=" * 85)
    print("СЕТОЧНАЯ СХОДИМОСТЬ ТОП-3 КАНДИДАТОВ")
    print(f"ε={EPS}, минеральное масло, без PV")
    print("=" * 85)

    results = []

    for cand in CANDIDATES:
        hp = cand["hp"]
        a, b = cand["a"], cand["b"]
        zone = cand["zone"]
        sf = cand["sf"]

        print(f"\n  Кандидат {cand['label']}: hp={hp*1e6:.0f}мкм, "
              f"a/b={a*1e3:.1f}/{b*1e3:.1f}, zone={zone[0]}-{zone[1]}°")
        print(f"  {'сетка':>10s} {'W_smooth':>9s} {'W_tex':>9s} "
              f"{'gain_W':>8s} {'Δ':>8s} {'warmup':>7s} {'time':>6s}")
        print("  " + "-" * 65)

        prev_gw = None
        for N_phi, N_Z in GRIDS:
            phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)

            p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                         for k in dir(base_params)
                                         if not k.startswith('_')})
            p.a_dim, p.b_dim, p.h_p = a, b, hp
            p.phi_start_deg, p.phi_end_deg = zone
            Nt_phi, Nt_Z = compute_N_tex(zone, a, b, p.R, p.L, sf)
            p.N_phi_tex, p.N_Z_tex = Nt_phi, Nt_Z

            phi_c, Z_c = setup_texture_custom(p)
            n_dimples = len(phi_c)

            # Smooth
            H_s = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
            W_s, wup_s, dt_s = solve_one(H_s, d_phi, d_Z,
                                          phi_1D, Z_1D, Phi_mesh)

            # Textured
            H0 = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
            A_nd = 2 * a / p.L
            B_nd = b / p.R
            H_p = hp / p.c
            H_tex = create_H_with_ellipsoidal_depressions(
                H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A_nd, B_nd,
                profile=PROFILE)
            W_t, wup_t, dt_t = solve_one(H_tex, d_phi, d_Z,
                                          phi_1D, Z_1D, Phi_mesh)

            gw = W_t / W_s if W_s > 0 else 0
            delta = gw - prev_gw if prev_gw is not None else 0
            delta_str = f"{delta:+8.4f}" if prev_gw is not None else "    —   "
            prev_gw = gw

            wup_str = "OK" if (wup_s and wup_t) else "[!]"
            dt_total = dt_s + dt_t
            marker = " <<<" if gw > 1.0 else ""

            print(f"  {N_phi:>4d}×{N_Z:<4d} {W_s:9.0f} {W_t:9.0f} "
                  f"{gw:8.4f} {delta_str} {wup_str:>7s} {dt_total:5.1f}с"
                  f"{marker}")

            results.append({
                "cand": cand["label"],
                "hp_um": hp * 1e6,
                "a_mm": a * 1e3, "b_mm": b * 1e3,
                "N_phi": N_phi, "N_Z": N_Z,
                "W_smooth": W_s, "W_tex": W_t,
                "gain_W": gw, "delta": delta,
                "warmup_ok": wup_s and wup_t,
                "time": dt_total,
            })

    # CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "pump_pv_ps")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "grid_convergence_top3.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand", "hp_um", "a_mm", "b_mm", "N_phi", "N_Z",
                     "W_smooth", "W_tex", "gain_W", "delta",
                     "warmup_ok", "time_s"])
        for r in results:
            w.writerow([r["cand"], f"{r['hp_um']:.0f}",
                        f"{r['a_mm']:.1f}", f"{r['b_mm']:.1f}",
                        r["N_phi"], r["N_Z"],
                        f"{r['W_smooth']:.1f}", f"{r['W_tex']:.1f}",
                        f"{r['gain_W']:.4f}", f"{r['delta']:.4f}",
                        r["warmup_ok"], f"{r['time']:.1f}"])
    print(f"\n  CSV: {csv_path}")

    # Итог
    print(f"\n{'=' * 85}")
    finest = [r for r in results if r["N_phi"] == GRIDS[-1][0]]
    any_positive = any(r["gain_W"] > 1.0 for r in finest)
    if any_positive:
        print("gain_W > 1.0 ПОДТВЕРЖДЁН на тонкой сетке:")
        for r in finest:
            if r["gain_W"] > 1.0:
                print(f"  Кандидат {r['cand']}: gain_W = {r['gain_W']:.4f}")
    else:
        print("gain_W ≤ 1.0 на тонкой сетке — эффект не подтверждён.")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
