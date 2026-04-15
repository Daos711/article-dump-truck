#!/usr/bin/env python3
"""Микролунки 100-300 мкм с Payvar-Salant на тонкой сетке.

Stage 0 (быстрый screening): zone=0-90, 200/300 мкм, hp=1-7 мкм,
ε=0.3/0.6, sf=1.5/2.0 → ~40 расчётов.

Stage 1 (расширенный, только если Stage 0 показал gain_W > 0.995):
все зоны и ε.
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
ETA = MINERAL_OIL["eta_pump"]

# Тонкая сетка для микролунок
N_PHI = 5000
N_Z = 1200

# Stage 0: быстрый screening
S0_DIMPLES = [
    (200e-6, 160e-6),  # 200×160 мкм
    (300e-6, 240e-6),  # 300×240 мкм
]
S0_HP = [1e-6, 2e-6, 3e-6, 5e-6, 7e-6]
S0_ZONES = [(0, 90)]
S0_SF = [2.0, 3.0, 5.0]
S0_EPS = [0.3, 0.6]

MAX_DIMPLES = 2000

# Stage 1: расширенный
S1_ZONES = [(0, 90), (0, 180), (90, 270)]
S1_EPS = [0.3, 0.5, 0.6, 0.7]
S1_SF = [2.0, 3.0, 5.0]
S1_HP = [1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 10e-6]

MIN_NODES = 6


def make_p():
    return types.SimpleNamespace(**{k: getattr(base_params, k)
                                    for k in dir(base_params)
                                    if not k.startswith('_')})


def compute_N_tex(zone, a, b, R, L, sf):
    phi_span = np.deg2rad(zone[1] - zone[0])
    B = b / R
    A = 2 * a / L
    return max(1, int(phi_span / (sf * 2 * B))), max(1, int(2.0 / (sf * 2 * A)))


def setup_tex(p):
    B = p.b_dim / p.R
    A = 2 * p.a_dim / p.L
    phi_s = np.deg2rad(p.phi_start_deg)
    phi_e = np.deg2rad(p.phi_end_deg)
    if p.phi_start_deg < p.phi_end_deg:
        phi_span = phi_e - phi_s
    else:
        phi_span = (2 * np.pi - phi_s) + phi_e
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
    return W, f, pmax, cav, wup, dt


def run_stage(dimples, hp_list, zones, sfs, eps_list, tag, out_dir):
    """Универсальный прогон."""
    print(f"\n{'=' * 90}")
    print(f"МИКРОЛУНКИ — {tag} (сетка {N_PHI}×{N_Z})")
    print(f"{'=' * 90}")

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)
    p_base = make_p()

    # Кэш smooth
    smooth_cache = {}
    for eps in eps_list:
        H_s = make_H(eps, Phi_mesh, Z_mesh, p_base, textured=False)
        W_s, f_s, _, _, wup_s, dt_s = solve_one(
            H_s, d_phi, d_Z, phi_1D, Z_1D, Phi_mesh)
        smooth_cache[eps] = {"W": W_s, "f": f_s}
        print(f"  Smooth ε={eps:.1f}: W={W_s:.0f} ({dt_s:.1f}с)")

    # Генерация + прогон
    results = []
    total = 0
    skipped = 0

    variants = []
    for (a, b) in dimples:
        for hp in hp_list:
            for zone in zones:
                for sf in sfs:
                    Nt_phi, Nt_Z = compute_N_tex(zone, a, b,
                                                  p_base.R, p_base.L, sf)
                    B = b / p_base.R
                    A = 2 * a / p_base.L
                    npd_phi = 2 * B / d_phi
                    npd_Z = 2 * A / d_Z
                    if npd_phi < MIN_NODES or npd_Z < MIN_NODES:
                        skipped += len(eps_list)
                        continue
                    if Nt_phi < 1 or Nt_Z < 1:
                        skipped += len(eps_list)
                        continue
                    n_total = Nt_phi * Nt_Z
                    if n_total > MAX_DIMPLES:
                        skipped += len(eps_list)
                        continue
                    variants.append((a, b, hp, zone, sf, Nt_phi, Nt_Z,
                                     npd_phi, npd_Z))

    total_calcs = len(variants) * len(eps_list)
    print(f"\n  Валидных комбинаций: {len(variants)}, "
          f"расчётов: {total_calcs}, пропущено: {skipped}")

    t0_all = time.time()
    done = 0

    for iv, (a, b, hp, zone, sf, Nt_phi, Nt_Z, npd_phi, npd_Z) in enumerate(variants):
        p = make_p()
        p.a_dim, p.b_dim, p.h_p = a, b, hp
        p.phi_start_deg, p.phi_end_deg = zone
        p.N_phi_tex, p.N_Z_tex = Nt_phi, Nt_Z

        phi_c, Z_c = setup_tex(p)
        if len(phi_c) == 0:
            continue
        n_dimples = len(phi_c)
        A_nd = 2 * a / p.L
        B_nd = b / p.R
        H_p = hp / p.c

        for eps in eps_list:
            print(f"  [{done+1}/{total_calcs}] "
                  f"hp={hp*1e6:.0f} a/b={a*1e6:.0f}/{b*1e6:.0f} "
                  f"zone={zone[0]}-{zone[1]} sf={sf:.1f} ε={eps:.1f} "
                  f"N={n_dimples} ...", end="", flush=True)

            H0 = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
            H_tex = create_H_with_ellipsoidal_depressions(
                H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A_nd, B_nd,
                profile=PROFILE)

            W_t, f_t, pmax_t, cav_t, wup, dt = solve_one(
                H_tex, d_phi, d_Z, phi_1D, Z_1D, Phi_mesh)

            sc = smooth_cache[eps]
            gw = W_t / sc["W"] if sc["W"] > 0 else 0
            gf = f_t / sc["f"] if sc["f"] > 0 else 0

            results.append({
                "hp_um": hp * 1e6, "a_um": a * 1e6, "b_um": b * 1e6,
                "zone": f"{zone[0]}-{zone[1]}",
                "sf": sf, "n_dimples": n_dimples, "eps": eps,
                "W_tex": W_t, "gain_W": gw, "gain_f": gf,
                "cav": cav_t, "warmup_ok": wup,
                "nodes_phi": npd_phi, "nodes_Z": npd_Z,
                "time": dt,
            })
            done += 1

            marker = " <<<" if gw > 1.0 else (" ~" if gw > 0.995 else "")
            wup_s = "" if wup else " [!]"
            print(f" gW={gw:.4f} ({dt:.1f}с){marker}{wup_s}")

    dt_all = time.time() - t0_all
    print(f"\n  Время: {dt_all:.0f}с ({dt_all/60:.1f} мин)")

    # Сохранить CSV
    csv_path = os.path.join(out_dir, f"micro_{tag.lower().replace(' ', '_')}.csv")
    fieldnames = ["hp_um", "a_um", "b_um", "zone", "sf", "n_dimples",
                  "eps", "W_tex", "gain_W", "gain_f", "cav",
                  "warmup_ok", "nodes_phi", "nodes_Z", "time"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            for k in ["gain_W", "gain_f", "cav"]:
                row[k] = f"{r[k]:.4f}"
            for k in ["W_tex", "time"]:
                row[k] = f"{r[k]:.1f}"
            for k in ["nodes_phi", "nodes_Z"]:
                row[k] = f"{r[k]:.1f}"
            w.writerow(row)
    print(f"  CSV: {csv_path}")

    # Топ-10
    results.sort(key=lambda x: x["gain_W"], reverse=True)
    print(f"\n  ТОП-10:")
    print(f"  {'#':>3} {'hp':>4} {'a/b мкм':>10} {'zone':>8} {'sf':>4} "
          f"{'ε':>4} {'N':>5} {'gain_W':>8} {'gf':>7} {'wup':>4}")
    print("  " + "-" * 70)
    for i, r in enumerate(results[:10]):
        wup = "OK" if r["warmup_ok"] else "[!]"
        m = " <<<" if r["gain_W"] > 1.0 else ""
        print(f"  {i+1:3d} {r['hp_um']:4.0f} {r['a_um']:.0f}/{r['b_um']:.0f}"
              f"   {r['zone']:>8s} {r['sf']:4.1f} {r['eps']:4.1f} "
              f"{r['n_dimples']:5d} {r['gain_W']:8.4f} {r['gain_f']:7.4f} "
              f"{wup:>4s}{m}")

    return results


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "pump_pv_ps")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 90)
    print("МИКРОЛУНКИ 100-300 мкм — PAYVAR-SALANT")
    print(f"Сетка: {N_PHI}×{N_Z}")
    print("=" * 90)

    # Stage 0
    r0 = run_stage(S0_DIMPLES, S0_HP, S0_ZONES, S0_SF, S0_EPS,
                    "Stage0", out_dir)

    # Проверка: есть ли намёк на gain_W > 0.995?
    best = max(r0, key=lambda x: x["gain_W"]) if r0 else None
    if best and best["gain_W"] > 0.995:
        print(f"\n  Лучший gain_W = {best['gain_W']:.4f} > 0.995 "
              f"→ запускаю Stage 1")
        run_stage(S0_DIMPLES + [(150e-6, 120e-6)] if N_PHI >= 6000
                  else S0_DIMPLES,
                  S1_HP, S1_ZONES, S1_SF, S1_EPS, "Stage1", out_dir)
    else:
        gw = best["gain_W"] if best else 0
        print(f"\n  Лучший gain_W = {gw:.4f} ≤ 0.995")
        print("  Микролунки не дают прироста → Stage 1 не запускается.")


if __name__ == "__main__":
    main()
