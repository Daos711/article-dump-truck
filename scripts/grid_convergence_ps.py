#!/usr/bin/env python3
"""Grid convergence для конфига B с Payvar-Salant.

Сравнение грубой (2000×500) и рабочей (4500×950) сеток при ε=0.6.
Разница gain_W должна быть < 5%.
"""
import sys
import os
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
)
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL
from config.pump_params_micro import CONFIG_B

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"
EPS = 0.6
ETA = MINERAL_OIL["eta_pump"]

GRIDS = [
    (2000, 500,  "грубая"),
    (4500, 950,  "рабочая"),
]


def make_params(cfg):
    p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                 for k in dir(base_params)
                                 if not k.startswith('_')})
    for k, v in cfg.items():
        if k not in ("label", "N_phi", "N_Z"):
            setattr(p, k, v)
    return p


def run_one(N_phi, N_Z, p):
    """Прогнать гладкий + текстурированный на одной сетке.

    Returns dict с W_s, W_t, gain_W, f_s, f_t, gain_f, ...
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)
    phi_c, Z_c = setup_texture(p)

    # Гладкий
    H_s = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
    _, W_s, f_s, _, _, pmax_s, _, _, _, cf_s = solve_and_compute(
        H_s, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)

    # Текстурированный
    H_t = make_H(EPS, Phi_mesh, Z_mesh, p, textured=True,
                 phi_c_flat=phi_c, Z_c_flat=Z_c, profile=PROFILE)
    _, W_t, f_t, _, _, pmax_t, _, _, _, cf_t = solve_and_compute(
        H_t, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)

    g = lambda a, b: a / b if b > 0 else 0.0
    return {
        "W_s": W_s, "W_t": W_t, "gain_W": g(W_t, W_s),
        "f_s": f_s, "f_t": f_t, "gain_f": g(f_t, f_s),
        "pmax_s": pmax_s, "pmax_t": pmax_t, "gain_pmax": g(pmax_t, pmax_s),
        "cav_s": cf_s, "cav_t": cf_t,
    }


def main():
    cfg = CONFIG_B
    p = make_params(cfg)

    print("=" * 70)
    print(f"GRID CONVERGENCE: конфиг {cfg['label']}, Payvar-Salant, ε={EPS}")
    print(f"Лунка: a={p.a_dim*1e3:.2f} мм, b={p.b_dim*1e3:.2f} мм, "
          f"hp={p.h_p*1e6:.0f} мкм")
    print(f"Зона: {p.phi_start_deg}°–{p.phi_end_deg}°, "
          f"N_tex={p.N_phi_tex}×{p.N_Z_tex}")
    print("=" * 70)

    results = []
    for N_phi, N_Z, tag in GRIDS:
        t0 = time.time()
        print(f"\n  {tag} ({N_phi}×{N_Z}) ...", end="", flush=True)
        r = run_one(N_phi, N_Z, p)
        dt = time.time() - t0
        print(f" {dt:.1f} с")
        print(f"    W_smooth={r['W_s']:.0f}, W_tex={r['W_t']:.0f}, "
              f"gain_W={r['gain_W']:.4f}")
        print(f"    f_smooth={r['f_s']:.4f}, f_tex={r['f_t']:.4f}, "
              f"gain_f={r['gain_f']:.4f}")
        results.append((N_phi, N_Z, tag, r, dt))

    # Сравнение
    r_coarse = results[0][3]
    r_fine = results[1][3]
    delta_gain_W = abs(r_fine["gain_W"] - r_coarse["gain_W"])
    rel_delta = delta_gain_W / max(r_fine["gain_W"], 1e-9) * 100

    print(f"\n{'=' * 70}")
    print(f"СРАВНЕНИЕ СЕТОК")
    print(f"{'=' * 70}")
    print(f"  gain_W грубая:  {r_coarse['gain_W']:.4f}")
    print(f"  gain_W рабочая: {r_fine['gain_W']:.4f}")
    print(f"  Δ(gain_W) = {delta_gain_W:.4f} ({rel_delta:.1f}%)")
    if rel_delta < 5:
        print(f"  OK — разница < 5%, сеточная сходимость достигнута.")
    else:
        print(f"  !!! Разница ≥ 5%, использовать только рабочую сетку.")

    # Сохранить CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump_ps")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "grid_convergence.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N_phi", "N_Z", "tag", "W_smooth", "W_tex", "gain_W",
                     "f_smooth", "f_tex", "gain_f", "pmax_smooth_MPa",
                     "pmax_tex_MPa", "gain_pmax", "time_s"])
        for N_phi, N_Z, tag, r, dt in results:
            w.writerow([N_phi, N_Z, tag,
                        f"{r['W_s']:.1f}", f"{r['W_t']:.1f}",
                        f"{r['gain_W']:.4f}",
                        f"{r['f_s']:.5f}", f"{r['f_t']:.5f}",
                        f"{r['gain_f']:.4f}",
                        f"{r['pmax_s']/1e6:.2f}", f"{r['pmax_t']/1e6:.2f}",
                        f"{r['gain_pmax']:.4f}",
                        f"{dt:.1f}"])
    print(f"\n  CSV: {csv_path}")


if __name__ == "__main__":
    main()
