#!/usr/bin/env python3
"""Верификация текстуры: 3 точечных прогона на надёжной сетке.

Сетка 2000×500, крупные лунки (a=2.0мм, b=1.5мм, hp=5мкм).
Проверка: даёт ли текстура прирост и сходится ли HS warmup.
"""
import sys
import os
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import setup_grid, make_H, solve_and_compute
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL

N_PHI = 2000
N_Z = 500
EPS = 0.6
ETA = MINERAL_OIL["eta_pump"]
CAVITATION = "payvar_salant"
PROFILE = "smoothcap"

# Крупные лунки — ~15 узлов/лунку по φ, ~14 по Z на 2000×500
A_DIM = 2.0e-3
B_DIM = 1.5e-3
HP = 5e-6

RUNS = [
    {"label": "A: 180-360", "phi_start": 180, "phi_end": 360},
    {"label": "B: 90-180",  "phi_start": 90,  "phi_end": 180},
    {"label": "C: 120-200", "phi_start": 120, "phi_end": 200},
]


def compute_N_tex(phi_start, phi_end, a_dim, b_dim, R, L):
    if phi_start < phi_end:
        phi_span = np.deg2rad(phi_end - phi_start)
    else:
        phi_span = np.deg2rad((360 - phi_start) + phi_end)
    B = b_dim / R
    A = 2 * a_dim / L
    N_phi_tex = max(1, int(phi_span / (2.5 * 2 * B)))
    N_Z_tex = max(1, int(2.0 / (2.5 * 2 * A)))
    return N_phi_tex, N_Z_tex


def setup_texture_custom(p):
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


def main():
    print("=" * 75)
    print("ВЕРИФИКАЦИЯ ТЕКСТУРЫ")
    print(f"ε={EPS}, сетка {N_PHI}×{N_Z}, a={A_DIM*1e3:.1f}мм, "
          f"b={B_DIM*1e3:.1f}мм, hp={HP*1e6:.0f}мкм")
    print("=" * 75)

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)

    p_base = types.SimpleNamespace(**{k: getattr(base_params, k)
                                      for k in dir(base_params)
                                      if not k.startswith('_')})

    # --- Гладкий ---
    H_smooth = make_H(EPS, Phi_mesh, Z_mesh, p_base, textured=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        _, W_s, f_s, _, _, pmax_s, _, n_s, _, cf_s = solve_and_compute(
            H_smooth, d_phi, d_Z, p_base.R, p_base.L, ETA, p_base.n, p_base.c,
            phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)
        dt_s = time.time() - t0
        warn_s = any("warmup" in str(w.message) for w in caught)

    warn_tag = "ДА" if warn_s else "нет"
    print(f"\nW_smooth = {W_s:.0f} Н, f={f_s:.4f}, cav={cf_s:.1%}, "
          f"n_iter={n_s}, time={dt_s:.1f}с, warning={warn_tag}")

    # --- Текстурированные ---
    print()
    for run in RUNS:
        p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                     for k in dir(base_params)
                                     if not k.startswith('_')})
        p.a_dim = A_DIM
        p.b_dim = B_DIM
        p.h_p = HP
        p.phi_start_deg = run["phi_start"]
        p.phi_end_deg = run["phi_end"]

        N_pt, N_zt = compute_N_tex(run["phi_start"], run["phi_end"],
                                    A_DIM, B_DIM, base_params.R, base_params.L)
        p.N_phi_tex = N_pt
        p.N_Z_tex = N_zt

        phi_c, Z_c = setup_texture_custom(p)
        n_dimples = len(phi_c)

        A = 2 * p.a_dim / p.L
        B = p.b_dim / p.R
        H_p = p.h_p / p.c

        H0 = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
        H_tex = create_H_with_ellipsoidal_depressions(
            H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A, B, profile=PROFILE)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.time()
            _, W_t, f_t, _, _, pmax_t, _, n_t, _, cf_t = solve_and_compute(
                H_tex, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
                phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)
            dt = time.time() - t0
            warn_t = any("warmup" in str(w.message) for w in caught)

        gw = W_t / W_s if W_s > 0 else 0
        gf = f_t / f_s if f_s > 0 else 0
        warn_tag = "ДА" if warn_t else "нет"

        print(f"  {run['label']:>12s}  N={n_dimples:>3d}  "
              f"W={W_t:8.0f}  gain_W={gw:.4f}  gain_f={gf:.4f}  "
              f"cav={cf_t:.1%}  n_iter={n_t}  warn={warn_tag}  "
              f"{dt:.1f}с")

    print(f"\n{'=' * 75}")


if __name__ == "__main__":
    main()
