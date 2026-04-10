#!/usr/bin/env python3
"""Насосный sweep с Payvar-Salant JFO кавитацией.

Задача 1/4: только гладкий подшипник (без текстуры).
Sweep по ε = 0.1..0.8, оба масла.
Вывод: таблица W(ε), timing одной точки.
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from models.bearing_model import setup_grid, make_H, solve_and_compute
from config import pump_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

CAVITATION = "payvar_salant"


def run_smooth_sweep(N_phi, N_Z, epsilon_values, oils):
    """Sweep гладкого подшипника по ε для каждого масла.

    Returns
    -------
    results : dict  oil_name -> dict(eps, W, f, hmin, Q, pmax, F_tr, cav_frac)
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)

    results = {}
    for oil_name, oil in oils:
        eta = oil["eta_pump"]
        n_eps = len(epsilon_values)
        W = np.zeros(n_eps)
        f = np.zeros(n_eps)
        hmin = np.zeros(n_eps)
        Q = np.zeros(n_eps)
        pmax = np.zeros(n_eps)
        F_tr = np.zeros(n_eps)
        cav_frac = np.zeros(n_eps)

        P_prev = None
        print(f"\n  {oil_name} (eta={eta} Па·с):")

        for ie, eps in enumerate(epsilon_values):
            H = make_H(eps, Phi_mesh, Z_mesh, params, textured=False)

            P, F, mu, Qv, h_m, p_m, F_friction, n_out, theta, cf = \
                solve_and_compute(
                    H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
                    phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
                    cavitation=CAVITATION,
                )
            P_prev = P

            W[ie] = F
            f[ie] = mu
            hmin[ie] = h_m
            Q[ie] = Qv
            pmax[ie] = p_m
            F_tr[ie] = F_friction
            cav_frac[ie] = cf

            print(f"    eps={eps:.2f}: W={F:8.0f} Н, f={mu:.4f}, "
                  f"h_min={h_m*1e6:5.1f} мкм, p_max={p_m/1e6:6.1f} МПа, "
                  f"cav={cf:.1%}")

        results[oil_name] = {
            "eps": epsilon_values, "W": W, "f": f, "hmin": hmin,
            "Q": Q, "pmax": pmax, "F_tr": F_tr, "cav_frac": cav_frac,
        }

    return results


def run_timing(N_phi, N_Z, eps=0.6):
    """Timing одной точки для каждого масла."""
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)
    H = make_H(eps, Phi_mesh, Z_mesh, params, textured=False)

    print(f"\n  Timing: сетка {N_phi}x{N_Z}, eps={eps}")
    for oil_name, oil in [("Минеральное", MINERAL_OIL),
                           ("Рапсовое", RAPESEED_OIL)]:
        eta = oil["eta_pump"]
        t0 = time.time()
        P, F, mu, Qv, h_m, p_m, F_friction, n_out, theta, cf = \
            solve_and_compute(
                H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
                phi_1D, Z_1D, Phi_mesh,
                cavitation=CAVITATION,
            )
        dt = time.time() - t0
        print(f"    {oil_name}: W={F:.0f} Н, time={dt:.2f} с, "
              f"n_iter={n_out}, cav={cf:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Гладкий подшипник насоса с Payvar-Salant кавитацией")
    parser.add_argument("--nphi", type=int, default=2000,
                        help="Узлов по phi (default: 2000)")
    parser.add_argument("--nz", type=int, default=500,
                        help="Узлов по Z (default: 500)")
    parser.add_argument("--neps", type=int, default=15,
                        help="Число точек по eps (default: 15)")
    parser.add_argument("--timing-only", action="store_true",
                        help="Только timing одной точки (eps=0.6)")
    args = parser.parse_args()

    epsilon_values = np.linspace(0.1, 0.8, args.neps)
    oils = [("Минеральное", MINERAL_OIL), ("Рапсовое", RAPESEED_OIL)]

    print("=" * 65)
    print("ПОДШИПНИК НАСОСА — PAYVAR-SALANT JFO (гладкий)")
    print(f"Сетка: {args.nphi}x{args.nz}")
    print(f"R={params.R*1e3:.1f} мм, c={params.c*1e6:.0f} мкм, "
          f"L={params.L*1e3:.1f} мм, n={params.n} об/мин")
    print(f"Без пьезовязкости, без текстуры")
    print("=" * 65)

    if args.timing_only:
        run_timing(args.nphi, args.nz)
        return

    # --- Full sweep ---
    t0 = time.time()
    results = run_smooth_sweep(args.nphi, args.nz, epsilon_values, oils)
    dt_total = time.time() - t0
    print(f"\nОбщее время sweep: {dt_total:.1f} с")

    # --- Сводная таблица ---
    print(f"\n{'=' * 65}")
    print("СВОДНАЯ ТАБЛИЦА W(eps) — гладкий подшипник")
    print(f"{'=' * 65}")
    print(f"{'eps':>5}  ", end="")
    for oil_name in results:
        print(f"{'W(Н)':>9} {'f':>7} {'cav%':>5}  ", end="")
    print()
    print("-" * 65)

    for ie, eps in enumerate(epsilon_values):
        print(f"{eps:5.2f}  ", end="")
        for oil_name, r in results.items():
            print(f"{r['W'][ie]:9.0f} {r['f'][ie]:7.4f} "
                  f"{r['cav_frac'][ie]*100:5.1f}  ", end="")
        print()

    # --- Timing ---
    run_timing(args.nphi, args.nz)

    # --- Сохранить данные ---
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump")
    os.makedirs(out_dir, exist_ok=True)
    save_dict = {"epsilon": epsilon_values, "grid": [args.nphi, args.nz]}
    for oil_name, r in results.items():
        prefix = oil_name[:3] + "_"
        for key in ["W", "f", "hmin", "Q", "pmax", "F_tr", "cav_frac"]:
            save_dict[prefix + key] = r[key]
    out_path = os.path.join(out_dir, "smooth_ps.npz")
    np.savez(out_path, **save_dict)
    print(f"\nДанные: {out_path}")


if __name__ == "__main__":
    main()
