#!/usr/bin/env python3
"""Проверка турбулентной поправки Constantinescu на эффект текстуры.

HS + laminar vs HS + constantinescu для 5 лучших кандидатов.
Ключевой вопрос: меняет ли Constantinescu ОТНОСИТЕЛЬНЫЙ gain_W?

PS не поддерживает constantinescu, поэтому используется half_sommerfeld.
HS завышает абсолютный gain — смотрим только на Δgain_W = gain_turb - gain_lam.
"""
import sys
import os
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import setup_grid, make_H, solve_and_compute
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL

PROFILE = "smoothcap"
N_PHI = 800
N_Z = 200
ETA = MINERAL_OIL["eta_pump"]
RHO = MINERAL_OIL["rho"]
EPS_VALUES = [0.3, 0.6]

# Топ-5 кандидатов из предыдущего поиска (зона 0-90, sf=1.5)
CANDIDATES = [
    {"label": "1", "hp": 30e-6, "a": 1.5e-3, "b": 1.2e-3},
    {"label": "2", "hp": 20e-6, "a": 1.5e-3, "b": 1.2e-3},
    {"label": "3", "hp": 15e-6, "a": 2.0e-3, "b": 1.5e-3},
    {"label": "4", "hp": 10e-6, "a": 1.5e-3, "b": 1.2e-3},
    {"label": "5", "hp": 7e-6,  "a": 2.0e-3, "b": 1.5e-3},
]
ZONE = (0, 90)
SF = 1.5


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
    phi_span = phi_e - phi_s
    margin = B * 1.1
    usable = phi_span - 2 * margin
    if usable <= 0:
        return np.array([]), np.array([])
    if p.N_phi_tex == 1:
        phi_c = np.array([phi_s + phi_span / 2])
    else:
        phi_c = phi_s + margin + np.linspace(0, usable, p.N_phi_tex)
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
    omega = 2 * np.pi * base_params.n / 60.0
    U = omega * base_params.R
    Re = RHO * U * base_params.c / ETA

    print("=" * 95)
    print("ПРОВЕРКА ТУРБУЛЕНТНОЙ ПОПРАВКИ CONSTANTINESCU")
    print(f"Сетка: {N_PHI}×{N_Z}, кавитация: half_sommerfeld")
    print(f"Re_global = ρ·U·c/μ = {RHO}·{U:.1f}·{base_params.c*1e6:.0f}e-6"
          f"/{ETA} = {Re:.1f}")
    print("=" * 95)

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)

    closures = [
        ("laminar", "laminar", {}),
        ("constantinescu", "constantinescu", {
            "rho": RHO, "U_velocity": U,
            "mu": ETA, "c_clearance": base_params.c,
        }),
    ]

    results = []

    for eps in EPS_VALUES:
        print(f"\n  ε = {eps}")

        # Smooth для обоих closures
        smooth = {}
        for cl_name, cl_val, cl_kw in closures:
            H_s = make_H(eps, Phi_mesh, Z_mesh, base_params, textured=False)

            # solve_and_compute через API (не PS, а HS)
            _, W_s, f_s, _, _, pmax_s, _, _, _, _ = solve_and_compute(
                H_s, d_phi, d_Z, base_params.R, base_params.L, ETA,
                base_params.n, base_params.c,
                phi_1D, Z_1D, Phi_mesh,
                closure=cl_val, cavitation="half_sommerfeld")
            smooth[cl_name] = {"W": W_s, "f": f_s, "pmax": pmax_s}
            print(f"    Smooth {cl_name:>15s}: W={W_s:.0f}")

        # Кандидаты
        for cand in CANDIDATES:
            p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                         for k in dir(base_params)
                                         if not k.startswith('_')})
            p.a_dim, p.b_dim, p.h_p = cand["a"], cand["b"], cand["hp"]
            p.phi_start_deg, p.phi_end_deg = ZONE
            Nt_phi, Nt_Z = compute_N_tex(ZONE, cand["a"], cand["b"],
                                          p.R, p.L, SF)
            p.N_phi_tex, p.N_Z_tex = Nt_phi, Nt_Z

            phi_c, Z_c = setup_tex(p)
            n_dimples = len(phi_c)
            A_nd = 2 * cand["a"] / p.L
            B_nd = cand["b"] / p.R
            H_p = cand["hp"] / p.c

            H0 = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
            H_tex = create_H_with_ellipsoidal_depressions(
                H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A_nd, B_nd,
                profile=PROFILE)

            gains = {}
            for cl_name, cl_val, cl_kw in closures:
                _, W_t, f_t, _, _, pmax_t, _, _, _, _ = solve_and_compute(
                    H_tex, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
                    phi_1D, Z_1D, Phi_mesh,
                    closure=cl_val, cavitation="half_sommerfeld")
                sc = smooth[cl_name]
                gw = W_t / sc["W"] if sc["W"] > 0 else 0
                gf = f_t / sc["f"] if sc["f"] > 0 else 0
                gains[cl_name] = {"W_tex": W_t, "gain_W": gw, "gain_f": gf}

            delta_gw = gains["constantinescu"]["gain_W"] - gains["laminar"]["gain_W"]

            print(f"    Кандидат {cand['label']}: hp={cand['hp']*1e6:.0f}мкм "
                  f"gW_lam={gains['laminar']['gain_W']:.4f} "
                  f"gW_turb={gains['constantinescu']['gain_W']:.4f} "
                  f"Δ={delta_gw:+.4f}")

            results.append({
                "cand": cand["label"], "eps": eps,
                "hp_um": cand["hp"] * 1e6,
                "a_mm": cand["a"] * 1e3, "b_mm": cand["b"] * 1e3,
                "W_s_lam": smooth["laminar"]["W"],
                "W_s_turb": smooth["constantinescu"]["W"],
                "W_t_lam": gains["laminar"]["W_tex"],
                "W_t_turb": gains["constantinescu"]["W_tex"],
                "gain_W_lam": gains["laminar"]["gain_W"],
                "gain_W_turb": gains["constantinescu"]["gain_W"],
                "delta_gW": delta_gw,
                "gain_f_lam": gains["laminar"]["gain_f"],
                "gain_f_turb": gains["constantinescu"]["gain_f"],
            })

    # CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "pump_pv_ps")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "constantinescu_check.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand", "eps", "hp_um", "a_mm", "b_mm",
                     "W_s_lam", "W_s_turb", "W_t_lam", "W_t_turb",
                     "gain_W_lam", "gain_W_turb", "delta_gW",
                     "gain_f_lam", "gain_f_turb"])
        for r in results:
            w.writerow([r["cand"], f"{r['eps']:.1f}", f"{r['hp_um']:.0f}",
                        f"{r['a_mm']:.1f}", f"{r['b_mm']:.1f}",
                        f"{r['W_s_lam']:.1f}", f"{r['W_s_turb']:.1f}",
                        f"{r['W_t_lam']:.1f}", f"{r['W_t_turb']:.1f}",
                        f"{r['gain_W_lam']:.4f}", f"{r['gain_W_turb']:.4f}",
                        f"{r['delta_gW']:.4f}",
                        f"{r['gain_f_lam']:.4f}", f"{r['gain_f_turb']:.4f}"])
    print(f"\n  CSV: {csv_path}")

    # Итог
    print(f"\n{'=' * 95}")
    print(f"Re = {Re:.1f}")
    deltas = [r["delta_gW"] for r in results]
    max_delta = max(abs(d) for d in deltas)
    print(f"|Δgain_W|_max = {max_delta:.4f}")
    if max_delta < 0.001:
        print("Constantinescu НЕ ВЛИЯЕТ на относительный эффект текстуры "
              "(|Δ| < 0.001).")
    elif max_delta < 0.01:
        print("Constantinescu СЛАБО влияет (|Δ| < 0.01).")
    else:
        print("Constantinescu ЗАМЕТНО влияет — требуется стыковка с PS.")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
