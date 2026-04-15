#!/usr/bin/env python3
"""Калибровка load_pct для дизельного подшипника.

Прогоняет 1 цикл с разными load_pct, печатает max ε, min h_min, max p_max.
Цель: выбрать load_pct где ε_max < 0.95 (без контакта).
"""
import sys
import os
import time

THIS = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(CASE_DIR))
sys.path.insert(0, CASE_DIR)
sys.path.insert(0, THIS)
sys.path.insert(0, ROOT)

import numpy as np

from reynolds_solver import solve_ausas_journal_dynamic_gpu

import case_config as cfg
from scaling import (
    omega_from_rpm, force_scale, pressure_scale, mass_nondim,
    make_load_fn_from_crank, CYCLE_TAU,
)

M_EFF_KG = 2.0

# Быстрый скрининг: 3 load_pct, мелкая сетка, большой dt
LOAD_CANDIDATES = [20, 50, 100]
N_RPM = 2200
DT_CAL = 4e-3                   # 2× быстрее
N1_CAL = 100                    # мельче сетка = 4× быстрее
N2_CAL = 10
MAX_INNER_CAL = 1000


def build_load(load_pct):
    from build_load_from_indicator import build_surrogate_load
    crank_deg, WaX_N, WaY_N = build_surrogate_load(
        n_rpm=N_RPM, load_pct=load_pct,
        n_cyl=cfg.n_cylinders, bore_m=cfg.bore_m,
        stroke_m=cfg.stroke_m, con_rod_m=cfg.con_rod_m,
        m_piston_kg=cfg.m_piston_kg,
        p_max_MPa=cfg.p_max_MPa, p_motoring_MPa=cfg.p_motoring_MPa,
        n_points=720)

    omega = omega_from_rpm(N_RPM)
    F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
    WaX_nd = WaX_N / F0
    WaY_nd = WaY_N / F0
    load_fn = make_load_fn_from_crank(crank_deg, WaX_nd, WaY_nd)
    W_max = np.sqrt(WaX_nd**2 + WaY_nd**2).max()
    return load_fn, W_max, F0


def run_one(load_fn):
    omega = omega_from_rpm(N_RPM)
    M_nd = mass_nondim(M_EFF_KG, cfg.eta, omega, cfg.R, cfg.c)
    B_ausas = cfg.L_bearing / (2 * cfg.R)
    NT = int(CYCLE_TAU / DT_CAL)

    result = solve_ausas_journal_dynamic_gpu(
        NT=NT, dt=DT_CAL,
        N_Z=N2_CAL + 2, N_phi=N1_CAL + 2,
        d_phi=1.0 / N1_CAL, d_Z=B_ausas / N2_CAL,
        R=0.5, L=1.0,
        mass_M=M_nd,
        load_fn=load_fn,
        X0=cfg.X0, Y0=cfg.Y0,
        texture_relief=None,
        omega_p=cfg.omega_p, omega_theta=cfg.omega_theta,
        tol_inner=cfg.tol_inner, max_inner=MAX_INNER_CAL,
        scheme=cfg.scheme, verbose=False,
    )
    eps = np.sqrt(np.asarray(result.X)**2 + np.asarray(result.Y)**2)
    return dict(eps_max=float(eps.max()),
                h_min=float(np.min(result.h_min)),
                p_max=float(np.max(result.p_max)))


def main():
    print("=" * 72)
    print(f"КАЛИБРОВКА LOAD_PCT (дизель, {N_RPM} rpm, m_eff={M_EFF_KG}кг)")
    print(f"Быстрая сетка {N1_CAL}×{N2_CAL}, dt={DT_CAL}, "
          f"NT/цикл={int(CYCLE_TAU/DT_CAL)}, max_inner={MAX_INNER_CAL}")
    print("=" * 72)

    omega = omega_from_rpm(N_RPM)
    F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
    p_sc = pressure_scale(cfg.eta, omega, cfg.R, cfg.c)
    print(f"F₀ = {F0:.0f} Н, p_scale = {p_sc/1e6:.2f} МПа\n")
    print(f"  {'load%':>5} {'|W|_max':>9} {'F_max,кН':>10} "
          f"{'ε_max':>6} {'h_min':>9} {'p_max':>9} {'time':>6}")
    print("  " + "-" * 62)

    for pct in LOAD_CANDIDATES:
        load_fn, W_max, F0 = build_load(pct)
        t0 = time.time()
        r = run_one(load_fn)
        dt = time.time() - t0
        F_max_kN = W_max * F0 / 1000
        h_um = r["h_min"] * cfg.c * 1e6
        p_MPa = r["p_max"] * p_sc / 1e6
        marker = " ← OK" if r["eps_max"] < 0.95 else ""
        print(f"  {pct:>4}% {W_max:>9.4f} {F_max_kN:>10.1f} "
              f"{r['eps_max']:>6.3f} {h_um:>7.2f}мкм {p_MPa:>7.1f}МПа "
              f"{dt:>5.0f}с{marker}")


if __name__ == "__main__":
    main()
