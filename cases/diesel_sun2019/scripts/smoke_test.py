#!/usr/bin/env python3
"""Очень быстрый smoke test параметров.

1 цикл на мелкой сетке 100×10 с текущими настройками case_config.
Цель: за 1-2 минуты увидеть ε_max и решить запускать ли full run.
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

M_EFF_KG = 10.0
N1 = 100
N2 = 10
DT = 1e-3


def main():
    omega = omega_from_rpm(cfg.n_rpm)
    F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
    p_sc = pressure_scale(cfg.eta, omega, cfg.R, cfg.c)
    M_nd = mass_nondim(M_EFF_KG, cfg.eta, omega, cfg.R, cfg.c)
    B_ausas = cfg.L_bearing / (2 * cfg.R)
    NT = int(CYCLE_TAU / DT)

    print("=" * 65)
    print("SMOKE TEST — diesel smooth, 1 cycle on coarse grid")
    print(f"p_max={cfg.p_max_MPa} МПа, load_pct={cfg.load_pct}%, "
          f"m_eff={M_EFF_KG}кг")
    print(f"Сетка {N1}×{N2}, dt={DT}, NT={NT}")
    print("=" * 65)

    # Build surrogate load on the fly
    from build_load_from_indicator import build_surrogate_load
    crank_deg, WaX_N, WaY_N = build_surrogate_load(
        n_rpm=cfg.n_rpm, load_pct=cfg.load_pct,
        n_cyl=cfg.n_cylinders, bore_m=cfg.bore_m,
        stroke_m=cfg.stroke_m, con_rod_m=cfg.con_rod_m,
        m_piston_kg=cfg.m_piston_kg,
        p_max_MPa=cfg.p_max_MPa, p_motoring_MPa=cfg.p_motoring_MPa,
        n_points=720)

    WaX_nd = WaX_N / F0
    WaY_nd = WaY_N / F0
    W_max = np.sqrt(WaX_nd**2 + WaY_nd**2).max()
    F_max_kN = W_max * F0 / 1000
    print(f"\nSurrogate load: |W_nd|_max = {W_max:.4f}, "
          f"F_max = {F_max_kN:.1f} кН")

    load_fn = make_load_fn_from_crank(crank_deg, WaX_nd, WaY_nd)

    print(f"\nStart cycle...")
    t0 = time.time()
    result = solve_ausas_journal_dynamic_gpu(
        NT=NT, dt=DT,
        N_Z=N2 + 2, N_phi=N1 + 2,
        d_phi=1.0 / N1, d_Z=B_ausas / N2,
        R=0.5, L=1.0,
        mass_M=M_nd,
        load_fn=load_fn,
        X0=cfg.X0, Y0=cfg.Y0,
        texture_relief=None,
        omega_p=cfg.omega_p, omega_theta=cfg.omega_theta,
        tol_inner=cfg.tol_inner, max_inner=1000,
        scheme=cfg.scheme, verbose=False,
    )
    dt_run = time.time() - t0

    X = np.asarray(result.X)
    Y = np.asarray(result.Y)
    eps = np.sqrt(X**2 + Y**2)
    h_min = np.asarray(result.h_min)
    p_max = np.asarray(result.p_max)

    print(f"\nЗа {dt_run:.0f} с:")
    print(f"  ε:        max={eps.max():.4f}, min={eps.min():.4f}, "
          f"end={eps[-1]:.4f}")
    print(f"  h_min:    min={h_min.min():.4f} ({h_min.min()*cfg.c*1e6:.2f} мкм)")
    print(f"  p_max:    max={p_max.max():.4f} ({p_max.max()*p_sc/1e6:.1f} МПа)")

    print()
    if eps.max() > 0.99:
        print(f"  ✗ ε_max > 0.99 — контакт. НЕ запускай full run.")
        print(f"    Уменьши load_pct или p_max ещё.")
    elif eps.max() > 0.95:
        print(f"  ~ ε_max > 0.95 — на границе. Сомнительно.")
    else:
        print(f"  ✓ ε_max = {eps.max():.3f} — параметры рабочие.")
        print(f"    Можно запускать run_smooth_cycle.py")


if __name__ == "__main__":
    main()
