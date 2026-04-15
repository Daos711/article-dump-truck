#!/usr/bin/env python3
"""Калибровка статической нагрузки для run_pump_harmonic.

Прогоняет серию W_static_x значений без гармоники, смотрит финальный ε.
Цель: выбрать такое W_static_x, при котором ε_final ≈ 0.4–0.6.
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from reynolds_solver import solve_ausas_journal_dynamic_gpu

from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Геометрия / режим ───────────────────────────────────────────
R = params.R
L = params.L
C = params.c
N_RPM = params.n
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * N_RPM / 60
P_SCALE = 6 * ETA * OMEGA * (R / C) ** 2
B_AUSAS = L / (2 * R)

# ─── Сетка ────────────────────────────────────────────────────────
N1 = 200
N2 = 20
N_PHI = N1 + 2
N_Z = N2 + 2
D_PHI = 1.0 / N1
D_Z = B_AUSAS / N2

R_SOLVER = 0.5
L_SOLVER = 1.0

# ─── Калибровка ──────────────────────────────────────────────────
MASS_M = 8e-4
DT = 1e-3
NT = 1000

W_CANDIDATES = [-0.01, -0.02, -0.04]

X0 = 0.3
Y0 = 0.0


def run_one(W_static_x):
    """Прогон до установления стационара."""
    def load_fn(t):
        return (W_static_x, 0.0)

    t0 = time.time()
    result = solve_ausas_journal_dynamic_gpu(
        NT=NT, dt=DT,
        N_Z=N_Z, N_phi=N_PHI,
        d_phi=D_PHI, d_Z=D_Z,
        R=R_SOLVER, L=L_SOLVER,
        mass_M=MASS_M,
        load_fn=load_fn,
        X0=X0, Y0=Y0,
        texture_relief=None,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
        scheme="rb", verbose=False,
    )
    dt_run = time.time() - t0

    X_f = float(result.X[-1])
    Y_f = float(result.Y[-1])
    eps_f = np.sqrt(X_f**2 + Y_f**2)
    h_min_f = float(result.h_min[-1])
    p_max_f = float(result.p_max[-1])
    cav_f = float(result.cav_frac[-1])

    return dict(W=W_static_x, eps=eps_f, X=X_f, Y=Y_f,
                h_min=h_min_f, p_max=p_max_f, cav=cav_f, dt=dt_run)


def main():
    print("=" * 70)
    print("КАЛИБРОВКА НАГРУЗКИ")
    print(f"R={R*1e3:.1f}мм, L={L*1e3:.1f}мм, C={C*1e6:.0f}мкм, "
          f"N={N_RPM}об/мин")
    print(f"Сетка: {N1}×{N2}, mass_M={MASS_M}, NT={NT}, dt={DT}")
    print(f"p_scale = {P_SCALE/1e6:.2f} МПа")
    print(f"F₀ = 24π·η·Ω·R⁴/C² = "
          f"{24*np.pi*ETA*OMEGA*R**4/C**2:.0f} Н")
    print("=" * 70)

    print(f"\n  {'W':>8} {'ε_final':>9} {'X':>8} {'Y':>8} "
          f"{'h_min':>8} {'p_max':>8} {'cav':>6} {'time':>6}")
    print(f"  {'-'*70}")

    results = []
    for W in W_CANDIDATES:
        r = run_one(W)
        results.append(r)
        print(f"  {r['W']:8.4f} {r['eps']:9.4f} {r['X']:8.4f} {r['Y']:8.4f} "
              f"{r['h_min']:8.4f} {r['p_max']:8.4f} {r['cav']:6.3f} "
              f"{r['dt']:5.1f}с")

    # Рекомендация
    print(f"\n  РАЗМЕРНЫЕ ЗНАЧЕНИЯ:")
    print(f"  {'W':>8} {'F, Н':>10} {'ε':>6} {'h_min, мкм':>12} {'p_max, МПа':>12}")
    F0 = 24 * np.pi * ETA * OMEGA * R**4 / C**2
    for r in results:
        F_dim = abs(r["W"]) * F0
        h_dim = r["h_min"] * C * 1e6
        p_dim = r["p_max"] * P_SCALE / 1e6
        print(f"  {r['W']:8.4f} {F_dim:10.0f} {r['eps']:6.3f} "
              f"{h_dim:12.2f} {p_dim:12.2f}")

    # Найти ближайший к ε=0.5
    best = min(results, key=lambda x: abs(x["eps"] - 0.5))
    print(f"\n  Рекомендуемое W_static_x: {best['W']:.4f} "
          f"(ε={best['eps']:.3f})")


if __name__ == "__main__":
    main()
