#!/usr/bin/env python3
"""Grid convergence проверка вариантов 12 и 14 на сетке 8000×200."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import setup_grid, make_H, solve_and_compute
from config import pump_params as params
from config.oil_properties import MINERAL_OIL
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

# Импортируем setup_texture_custom из sweep_micro
from scripts.run_sweep_micro import setup_texture_custom

N_PHI = 8000
N_Z = 600
EPS_TARGET = 0.60
PROFILE = "sqrt"

VARIANTS = [
    # (имя, phi_start, phi_end, a_mm, b_mm, h_p_um, nphi_tex, nz_tex)
    ("12 b0.5 h5  90-180",  90, 180,  0.5, 0.5,   5,  36, 37),
    ("14 b0.5 h5  0-180",   0,  180,  0.5, 0.5,   5,  72, 37),
]


def run_single(phi_start_deg, phi_end_deg, a_mm, b_mm, h_p_um, nphi_tex, nz_tex):
    p = types.SimpleNamespace(**{k: getattr(params, k)
                                 for k in dir(params) if not k.startswith('_')})
    p.h_p = h_p_um * 1e-6
    p.a_dim = a_mm * 1e-3
    p.b_dim = b_mm * 1e-3
    p.phi_start_deg = phi_start_deg
    p.phi_end_deg = phi_end_deg
    p.N_phi_tex = nphi_tex
    p.N_Z_tex = nz_tex

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)
    phi_c, Z_c = setup_texture_custom(p)

    eta = MINERAL_OIL["eta_pump"]
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c

    H_s = make_H(EPS_TARGET, Phi_mesh, Z_mesh, p, textured=False)
    _, W_s, f_s, _, _, pmax_s, _, _ = solve_and_compute(
        H_s, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh)

    H0 = make_H(EPS_TARGET, Phi_mesh, Z_mesh, p, textured=False)
    H_t = create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A, B, profile=PROFILE)

    _, W_t, f_t, _, _, pmax_t, _, _ = solve_and_compute(
        H_t, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
        phi_1D, Z_1D, Phi_mesh)

    g = lambda a, b: a / b if b > 0 else 0.0
    return {
        "W_s": W_s, "W_t": W_t, "gain_W": g(W_t, W_s),
        "f_s": f_s, "f_t": f_t, "gain_f": g(f_t, f_s),
        "pmax_s": pmax_s, "pmax_t": pmax_t, "gain_pmax": g(pmax_t, pmax_s),
        "n_dimples": len(phi_c),
    }


def main():
    print("=" * 80)
    print("GRID CONVERGENCE: варианты 12, 14 на 8000×200")
    print(f"ε = {EPS_TARGET}, профиль = {PROFILE}, без PV")
    print("=" * 80)

    print(f"\nСравнение с 5000×200:")
    print(f"  Вар.12 (5000): gain_W = 1.1551")
    print(f"  Вар.14 (5000): gain_W = 1.1601")
    print()

    for name, ps, pe, a, b, hp, npt, nzt in VARIANTS:
        t0 = time.time()
        sys.stdout.write(f"  {name:25s} ... ")
        sys.stdout.flush()
        r = run_single(ps, pe, a, b, hp, npt, nzt)
        dt = time.time() - t0
        print(f"gain_W={r['gain_W']:.4f}  ({dt:.1f} с)")
        print(f"    W_smooth={r['W_s']:.0f}  W_tex={r['W_t']:.0f}  "
              f"gain_f={r['gain_f']:.4f}  gain_pmax={r['gain_pmax']:.4f}")

    print("\nЕсли Δgain_W < 5% от 5000 → сетка сошлась.")


if __name__ == "__main__":
    main()
