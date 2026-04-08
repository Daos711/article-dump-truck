#!/usr/bin/env python3
"""Точечные прогоны текстуры с JFO кавитацией + мелкая глубина."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from models.bearing_model import setup_grid, make_H
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

EPS = 0.6
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * params.n / 60.0
P_SCALE = 6.0 * ETA * OMEGA * (params.R / params.c) ** 2
PROFILE = "sqrt"
CAVITATION = "jfo"

# Варианты B, C, D с h_p = 3–5 мкм
VARIANTS = [
    # (имя, a_mm, b_mm, h_p_um, phi_start, phi_end, nphi_tex, nz_tex, N_phi, N_Z)
    ("B1 сред h3",   1.0, 1.0,  3, 90, 180, 18, 18, 2000, 200),
    ("B2 сред h5",   1.0, 1.0,  5, 90, 180, 18, 18, 2000, 200),
    ("C1 микро h3",  0.5, 0.5,  3, 90, 180, 36, 37, 5000, 400),
    ("C2 микро h5",  0.5, 0.5,  5, 90, 180, 36, 37, 5000, 400),
    ("D1 микро h3 wide", 0.5, 0.5, 3, 0, 180, 36, 37, 5000, 400),
    ("D2 микро h5 wide", 0.5, 0.5, 5, 0, 180, 36, 37, 5000, 400),
]


def setup_texture_custom(p):
    """Расставить центры лунок с учётом wrap-around."""
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    phi_s = np.deg2rad(p.phi_start_deg)
    phi_e = np.deg2rad(p.phi_end_deg)

    if phi_s > phi_e:
        phi_span = (2 * np.pi - phi_s) + phi_e
    else:
        phi_span = phi_e - phi_s

    N_phi_t = p.N_phi_tex
    N_Z_t = p.N_Z_tex

    total_dimple = N_phi_t * 2 * B
    if total_dimple > phi_span and N_phi_t > 1:
        N_phi_t = max(1, int(phi_span / (2.5 * B)))
    if N_phi_t == 1:
        phi_centers = np.array([phi_s + phi_span / 2])
    else:
        gap = (phi_span - N_phi_t * 2 * B) / max(N_phi_t - 1, 1)
        step = 2 * B + gap
        phi_centers = phi_s + B + step * np.arange(N_phi_t)
    phi_centers = phi_centers % (2 * np.pi)

    total_Z = N_Z_t * 2 * A
    if total_Z > 2.0 and N_Z_t > 1:
        N_Z_t = max(1, int(2.0 / (2.5 * A)))
    if N_Z_t == 1:
        Z_centers = np.array([0.0])
    else:
        gap_Z = (2.0 - N_Z_t * 2 * A) / max(N_Z_t - 1, 1)
        step_Z = 2 * A + gap_Z
        Z_centers = -1.0 + A + step_Z * np.arange(N_Z_t)

    phi_g, Z_g = np.meshgrid(phi_centers, Z_centers)
    return phi_g.flatten(), Z_g.flatten()


def run_variant(name, a_mm, b_mm, h_p_um, phi_s, phi_e, npt, nzt, N_phi, N_Z):
    p = types.SimpleNamespace(**{k: getattr(params, k)
                                 for k in dir(params) if not k.startswith('_')})
    p.h_p = h_p_um * 1e-6
    p.a_dim = a_mm * 1e-3
    p.b_dim = b_mm * 1e-3
    p.phi_start_deg = phi_s
    p.phi_end_deg = phi_e
    p.N_phi_tex = npt
    p.N_Z_tex = nzt

    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = setup_grid(N_phi, N_Z)
    phi_c, Z_c = setup_texture_custom(p)

    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c

    def solve_W(H_field):
        result = solve_reynolds(
            H_field, d_phi, d_Z, params.R, params.L,
            closure="laminar", cavitation=CAVITATION,
            jfo_max_outer=2000)

        # JFO возвращает (P, theta, residual, n_outer, n_inner)
        P, theta, residual, n_outer, n_inner = result

        P_dim = P * P_SCALE
        Fx = -np.trapezoid(np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
                           Z_1D) * params.R * params.L / 2
        Fy = -np.trapezoid(np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
                           Z_1D) * params.R * params.L / 2
        W = np.sqrt(Fx**2 + Fy**2)

        h_dim = H_field * params.c
        h_min = np.min(h_dim)
        p_max = np.max(P_dim)

        tau_c = ETA * OMEGA * params.R / h_dim
        dP = np.gradient(P_dim, phi_1D[1] - phi_1D[0], axis=1)
        tau_p = h_dim / 2.0 * dP / params.R
        F_fr = np.trapezoid(np.trapezoid(np.abs(tau_c + tau_p), phi_1D, axis=1),
                            Z_1D) * params.R * params.L / 2
        f = F_fr / max(W, 1.0)
        return W, f, h_min, p_max, n_outer, n_inner

    # Гладкий
    H_s = make_H(EPS, Phi, Zm, p, textured=False)
    W_s, f_s, hmin_s, pmax_s, no_s, ni_s = solve_W(H_s)

    # Текстурированный
    H0 = make_H(EPS, Phi, Zm, p, textured=False)
    H_t = create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi, Zm, phi_c, Z_c, A, B, profile=PROFILE)
    W_t, f_t, hmin_t, pmax_t, no_t, ni_t = solve_W(H_t)

    g = lambda a, b: a / b if b > 0 else 0
    return {
        "W_s": W_s, "W_t": W_t, "gain_W": g(W_t, W_s),
        "f_s": f_s, "f_t": f_t, "gain_f": g(f_t, f_s),
        "hmin_t": hmin_t * 1e6, "pmax_t": pmax_t / 1e6,
        "n_outer": no_t, "n_inner": ni_t,
        "n_dimples": len(phi_c),
    }


def main():
    print("=" * 100)
    print("ТЕКСТУРА НАСОСА С JFO КАВИТАЦИЕЙ")
    print(f"ε = {EPS}, η = {ETA} Па·с, профиль = {PROFILE}, cavitation = {CAVITATION}")
    print(f"h_p = 3–5 мкм (h_p/h_min ≈ 0.15–0.25)")
    print("=" * 100)

    results = []
    t0_all = time.time()

    for v in VARIANTS:
        name = v[0]
        t0 = time.time()
        sys.stdout.write(f"  {name:20s} ({v[8]}x{v[9]}) ... ")
        sys.stdout.flush()
        r = run_variant(*v)
        dt = time.time() - t0
        marker = " <<<" if r["gain_W"] > 1.03 else ""
        print(f"gain_W={r['gain_W']:.4f}  gain_f={r['gain_f']:.4f}  "
              f"n_out={r['n_outer']}  ({dt:.1f} с){marker}")
        results.append((v, r, dt))

    dt_all = time.time() - t0_all
    print(f"\nОбщее время: {dt_all:.0f} с")

    # Сводная таблица
    print(f"\n{'=' * 110}")
    print("СВОДНАЯ ТАБЛИЦА (JFO)")
    print(f"{'=' * 110}")
    print(f"{'Вариант':<20} {'a,b мм':>8} {'h_p':>4} {'Зона':>10} {'Сетка':>10} "
          f"{'W_sm':>7} {'W_tex':>7} {'gain_W':>7} {'gain_f':>7} "
          f"{'n_out':>5} {'n_in':>6}")
    print("-" * 110)

    # Сортировка по gain_W
    results.sort(key=lambda x: x[1]["gain_W"], reverse=True)

    for v, r, dt in results:
        name = v[0]
        ab = f"{v[1]:.1f}/{v[2]:.1f}"
        zone = f"{v[4]}-{v[5]}°"
        grid = f"{v[8]}x{v[9]}"
        marker = " <<<" if r["gain_W"] > 1.03 else ""
        print(f"{name:<20} {ab:>8} {v[3]:>4} {zone:>10} {grid:>10} "
              f"{r['W_s']:>7.0f} {r['W_t']:>7.0f} {r['gain_W']:>7.4f} "
              f"{r['gain_f']:>7.4f} {r['n_outer']:>5} {r['n_inner']:>6}{marker}")

    best = max(results, key=lambda x: x[1]["gain_W"])
    print(f"\nЛучший: {best[0][0]} — gain_W={best[1]['gain_W']:.4f}, "
          f"gain_f={best[1]['gain_f']:.4f}")

    # Сохранить
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "texture_jfo.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Текстура с JFO: ε={EPS}, {PROFILE}, {CAVITATION}\n")
        f.write(f"Время: {dt_all:.0f} с\n\n")
        for v, r, dt in results:
            f.write(f"{v[0]:<20} a/b={v[1]}/{v[2]}мм h_p={v[3]}мкм "
                    f"zone={v[4]}-{v[5]}° grid={v[8]}x{v[9]} "
                    f"gain_W={r['gain_W']:.4f} gain_f={r['gain_f']:.4f}\n")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
