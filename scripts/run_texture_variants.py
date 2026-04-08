#!/usr/bin/env python3
"""Точечные прогоны текстуры насоса: 4 варианта + мини-convergence лучшего."""
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

# (имя, a_mm, b_mm, h_p_um, phi_start, phi_end, nphi_tex, nz_tex, N_phi, N_Z)
VARIANTS = [
    ("A крупные",  2.41, 2.21, 15, 90, 270,  8,  9, 2000, 200),
    ("B средние",  1.0,  1.0,  15, 90, 180, 18, 18, 2000, 200),
    ("C микро h10", 0.5, 0.5,  10, 90, 180, 36, 37, 5000, 400),
    ("D микро h20", 0.5, 0.5,  20, 90, 180, 36, 37, 5000, 400),
]


def setup_texture_custom(p):
    """Расставить центры лунок с учётом wrap-around и произвольных размеров."""
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

    # Центры по φ
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

    # Центры по Z
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
    """Прогнать один вариант: гладкий + текстурированный."""
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
        P, delta, n_iter, converged = solve_reynolds(
            H_field, d_phi, d_Z, params.R, params.L,
            closure="laminar", cavitation="half_sommerfeld",
            return_converged=True)
        P_dim = P * P_SCALE
        Fx = -np.trapz(np.trapz(P_dim * np.cos(Phi), phi_1D, axis=1),
                       Z_1D) * params.R * params.L / 2
        Fy = -np.trapz(np.trapz(P_dim * np.sin(Phi), phi_1D, axis=1),
                       Z_1D) * params.R * params.L / 2
        W = np.sqrt(Fx**2 + Fy**2)

        h_dim = H_field * params.c
        h_min = np.min(h_dim)
        p_max = np.max(P_dim)

        # Трение
        tau_c = ETA * OMEGA * params.R / h_dim
        dP = np.gradient(P_dim, phi_1D[1] - phi_1D[0], axis=1)
        tau_p = h_dim / 2.0 * dP / params.R
        F_fr = np.trapz(np.trapz(np.abs(tau_c + tau_p), phi_1D, axis=1),
                        Z_1D) * params.R * params.L / 2
        f = F_fr / max(W, 1.0)
        return W, f, h_min, p_max, converged, n_iter

    # Гладкий
    H_s = make_H(EPS, Phi, Zm, p, textured=False)
    W_s, f_s, hmin_s, pmax_s, conv_s, nit_s = solve_W(H_s)

    # Текстурированный
    H0 = make_H(EPS, Phi, Zm, p, textured=False)
    H_t = create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi, Zm, phi_c, Z_c, A, B, profile=PROFILE)
    W_t, f_t, hmin_t, pmax_t, conv_t, nit_t = solve_W(H_t)

    g = lambda a, b: a / b if b > 0 else 0
    return {
        "W_s": W_s, "W_t": W_t, "gain_W": g(W_t, W_s),
        "f_s": f_s, "f_t": f_t, "gain_f": g(f_t, f_s),
        "hmin_t": hmin_t * 1e6, "pmax_t": pmax_t / 1e6,
        "conv_s": conv_s, "conv_t": conv_t,
        "nit_s": nit_s, "nit_t": nit_t,
        "n_dimples": len(phi_c),
    }


def main():
    global PROFILE
    print("=" * 100)
    print("ТОЧЕЧНЫЕ ПРОГОНЫ ТЕКСТУРЫ НАСОСА")
    print(f"ε = {EPS}, η = {ETA} Па·с, профиль = {PROFILE}, без PV")
    print("=" * 100)

    results = []
    t0_all = time.time()

    for v in VARIANTS:
        name = v[0]
        t0 = time.time()
        sys.stdout.write(f"  {name:15s} ({v[8]}x{v[9]}) ... ")
        sys.stdout.flush()
        r = run_variant(*v)
        dt = time.time() - t0
        marker = " <<<" if r["gain_W"] > 1.03 else ""
        print(f"gain_W={r['gain_W']:.4f}  gain_f={r['gain_f']:.4f}  "
              f"({dt:.1f} с){marker}")
        results.append((v, r, dt))

    dt_all = time.time() - t0_all
    print(f"\nОбщее время: {dt_all:.0f} с")

    # Сводная таблица
    print(f"\n{'=' * 105}")
    print("СВОДНАЯ ТАБЛИЦА")
    print(f"{'=' * 105}")
    print(f"{'Вариант':<15} {'a,b мм':>8} {'h_p':>4} {'Зона':>10} {'Сетка':>10} "
          f"{'W_sm':>7} {'W_tex':>7} {'gain_W':>7} {'gain_f':>7} "
          f"{'conv':>5} {'nit_t':>6}")
    print("-" * 105)

    for v, r, dt in results:
        name = v[0]
        ab = f"{v[1]:.1f}/{v[2]:.1f}"
        zone = f"{v[4]}-{v[5]}°"
        grid = f"{v[8]}x{v[9]}"
        marker = " <<<" if r["gain_W"] > 1.03 else ""
        print(f"{name:<15} {ab:>8} {v[3]:>4} {zone:>10} {grid:>10} "
              f"{r['W_s']:>7.0f} {r['W_t']:>7.0f} {r['gain_W']:>7.4f} "
              f"{r['gain_f']:>7.4f} {str(r['conv_t']):>5} {r['nit_t']:>6}{marker}")

    # Лучший по gain_W
    best_idx = max(range(len(results)), key=lambda i: results[i][1]["gain_W"])
    best_v, best_r, _ = results[best_idx]
    print(f"\nЛучший: {best_v[0]} — gain_W={best_r['gain_W']:.4f}, gain_f={best_r['gain_f']:.4f}")

    # Мини-convergence для лучшего: если был на 2000×200, перепрогнать на 5000×400
    if best_r["gain_W"] > 1.01:
        bv = list(best_v)
        if bv[8] <= 2000:
            print(f"\n--- Мини-convergence: перепрогон {bv[0]} на 5000×400 ---")
            bv_fine = tuple(bv[:8] + [5000, 400])
            t0 = time.time()
            r_fine = run_variant(*bv_fine)
            dt = time.time() - t0
            print(f"  5000×400: gain_W={r_fine['gain_W']:.4f}  "
                  f"gain_f={r_fine['gain_f']:.4f}  ({dt:.1f} с)")
            delta_gain = abs(r_fine["gain_W"] - best_r["gain_W"]) / max(best_r["gain_W"], 1e-9) * 100
            print(f"  Δgain_W = {delta_gain:.1f}%"
                  f"  {'OK (<2%)' if delta_gain < 2 else 'Нужна тонкая сетка'}")

    # Контроль sqrt vs smoothcap для лучшего
    if best_r["gain_W"] > 1.01:
        print(f"\n--- Контроль профиля: {best_v[0]} с smoothcap ---")
        PROFILE = "smoothcap"
        t0 = time.time()
        r_sc = run_variant(*best_v)
        dt = time.time() - t0
        PROFILE = "sqrt"
        print(f"  smoothcap: gain_W={r_sc['gain_W']:.4f}  "
              f"gain_f={r_sc['gain_f']:.4f}  ({dt:.1f} с)")
        delta_p = abs(r_sc["gain_W"] - best_r["gain_W"]) / max(best_r["gain_W"], 1e-9) * 100
        print(f"  Δ от sqrt = {delta_p:.1f}%"
              f"  {'Профиль не критичен' if delta_p < 1 else 'Профиль влияет!'}")

    # Сохранить
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "texture_variants.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Точечные прогоны текстуры: ε={EPS}, {PROFILE}, без PV\n")
        f.write(f"Время: {dt_all:.0f} с\n\n")
        for v, r, dt in results:
            f.write(f"{v[0]:<15} a/b={v[1]}/{v[2]}мм h_p={v[3]}мкм "
                    f"zone={v[4]}-{v[5]}° grid={v[8]}x{v[9]} "
                    f"gain_W={r['gain_W']:.4f} gain_f={r['gain_f']:.4f}\n")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
