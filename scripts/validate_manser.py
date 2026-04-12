#!/usr/bin/env python3
"""Валидация по Manser (2019).

Scenario 1 (Table 3): Tala-Ighil params, cylindrical dimples, Reynolds BC.
  Equilibrium search: 1D brentq по ε, φ вычисляется.

Scenario 2 (Fig. 20 trends): Manser params, square dimples, JFO (PS),
  фиксированный ε=0.6.

NOTE: точная раскладка лунок Tala-Ighil [52] не выписана явно в Manser.
Используем приближённую реконструкцию (rx=rz=1мм, 10×10 узлов на лунку).
Table 3 — semi-quantitative benchmark.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu

from reynolds_solver import solve_reynolds

R_SOLVER = 1.0
L_SOLVER = 2.0   # α² = (2R/L)² = 1.0 для L/D=1

# ─── Scenario 1: Tala-Ighil ────────────────────────────────────────
S1_R = 0.0315
S1_L = 0.063
S1_C = 30e-6
S1_OMEGA = 625.4
S1_MU = 0.0035
S1_F = 12600.0
S1_P_SCALE = 6 * S1_MU * S1_OMEGA * S1_R**2 / S1_C**2

S1_DIMPLE_R = 1.0e-3
S1_DIMPLE_DEPTH = 15e-6

# ─── Scenario 2: Manser ───────────────────────────────────────────
S2_R = 0.020
S2_L = 0.040
S2_C = 50e-6
S2_OMEGA = 2 * np.pi * 3000 / 60
S2_MU = 0.05
S2_EPS = 0.6
S2_P_SCALE = 6 * S2_MU * S2_OMEGA * S2_R**2 / S2_C**2

S2_DX = 6e-3
S2_DZ = 6e-3
S2_RY = 25e-6


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def add_cylindrical_dimples(H, Phi, Zm, centers_phi, centers_Z,
                            r_phi, r_Z, depth):
    H_out = H.copy()
    for phi_c, Z_c in zip(centers_phi, centers_Z):
        dphi = np.abs(Phi - phi_c)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dz = np.abs(Zm - Z_c)
        mask = (dphi / r_phi)**2 + (dz / r_Z)**2 < 1
        H_out[mask] += depth
    return H_out


def add_square_dimples(H, Phi, Zm, centers_phi, centers_Z,
                       a_phi, a_Z, depth):
    H_out = H.copy()
    for phi_c, Z_c in zip(centers_phi, centers_Z):
        dphi = np.abs(Phi - phi_c)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dz = np.abs(Zm - Z_c)
        mask = (dphi < a_phi) & (dz < a_Z)
        H_out[mask] += depth
    return H_out


def make_dimple_centers(N_phi_tex, N_Z_tex, phi_start_deg=0, phi_end_deg=360):
    phi_s = np.deg2rad(phi_start_deg)
    phi_e = np.deg2rad(phi_end_deg)
    if N_phi_tex == 1:
        phi_c = np.array([(phi_s + phi_e) / 2])
    else:
        phi_c = np.linspace(phi_s, phi_e, N_phi_tex, endpoint=False)
        phi_c += (phi_e - phi_s) / (2 * N_phi_tex)
    if N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_M_centers = (np.arange(N_Z_tex) + 0.5) / N_Z_tex
        Z_c = 2 * Z_M_centers - 1
    pg, zg = np.meshgrid(phi_c, Z_c)
    return pg.ravel(), zg.ravel()


def solve_hs(H, dp, dz):
    P, _, _, _ = solve_reynolds(
        H, dp, dz, R_SOLVER, L_SOLVER,
        closure="laminar", cavitation="half_sommerfeld",
        tol=1e-5, max_iter=1_000_000,
        return_converged=True)
    return P


def solve_ps(H, dp, dz, hs_warmup_iter=200_000, hs_warmup_omega=1.5):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, R_SOLVER, L_SOLVER, tol=1e-6, max_iter=10_000_000,
        hs_warmup_iter=hs_warmup_iter,
        hs_warmup_omega=hs_warmup_omega)
    return P, theta


def compute_metrics(P, H, Phi, phi_1D, Z_1D):
    """Force components (Manser eq. 15-16).

    W_theta = -∫∫ p·cos(θ) dA  (horizontal)
    W_Z     = +∫∫ p·sin(θ) dA  (vertical)
    φ_att = atan2(W_theta, W_Z)
    dA = dθ · dZ_M, dZ_M = dZ/2 → scale = 0.5
    """
    scale = 0.5
    W_theta = -np.trapezoid(np.trapezoid(P * np.cos(Phi), phi_1D, axis=1),
                             Z_1D, axis=0) * scale
    W_Z = np.trapezoid(np.trapezoid(P * np.sin(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    W = np.sqrt(W_theta**2 + W_Z**2)
    phi_att = np.rad2deg(np.arctan2(W_theta, W_Z))
    P_max = np.max(P)
    return W, phi_att, P_max, W_theta, W_Z


def solve_at_eps(eps, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                 texture_fn=None):
    """Решить Reynolds при заданном ε."""
    H = 1.0 + eps * np.cos(Phi)
    if texture_fn is not None:
        H = texture_fn(H)
    P = solve_hs(H, d_phi, d_Z)
    W, phi_att, P_max, W_theta, W_Z = compute_metrics(
        P, H, Phi, phi_1D, Z_1D)
    hmin = np.min(H)  # безразмерный
    return W, phi_att, P_max, hmin, P, H


def find_equilibrium_1d(W_target, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                         texture_fn=None, eps_lo=0.1, eps_hi=0.95):
    """1D brentq: W(ε) = W_target."""
    def residual(eps):
        W, _, _, _, _, _ = solve_at_eps(
            eps, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, texture_fn)
        return W - W_target

    try:
        eps_eq = brentq(residual, eps_lo, eps_hi, xtol=1e-4)
    except ValueError as e:
        # Границы не меняют знак — возвратим ближайшую
        W_lo, _, _, _, _, _ = solve_at_eps(
            eps_lo, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, texture_fn)
        W_hi, _, _, _, _, _ = solve_at_eps(
            eps_hi, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, texture_fn)
        print(f"    WARN: W({eps_lo})={W_lo:.4f}, W({eps_hi})={W_hi:.4f}, "
              f"target={W_target:.4f}")
        eps_eq = eps_hi if abs(W_hi - W_target) < abs(W_lo - W_target) else eps_lo

    W, phi_att, P_max, hmin, P, H = solve_at_eps(
        eps_eq, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, texture_fn)
    return eps_eq, W, phi_att, P_max, hmin, P, H


# ===================================================================
#  Сценарий 1
# ===================================================================

def run_scenario_1(out_dir):
    print(f"\n{'=' * 80}")
    print(f"СЦЕНАРИЙ 1: Tala-Ighil (Manser Table 3)")
    print(f"R={S1_R*1e3:.1f}мм, L={S1_L*1e3:.1f}мм, C={S1_C*1e6:.0f}мкм")
    print(f"F={S1_F:.0f}Н, Reynolds BC")
    print(f"NOTE: точный layout Tala-Ighil [52] не восстановлен из Manser.")
    print(f"      Используется приближённая реконструкция.")
    print(f"{'=' * 80}")

    W_target = S1_F / (S1_P_SCALE * S1_R * S1_L)
    print(f"  W̄_target = {W_target:.4f}, p_scale = {S1_P_SCALE/1e6:.2f} МПа")

    N_phi, N_Z = 450, 70
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)

    r_phi = S1_DIMPLE_R / S1_R
    r_Z = S1_DIMPLE_R / (S1_L / 2)
    depth = S1_DIMPLE_DEPTH / S1_C

    # Приближённая раскладка (N_tex из 891/10 и 142/10)
    N_phi_tex = 89
    N_Z_tex = 14

    phi_c_full, Z_c_full = make_dimple_centers(N_phi_tex, N_Z_tex)

    def tex_full(H):
        return add_cylindrical_dimples(H, Phi, Zm, phi_c_full, Z_c_full,
                                        r_phi, r_Z, depth)

    N_phi_tex_partial = N_phi_tex // 2
    phi_c_part, Z_c_part = make_dimple_centers(
        N_phi_tex_partial, N_Z_tex, 180, 360)

    def tex_partial(H):
        return add_cylindrical_dimples(H, Phi, Zm, phi_c_part, Z_c_part,
                                        r_phi, r_Z, depth)

    configs = [
        ("Smooth", None),
        ("Full textured", tex_full),
        ("Partial textured", tex_partial),
    ]

    results = []
    for name, tex_fn in configs:
        print(f"\n  {name}:")
        t0 = time.time()
        eps, W, phi_att, P_max, hmin, P, H = find_equilibrium_1d(
            W_target, Phi, Zm, phi, Z, dp, dz, texture_fn=tex_fn)
        dt = time.time() - t0

        hmin_um = hmin * S1_C * 1e6
        P_max_MPa = P_max * S1_P_SCALE / 1e6
        print(f"    ε={eps:.4f}, h_min={hmin_um:.2f}мкм, "
              f"P_max={P_max_MPa:.2f}МПа, φ={phi_att:.1f}°, {dt:.1f}с")
        results.append({
            "name": name, "eps": eps, "hmin": hmin_um,
            "Pmax": P_max_MPa, "phi_att": phi_att,
        })

    print(f"\n  {'=' * 60}")
    print(f"  REFERENCE (Manser Table 3):")
    print(f"  {'Config':<20s} {'ε':>6} {'hmin':>7} {'Pmax':>7} {'φ':>6}")
    print(f"  {'Smooth':<20s} {0.601:>6.3f} {11.97:>7.2f} {7.71:>7.2f} {50.5:>6.1f}")
    print(f"  {'Full textured':<20s} {0.709:>6.3f} {8.71:>7.2f} {8.26:>7.2f} {46.1:>6.1f}")
    print(f"  {'Partial textured':<20s} {0.595:>6.3f} {12.16:>7.2f} {7.58:>7.2f} {49.3:>6.1f}")

    print(f"\n  OUR:")
    print(f"  {'Config':<20s} {'ε':>6} {'hmin':>7} {'Pmax':>7} {'φ':>6}")
    for r in results:
        print(f"  {r['name']:<20s} {r['eps']:>6.3f} "
              f"{r['hmin']:>7.2f} {r['Pmax']:>7.2f} {r['phi_att']:>6.1f}")

    r_s, r_f, r_p = results
    print(f"\n  CHECKLIST (semi-quantitative):")
    print(f"  [{'✓' if abs(r_s['eps'] - 0.601) / 0.601 < 0.10 else '✗'}] "
          f"Smooth ε≈0.601 (got {r_s['eps']:.3f})")
    print(f"  [{'✓' if abs(r_s['Pmax'] - 7.71) / 7.71 < 0.15 else '✗'}] "
          f"Smooth Pmax≈7.71 (got {r_s['Pmax']:.2f})")
    print(f"  [{'✓' if abs(r_s['phi_att'] - 50.5) / 50.5 < 0.15 else '✗'}] "
          f"Smooth φ≈50.5° (got {r_s['phi_att']:.1f})")
    print(f"  [{'✓' if r_f['eps'] > r_s['eps'] else '✗'}] Full ε > Smooth ε")
    print(f"  [{'✓' if r_p['eps'] < r_s['eps'] else '✗'}] Partial ε < Smooth ε")
    print(f"  [{'✓' if r_p['hmin'] > r_s['hmin'] else '✗'}] "
          f"Partial hmin > Smooth hmin")


# ===================================================================
#  Сценарий 2
# ===================================================================

def run_scenario_2(out_dir):
    print(f"\n{'=' * 80}")
    print(f"СЦЕНАРИЙ 2: Manser JFO (Fig. 20 trends)")
    print(f"R={S2_R*1e3:.0f}мм, L={S2_L*1e3:.0f}мм, C={S2_C*1e6:.0f}мкм, "
          f"ε={S2_EPS}")
    print(f"{'=' * 80}")

    N_phi, N_Z = 421, 121
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)

    a_phi = (S2_DX / 2) / S2_R
    a_Z = (S2_DZ / 2) / (S2_L / 2)
    depth = S2_RY / S2_C

    H_s = 1.0 + S2_EPS * np.cos(Phi)

    # Smooth
    t0 = time.time()
    P_s, theta_s = solve_ps(H_s, dp, dz)
    dt_s = time.time() - t0
    W_s, phi_s_att, Pmax_s, _, _ = compute_metrics(P_s, H_s, Phi, phi, Z)
    print(f"\n  Smooth PS: W={W_s:.4f}, P_max={Pmax_s:.4f} "
          f"({Pmax_s * S2_P_SCALE / 1e6:.2f}МПа), {dt_s:.1f}с")

    # Full 16×5
    phi_c_f, Z_c_f = make_dimple_centers(16, 5)
    H_f = add_square_dimples(H_s, Phi, Zm, phi_c_f, Z_c_f,
                              a_phi, a_Z, depth)
    t0 = time.time()
    P_f, theta_f = solve_ps(H_f, dp, dz)
    dt_f = time.time() - t0
    W_f, _, Pmax_f, _, _ = compute_metrics(P_f, H_f, Phi, phi, Z)
    print(f"  Full tex PS: W={W_f:.4f} ({W_f/W_s:.3f}×smooth), "
          f"P_max={Pmax_f:.4f} ({Pmax_f/Pmax_s:.3f}×smooth), {dt_f:.1f}с")

    # Partial 8×5
    phi_c_p, Z_c_p = make_dimple_centers(8, 5, 180, 360)
    H_p = add_square_dimples(H_s, Phi, Zm, phi_c_p, Z_c_p,
                              a_phi, a_Z, depth)
    t0 = time.time()
    P_p, theta_p = solve_ps(H_p, dp, dz)
    dt_p = time.time() - t0
    W_p, _, Pmax_p, _, _ = compute_metrics(P_p, H_p, Phi, phi, Z)
    print(f"  Partial tex PS: W={W_p:.4f} ({W_p/W_s:.3f}×smooth), "
          f"P_max={Pmax_p:.4f} ({Pmax_p/Pmax_s:.3f}×smooth), {dt_p:.1f}с")

    iz = N_Z // 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(phi, P_s[iz, :], "k-", lw=2, label="Smooth")
    ax.plot(phi, P_f[iz, :], "b--", lw=1.5, label="Full texture")
    ax.plot(phi, P_p[iz, :], "r-", lw=1.5, label="Partial texture")
    ax.set_xlabel("θ (рад)")
    ax.set_ylabel("P (безразмерное)")
    ax.set_title("Midplane P(θ), ε=0.6 (cf. Manser Fig. 20)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "manser_fig20_midplane.png"), dpi=300)
    plt.close(fig)

    print(f"\n  CHECKLIST (Fig. 20 trends):")
    print(f"  [{'✓' if Pmax_f < Pmax_s else '✗'}] Full: P_max_tex < P_max_smooth")
    print(f"  [{'✓' if Pmax_p >= 0.95 * Pmax_s else '✗'}] Partial: P_max ≥ smooth")
    print(f"  [{'✓' if W_f < W_s else '✗'}] Full: W < W_smooth")
    print(f"  [{'✓' if W_p >= 0.95 * W_s else '✗'}] Partial: W ≥ W_smooth")
    print(f"\n  Gain_W: full={W_f/W_s:.4f}, partial={W_p/W_s:.4f}")
    print(f"  Expected Manser: full ≈ 0.44, partial ≈ 1.05")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "validation_manser")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ВАЛИДАЦИЯ PS vs MANSER (2019)")
    print(f"Результаты → {out_dir}")
    print("=" * 80)

    # Scenario 1 временно отключён: точный layout Tala-Ighil не восстановлен
    # run_scenario_1(out_dir)
    run_scenario_2(out_dir)


if __name__ == "__main__":
    main()
