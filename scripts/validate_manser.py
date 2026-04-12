#!/usr/bin/env python3
"""Валидация по Manser (2019).

Два сценария:
  1. Table 3 (Tala-Ighil parameters): cylindrical dimples,
     Reynolds BC (half_sommerfeld), поиск равновесия для F=12600Н
  2. Fig. 20 тренды (Manser parameters): square dimples, JFO (PS),
     фиксированный ε=0.6

Координаты Manser:
  θ = x/R ∈ [0, 2π]  = наш φ
  Z_M = z/L ∈ [0, 1] → наш Z ∈ [-1, 1]: Z_нам = 2·Z_M - 1

Solver: R=1, L=4 (так как α² = (R/L)² = 0.25).
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu

from reynolds_solver import solve_reynolds

R_SOLVER = 1.0
L_SOLVER = 2.0   # α² = (2R/L)² = 1.0 для L/D=1


# ─── Сценарий 1: Tala-Ighil parameters ────────────────────────────
S1_R = 0.0315      # м
S1_L = 0.063       # м
S1_C = 30e-6       # м
S1_OMEGA = 625.4   # рад/с
S1_MU = 0.0035     # Па·с
S1_F = 12600.0     # Н
S1_P_SCALE = 6 * S1_MU * S1_OMEGA * S1_R**2 / S1_C**2

# Цилиндрические лунки
S1_DIMPLE_R = 1.0e-3     # радиус лунки (м)
S1_DIMPLE_DEPTH = 15e-6  # глубина (м)

# ─── Сценарий 2: Manser parameters ────────────────────────────────
S2_R = 0.020
S2_L = 0.040
S2_C = 50e-6
S2_OMEGA = 2 * np.pi * 3000 / 60
S2_MU = 0.05
S2_EPS = 0.6
S2_P_SCALE = 6 * S2_MU * S2_OMEGA * S2_R**2 / S2_C**2

# Квадратные лунки
S2_DX = 6e-3          # сторона по x (м)
S2_DZ = 6e-3          # сторона по z (м)
S2_RY = 25e-6         # глубина (м)


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_H_smooth(eps, Phi):
    return 1.0 + eps * np.cos(Phi)


def add_cylindrical_dimples(H, Phi, Zm, centers_phi, centers_Z,
                            r_phi, r_Z, depth):
    """Цилиндрические лунки (круглые в физических координатах).

    r_phi — радиус в радианах по φ, r_Z — радиус в безразмерных Z.
    Depth — безразмерная глубина (добавляется к H где внутри круга).
    """
    H_out = H.copy()
    for phi_c, Z_c in zip(centers_phi, centers_Z):
        dphi = np.abs(Phi - phi_c)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dz = np.abs(Zm - Z_c)
        # Эллипс: (dphi/r_phi)² + (dz/r_Z)² < 1
        mask = (dphi / r_phi)**2 + (dz / r_Z)**2 < 1
        H_out[mask] += depth
    return H_out


def add_square_dimples(H, Phi, Zm, centers_phi, centers_Z,
                       a_phi, a_Z, depth):
    """Квадратные ступенчатые лунки (половины сторон)."""
    H_out = H.copy()
    for phi_c, Z_c in zip(centers_phi, centers_Z):
        dphi = np.abs(Phi - phi_c)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dz = np.abs(Zm - Z_c)
        mask = (dphi < a_phi) & (dz < a_Z)
        H_out[mask] += depth
    return H_out


def make_dimple_centers(N_phi_tex, N_Z_tex, phi_start_deg=0, phi_end_deg=360):
    """Равномерная раскладка центров."""
    phi_s = np.deg2rad(phi_start_deg)
    phi_e = np.deg2rad(phi_end_deg)
    if N_phi_tex == 1:
        phi_c = np.array([(phi_s + phi_e) / 2])
    else:
        phi_c = np.linspace(phi_s, phi_e, N_phi_tex, endpoint=False)
        # Сдвиг так чтобы лунки были в середине ячеек
        phi_c += (phi_e - phi_s) / (2 * N_phi_tex)
    if N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        # Z_M ∈ [0, 1] → Z ∈ [-1, 1]
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


def solve_ps(H, dp, dz):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, R_SOLVER, L_SOLVER, tol=1e-6, max_iter=10_000_000)
    return P, theta


def compute_metrics(P, H, Phi, phi_1D, Z_1D, theta=None):
    """Интегральные характеристики (безразмерные, Manser coordinates)."""
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]

    # Load: WX, WY в безразмерных единицах
    # Integral domain: θ ∈ [0,2π], Z_M ∈ [0,1] → Z ∈ [-1,1]
    # dA_M = dθ · dZ_M, dZ_M = dZ/2. Scale = 1/2.
    scale = 0.5
    WX = np.trapezoid(np.trapezoid(P * np.cos(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    WY = np.trapezoid(np.trapezoid(P * np.sin(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    W = np.sqrt(WX**2 + WY**2)

    # Угол линии центров
    # Нагрузка вертикальная (−Y): attitude angle φ_att = atan2(WX, WY)
    phi_att = np.rad2deg(np.arctan2(WX, WY))

    P_max = np.max(P)

    return WX, WY, W, phi_att, P_max


def find_equilibrium_hs(W_target, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                         textured_fn=None, max_iter=40, tol=1e-4):
    """Newton-Raphson: ищет (X, Y) для заданной нагрузки (вертикальная −Y).

    W_hydro.cos(phi_att) направлено вдоль −Y при правильном равновесии.
    Условие: WX = 0, WY = W_target.
    """
    X, Y = 0.0, -0.6  # ~ε=0.6
    dXY = 1e-5

    for it in range(max_iter):
        H = 1.0 + X * np.cos(Phi) + Y * np.sin(Phi)
        if textured_fn is not None:
            H = textured_fn(H)

        P = solve_hs(H, d_phi, d_Z)
        WX, WY, W, phi_att, P_max = compute_metrics(P, H, Phi, phi_1D, Z_1D)

        Rx = WX
        Ry = WY - W_target
        err = np.sqrt(Rx**2 + Ry**2) / max(W_target, 1e-9)

        if err < tol:
            eps = np.sqrt(X**2 + Y**2)
            return X, Y, eps, W, phi_att, P_max, P, H, it + 1

        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0), (0, dXY)]):
            H_p = 1.0 + (X + dX_) * np.cos(Phi) + (Y + dY_) * np.sin(Phi)
            if textured_fn is not None:
                H_p = textured_fn(H_p)
            Pp = solve_hs(H_p, d_phi, d_Z)
            WXp, WYp, _, _, _ = compute_metrics(Pp, H_p, Phi, phi_1D, Z_1D)
            J[0, col] = (WXp - WX) / dXY
            J[1, col] = (WYp - WY) / dXY

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-20:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det
        step = min(1.0, 0.2 / max(abs(dX), abs(dY), 1e-10))
        X += step * dX
        Y += step * dY

    eps = np.sqrt(X**2 + Y**2)
    return X, Y, eps, W, phi_att, P_max, P, H, max_iter


# ===================================================================
#  Сценарий 1: Table 3
# ===================================================================

def run_scenario_1(out_dir):
    print(f"\n{'=' * 80}")
    print(f"СЦЕНАРИЙ 1: Tala-Ighil (Manser Table 3)")
    print(f"R={S1_R*1e3:.1f}мм, L={S1_L*1e3:.1f}мм, C={S1_C*1e6:.0f}мкм")
    print(f"F={S1_F:.0f}Н, Reynolds BC (HS)")
    print(f"p_scale = {S1_P_SCALE/1e6:.2f} МПа")
    print(f"{'=' * 80}")

    W_target = S1_F / (S1_P_SCALE * S1_R * S1_L)
    print(f"  W̄_target = {W_target:.4f}")

    N_phi, N_Z = 450, 70
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)

    # Безразмерные размеры лунки
    r_phi = S1_DIMPLE_R / S1_R           # рад
    r_Z = S1_DIMPLE_R / (S1_L / 2)       # безразмерный
    depth = S1_DIMPLE_DEPTH / S1_C       # = 0.5

    # Расстояние между лунками = 2*r_phi (плотная упаковка)
    N_phi_tex_full = int(2 * np.pi / (2.2 * r_phi))
    N_Z_tex_full = int(2.0 / (2.2 * r_Z))

    # Полная текстура
    phi_c_full, Z_c_full = make_dimple_centers(N_phi_tex_full, N_Z_tex_full)

    def tex_full(H):
        return add_cylindrical_dimples(H, Phi, Zm, phi_c_full, Z_c_full,
                                        r_phi, r_Z, depth)

    # Partial (180°-360°)
    N_phi_tex_partial = N_phi_tex_full // 2
    phi_c_part, Z_c_part = make_dimple_centers(
        N_phi_tex_partial, N_Z_tex_full, 180, 360)

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
        X, Y, eps, W, phi_att, P_max, P, H, nit = find_equilibrium_hs(
            W_target, Phi, Zm, phi, Z, dp, dz, textured_fn=tex_fn)
        dt = time.time() - t0

        hmin = S1_C * (1 - eps) * 1e6  # мкм
        P_max_MPa = P_max * S1_P_SCALE / 1e6
        Q_marker = 0.0  # упрощённо — пока не считаем

        print(f"    ε={eps:.4f}, h_min={hmin:.2f}мкм, "
              f"P_max={P_max_MPa:.2f}МПа, φ={phi_att:.1f}°, "
              f"nit={nit}, {dt:.1f}с")
        results.append({
            "name": name, "eps": eps, "hmin": hmin,
            "Pmax": P_max_MPa, "phi_att": phi_att,
        })

    # Acceptance
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

    # Checklist
    print(f"\n  CHECKLIST:")
    r_s = results[0]
    r_f = results[1]
    r_p = results[2]
    print(f"  [{'✓' if abs(r_s['eps'] - 0.601)/0.601 < 0.05 else '✗'}] "
          f"Smooth ε≈0.601 (got {r_s['eps']:.3f})")
    print(f"  [{'✓' if abs(r_s['Pmax'] - 7.71)/7.71 < 0.10 else '✗'}] "
          f"Smooth Pmax≈7.71 (got {r_s['Pmax']:.2f})")
    print(f"  [{'✓' if abs(r_s['phi_att'] - 50.5)/50.5 < 0.10 else '✗'}] "
          f"Smooth φ≈50.5° (got {r_s['phi_att']:.1f})")
    print(f"  [{'✓' if r_f['eps'] > r_s['eps'] else '✗'}] "
          f"Full ε > Smooth ε")
    print(f"  [{'✓' if r_p['eps'] < r_s['eps'] else '✗'}] "
          f"Partial ε < Smooth ε")
    print(f"  [{'✓' if r_p['hmin'] > r_s['hmin'] else '✗'}] "
          f"Partial hmin > Smooth hmin")


# ===================================================================
#  Сценарий 2: Fig. 20 trends
# ===================================================================

def run_scenario_2(out_dir):
    print(f"\n{'=' * 80}")
    print(f"СЦЕНАРИЙ 2: Manser JFO (Fig. 20 trends)")
    print(f"R={S2_R*1e3:.0f}мм, L={S2_L*1e3:.0f}мм, C={S2_C*1e6:.0f}мкм, "
          f"ε={S2_EPS}")
    print(f"p_scale = {S2_P_SCALE/1e6:.2f} МПа")
    print(f"{'=' * 80}")

    N_phi, N_Z = 421, 121
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)

    # Квадратные лунки: dx=dz=6мм, ry=25мкм
    # Полуширина в безразмерных:
    a_phi = (S2_DX / 2) / S2_R         # = 3e-3/20e-3 = 0.15 рад
    a_Z = (S2_DZ / 2) / (S2_L / 2)     # = 3e-3/20e-3 = 0.15
    depth = S2_RY / S2_C               # = 0.5

    # Smooth
    H_s = make_H_smooth(S2_EPS, Phi)
    t0 = time.time()
    P_s, theta_s = solve_ps(H_s, dp, dz)
    dt_s = time.time() - t0
    WX_s, WY_s, W_s, phi_s, Pmax_s = compute_metrics(P_s, H_s, Phi, phi, Z, theta_s)
    print(f"\n  Smooth PS: W={W_s:.4f}, P_max={Pmax_s:.4f} "
          f"({Pmax_s * S2_P_SCALE / 1e6:.2f}МПа), {dt_s:.1f}с")

    # Full 16×5
    phi_c_f, Z_c_f = make_dimple_centers(16, 5)
    H_f = add_square_dimples(H_s, Phi, Zm, phi_c_f, Z_c_f, a_phi, a_Z, depth)
    t0 = time.time()
    P_f, theta_f = solve_ps(H_f, dp, dz)
    dt_f = time.time() - t0
    WX_f, WY_f, W_f, phi_f, Pmax_f = compute_metrics(P_f, H_f, Phi, phi, Z, theta_f)
    print(f"  Full tex PS: W={W_f:.4f} ({W_f/W_s:.3f}×smooth), "
          f"P_max={Pmax_f:.4f} ({Pmax_f/Pmax_s:.3f}×smooth), {dt_f:.1f}с")

    # Partial 8×5 в зоне 180-360°
    phi_c_p, Z_c_p = make_dimple_centers(8, 5, 180, 360)
    H_p = add_square_dimples(H_s, Phi, Zm, phi_c_p, Z_c_p, a_phi, a_Z, depth)
    t0 = time.time()
    P_p, theta_p = solve_ps(H_p, dp, dz)
    dt_p = time.time() - t0
    WX_p, WY_p, W_p, phi_p, Pmax_p = compute_metrics(P_p, H_p, Phi, phi, Z, theta_p)
    print(f"  Partial tex PS: W={W_p:.4f} ({W_p/W_s:.3f}×smooth), "
          f"P_max={Pmax_p:.4f} ({Pmax_p/Pmax_s:.3f}×smooth), {dt_p:.1f}с")

    # Графики midplane
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

    # Checklist
    print(f"\n  CHECKLIST (Manser Fig. 20 trends):")
    print(f"  [{'✓' if Pmax_f < Pmax_s else '✗'}] "
          f"Full: P_max_tex < P_max_smooth "
          f"({Pmax_f:.4f} vs {Pmax_s:.4f})")
    print(f"  [{'✓' if Pmax_p >= 0.95 * Pmax_s else '✗'}] "
          f"Partial: P_max ≥ smooth "
          f"({Pmax_p:.4f} vs {Pmax_s:.4f})")
    print(f"  [{'✓' if W_f < W_s else '✗'}] "
          f"Full: W < W_smooth")
    print(f"  [{'✓' if W_p >= 0.95 * W_s else '✗'}] "
          f"Partial: W ≥ W_smooth")

    print(f"\n  Gain_W: full={W_f/W_s:.4f}, partial={W_p/W_s:.4f}")
    print(f"  Expected Manser: full ≈ 0.44 (drop), partial ≈ 1.05 (gain)")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "validation_manser")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ВАЛИДАЦИЯ PS vs MANSER (2019)")
    print(f"Результаты → {out_dir}")
    print("=" * 80)

    run_scenario_1(out_dir)
    run_scenario_2(out_dir)


if __name__ == "__main__":
    main()
