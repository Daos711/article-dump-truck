#!/usr/bin/env python3
"""Диагностика micro-wedge эффекта T2/T3.

3 быстрых теста для проверки гипотезы "PS убивает клин":
1. HS vs PS на T2/T3 full — сравнение эффекта модели кавитации
2. Локальный zoom одной лунки — видно ли раннюю кавитацию на входе
3. Одна T2/T3 лунка в зоне максимума — работает ли клин локально

Параметры: Manser (D=L=40мм, C=50мкм, ε=0.6, r_x=r_z=3мм, r_y=15мкм,
Ptex=40%, full 0-360°, phi_bc='groove', сетка 441×121).
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
L_SOLVER = 2.0

R_M = 0.020
L_M = 0.040
C_M = 50e-6

RX = 3.0e-3
RZ = 3.0e-3
RY = 15e-6

N_PHI = 441
N_Z = 121
EPS = 0.6
NCTH = 14
NCZ = 4


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_H_smooth(eps, Phi):
    return 1.0 + eps * np.cos(Phi)


def make_dimple_centers(N_phi_tex, N_Z_tex, phi_start_deg, phi_end_deg,
                         a_phi_margin=0.0):
    phi_s = np.deg2rad(phi_start_deg) + a_phi_margin
    phi_e = np.deg2rad(phi_end_deg) - a_phi_margin
    if phi_e <= phi_s:
        return np.array([]), np.array([])
    if N_phi_tex == 1:
        phi_c = np.array([(phi_s + phi_e) / 2])
    else:
        phi_c = np.linspace(phi_s, phi_e, N_phi_tex)
    if N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_margin = 0.1
        Z_c = np.linspace(-1 + Z_margin, 1 - Z_margin, N_Z_tex)
    pg, zg = np.meshgrid(phi_c, Z_c)
    return pg.ravel(), zg.ravel()


def add_wedge_dimples(H, Phi, Zm, centers_phi, centers_Z,
                      a_phi, a_Z, depth, direction="convergent"):
    H_out = H.copy()
    for phi_c, Z_c in zip(centers_phi, centers_Z):
        dphi = Phi - phi_c
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dz = np.abs(Zm - Z_c)
        mask = (np.abs(dphi) < a_phi) & (dz < a_Z)
        x_local = dphi[mask] / a_phi
        if direction == "convergent":
            H_out[mask] += depth * (1.0 - x_local)
        elif direction == "divergent":
            H_out[mask] += depth * (1.0 + x_local)
        elif direction == "flat":
            H_out[mask] += depth
    return H_out


def solve_ps(H, dp, dz, phi_bc="groove"):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, R_SOLVER, L_SOLVER, tol=1e-6, max_iter=10_000_000,
        hs_warmup_iter=200_000, hs_warmup_omega=1.5, phi_bc=phi_bc)
    return P, theta


def solve_hs(H, dp, dz):
    P, _, _, _ = solve_reynolds(
        H, dp, dz, R_SOLVER, L_SOLVER,
        closure="laminar", cavitation="half_sommerfeld",
        tol=1e-5, max_iter=1_000_000, return_converged=True)
    return P


def compute_W(P, Phi, phi_1D, Z_1D):
    scale = 0.5
    W_th = -np.trapezoid(np.trapezoid(P * np.cos(Phi), phi_1D, axis=1),
                          Z_1D, axis=0) * scale
    W_Z = np.trapezoid(np.trapezoid(P * np.sin(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    return np.sqrt(W_th**2 + W_Z**2)


# ===================================================================
#  Тест 1: HS vs PS на T2/T3
# ===================================================================

def test1_hs_vs_ps(out_dir):
    print("\n" + "=" * 80)
    print("ТЕСТ 1: Half-Sommerfeld vs Payvar-Salant на T2/T3 full")
    print("=" * 80)

    phi, Z, Phi, Zm, dp, dz = make_grid(N_PHI, N_Z)
    a_phi = RX / R_M
    a_Z = RZ / (L_M / 2)
    depth = RY / C_M

    H_s = make_H_smooth(EPS, Phi)
    phi_c, Z_c = make_dimple_centers(NCTH, NCZ, 0, 360,
                                       a_phi_margin=a_phi)
    H_T2 = add_wedge_dimples(H_s, Phi, Zm, phi_c, Z_c,
                              a_phi, a_Z, depth, direction="convergent")
    H_T3 = add_wedge_dimples(H_s, Phi, Zm, phi_c, Z_c,
                              a_phi, a_Z, depth, direction="divergent")

    results = {}

    # HS (half_sommerfeld не поддерживает groove — использует periodic)
    print("\n  Half-Sommerfeld (periodic BC):")
    P_hs_s = solve_hs(H_s, dp, dz)
    W_hs_s = compute_W(P_hs_s, Phi, phi, Z)
    print(f"    Smooth: W = {W_hs_s:.4f}, Pmax = {np.max(P_hs_s):.4f}")
    results["HS_s"] = W_hs_s

    P_hs_T2 = solve_hs(H_T2, dp, dz)
    W_hs_T2 = compute_W(P_hs_T2, Phi, phi, Z)
    print(f"    T2:     W = {W_hs_T2:.4f}, gain = {W_hs_T2/W_hs_s:.4f}, "
          f"Pmax = {np.max(P_hs_T2):.4f}")
    results["HS_T2"] = W_hs_T2

    P_hs_T3 = solve_hs(H_T3, dp, dz)
    W_hs_T3 = compute_W(P_hs_T3, Phi, phi, Z)
    print(f"    T3:     W = {W_hs_T3:.4f}, gain = {W_hs_T3/W_hs_s:.4f}, "
          f"Pmax = {np.max(P_hs_T3):.4f}")
    results["HS_T3"] = W_hs_T3

    # PS groove
    print("\n  Payvar-Salant (groove BC):")
    P_ps_s, _ = solve_ps(H_s, dp, dz)
    W_ps_s = compute_W(P_ps_s, Phi, phi, Z)
    print(f"    Smooth: W = {W_ps_s:.4f}, Pmax = {np.max(P_ps_s):.4f}")
    results["PS_s"] = W_ps_s

    P_ps_T2, theta_T2 = solve_ps(H_T2, dp, dz)
    W_ps_T2 = compute_W(P_ps_T2, Phi, phi, Z)
    print(f"    T2:     W = {W_ps_T2:.4f}, gain = {W_ps_T2/W_ps_s:.4f}, "
          f"Pmax = {np.max(P_ps_T2):.4f}")
    results["PS_T2"] = W_ps_T2

    P_ps_T3, theta_T3 = solve_ps(H_T3, dp, dz)
    W_ps_T3 = compute_W(P_ps_T3, Phi, phi, Z)
    print(f"    T3:     W = {W_ps_T3:.4f}, gain = {W_ps_T3/W_ps_s:.4f}, "
          f"Pmax = {np.max(P_ps_T3):.4f}")
    results["PS_T3"] = W_ps_T3

    # Сводка
    print("\n" + "-" * 60)
    print(f"{'':12s} {'HS':>15s} {'PS':>15s}")
    print(f"{'Smooth W':12s} {W_hs_s:>15.4f} {W_ps_s:>15.4f}")
    print(f"{'T2 gain':12s} {W_hs_T2/W_hs_s:>15.4f} {W_ps_T2/W_ps_s:>15.4f}")
    print(f"{'T3 gain':12s} {W_hs_T3/W_hs_s:>15.4f} {W_ps_T3/W_ps_s:>15.4f}")
    hs_ratio = W_hs_T2 / W_hs_T3 if W_hs_T3 > 0 else 0
    ps_ratio = W_ps_T2 / W_ps_T3 if W_ps_T3 > 0 else 0
    print(f"{'T2/T3':12s} {hs_ratio:>15.4f} {ps_ratio:>15.4f}")

    print(f"\n  Ожидание: HS T2/T3 >> PS T2/T3 → PS убивает клин.")
    print(f"  Факт: HS={hs_ratio:.4f}, PS={ps_ratio:.4f}")
    if hs_ratio / max(ps_ratio, 1e-9) > 2:
        print(f"  ✓ Гипотеза ПОДТВЕРЖДЕНА — расхождение в {hs_ratio/ps_ratio:.1f}×")
    else:
        print(f"  ✗ Гипотеза НЕ подтверждена — оба режима одинаково душат клин")

    # Сохранить поля для теста 2
    return dict(phi=phi, Z=Z, Phi=Phi, Zm=Zm,
                H_T2=H_T2, H_T3=H_T3,
                P_ps_s=P_ps_s, P_ps_T2=P_ps_T2, P_ps_T3=P_ps_T3,
                theta_T2=theta_T2, theta_T3=theta_T3,
                phi_c=phi_c, Z_c=Z_c, a_phi=a_phi, a_Z=a_Z)


# ===================================================================
#  Тест 2: локальный zoom
# ===================================================================

def test2_zoom(data, out_dir):
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Локальный zoom одной T2/T3 лунки")
    print("=" * 80)

    phi, Z = data["phi"], data["Z"]
    Phi, Zm = data["Phi"], data["Zm"]
    P_ps_s = data["P_ps_s"]
    P_T2, P_T3 = data["P_ps_T2"], data["P_ps_T3"]
    H_T2, H_T3 = data["H_T2"], data["H_T3"]
    theta_T2 = data["theta_T2"]
    theta_T3 = data["theta_T3"]
    phi_c, Z_c = data["phi_c"], data["Z_c"]
    a_phi, a_Z = data["a_phi"], data["a_Z"]

    # Лунка ближе к макс. P smooth
    iz_max, iphi_max = np.unravel_index(np.argmax(P_ps_s), P_ps_s.shape)
    phi_max = phi[iphi_max]
    Z_max = Z[iz_max]
    print(f"\n  Pmax_smooth при φ={np.rad2deg(phi_max):.1f}°, Z={Z_max:.3f}")

    # Ближайшая лунка
    dist = np.sqrt(((phi_c - phi_max) / a_phi)**2
                    + ((Z_c - Z_max) / a_Z)**2)
    i_best = int(np.argmin(dist))
    phi_cb, Z_cb = phi_c[i_best], Z_c[i_best]
    print(f"  Ближайшая лунка: φ_c={np.rad2deg(phi_cb):.1f}°, Z_c={Z_cb:.3f}")

    # Патч ±2·a
    phi_lo = phi_cb - 2 * a_phi
    phi_hi = phi_cb + 2 * a_phi
    Z_lo = Z_cb - 2 * a_Z
    Z_hi = Z_cb + 2 * a_Z

    mask_phi = (phi >= phi_lo) & (phi <= phi_hi)
    mask_Z = (Z >= Z_lo) & (Z <= Z_hi)
    phi_patch = phi[mask_phi]
    Z_patch = Z[mask_Z]

    def extract(arr):
        return arr[np.ix_(mask_Z, mask_phi)]

    H_T2_p = extract(H_T2)
    H_T3_p = extract(H_T3)
    P_T2_p = extract(P_T2)
    P_T3_p = extract(P_T3)
    th_T2_p = extract(theta_T2) if theta_T2 is not None else None
    th_T3_p = extract(theta_T3) if theta_T3 is not None else None

    # Метрики
    print(f"\n  T2 patch:")
    print(f"    min(theta) = {np.min(th_T2_p):.4f}")
    print(f"    frac(theta<0.99) = "
          f"{np.mean(th_T2_p < 0.99):.3f}")
    print(f"    max(P) = {np.max(P_T2_p):.4f}")
    print(f"\n  T3 patch:")
    print(f"    min(theta) = {np.min(th_T3_p):.4f}")
    print(f"    frac(theta<0.99) = "
          f"{np.mean(th_T3_p < 0.99):.3f}")
    print(f"    max(P) = {np.max(P_T3_p):.4f}")

    # 6 графиков (2 столбца, 3 строки)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    phi_deg = np.rad2deg(phi_patch)

    for col, (H_p, P_p, th_p, label) in enumerate([
        (H_T2_p, P_T2_p, th_T2_p, "T2"),
        (H_T3_p, P_T3_p, th_T3_p, "T3"),
    ]):
        ax = axes[0, col]
        c = ax.contourf(phi_deg, Z_patch, H_p, levels=30, cmap="viridis")
        fig.colorbar(c, ax=ax, label="H")
        ax.set_title(f"H ({label})")
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("Z")

        ax = axes[1, col]
        c = ax.contourf(phi_deg, Z_patch, P_p, levels=30, cmap="hot_r")
        fig.colorbar(c, ax=ax, label="P")
        ax.set_title(f"P ({label})")
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("Z")

        ax = axes[2, col]
        if th_p is not None:
            c = ax.contourf(phi_deg, Z_patch, th_p, levels=30, cmap="RdYlBu")
            fig.colorbar(c, ax=ax, label="θ_fill")
        ax.set_title(f"θ_fill ({label})")
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("Z")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "test2_zoom.png"), dpi=300)
    plt.close(fig)
    print(f"\n  График: test2_zoom.png")


# ===================================================================
#  Тест 3: одна лунка в зоне максимума
# ===================================================================

def test3_single_dimple(out_dir):
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Одна T2 / одна T3 лунка в зоне максимума")
    print("=" * 80)

    phi, Z, Phi, Zm, dp, dz = make_grid(N_PHI, N_Z)
    a_phi = RX / R_M
    a_Z = RZ / (L_M / 2)
    depth = RY / C_M

    H_s = make_H_smooth(EPS, Phi)

    P_s, _ = solve_ps(H_s, dp, dz)
    W_s = compute_W(P_s, Phi, phi, Z)
    Pmax_s = np.max(P_s)
    print(f"\n  Smooth:  W={W_s:.4f}, Pmax={Pmax_s:.4f}")

    # Одна лунка при θ=130°, Z=0.5
    phi_c = np.array([np.deg2rad(130.0)])
    Z_c = np.array([0.5])

    H_T2 = add_wedge_dimples(H_s, Phi, Zm, phi_c, Z_c, a_phi, a_Z, depth,
                              direction="convergent")
    P_T2, _ = solve_ps(H_T2, dp, dz)
    W_T2 = compute_W(P_T2, Phi, phi, Z)
    print(f"  1×T2:    W={W_T2:.4f} (gain={W_T2/W_s:.4f}), "
          f"Pmax={np.max(P_T2):.4f} ({np.max(P_T2)/Pmax_s:.4f}×)")

    H_T3 = add_wedge_dimples(H_s, Phi, Zm, phi_c, Z_c, a_phi, a_Z, depth,
                              direction="divergent")
    P_T3, _ = solve_ps(H_T3, dp, dz)
    W_T3 = compute_W(P_T3, Phi, phi, Z)
    print(f"  1×T3:    W={W_T3:.4f} (gain={W_T3/W_s:.4f}), "
          f"Pmax={np.max(P_T3):.4f} ({np.max(P_T3)/Pmax_s:.4f}×)")

    ratio = W_T2 / W_T3 if W_T3 > 0 else 0
    print(f"\n  T2/T3 ratio (1 лунка): {ratio:.4f}")
    if ratio > 1.05:
        print(f"  ✓ Клин локально работает — проблема в массовом взаимодействии")
    elif ratio < 0.95:
        print(f"  ✗ Одна T3 лучше T2 — клин фундаментально не работает")
    else:
        print(f"  ~ T2 ≈ T3 — клин не даёт локального преимущества")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "wedge_diag")
    os.makedirs(out_dir, exist_ok=True)

    print(f"ДИАГНОСТИКА WEDGE T2/T3")
    print(f"Параметры: D=L=40мм, C=50мкм, ε={EPS}, Ptex=40%, "
          f"phi_bc=groove")
    print(f"Результаты → {out_dir}")

    data = test1_hs_vs_ps(out_dir)
    test2_zoom(data, out_dir)
    test3_single_dimple(out_dir)


if __name__ == "__main__":
    main()
