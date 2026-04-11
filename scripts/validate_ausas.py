#!/usr/bin/env python3
"""Валидация PS-солвера против Ausas et al. (2006).

Безразмерная задача. Домен Ω = (0, 2π) × (-1, 1), B_ausas = 0.1.
h = 1 + ε·cos(φ) + h_t(φ, z)
Текстура: квадратные ступенчатые лунки (step profile).
Без пьезовязкости, без шероховатости, laminar.

Stage A: pa=0 (оба торца), качественная проверка формы P(x).
Stage B: pa=0.0075 на inlet, количественная (Table 1).
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Прямой импорт солверов ───────────────────────────────────────
try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu

from reynolds_solver import solve_reynolds  # для HS

# ─── Параметры Ausas ──────────────────────────────────────────────
B_AUSAS = 0.1          # L/(2R) в безразмерном виде
EPS = 0.6              # эксцентриситет (оценка по Fig. 4)
PA = 0.0               # Stage A: pa=0
PA_STAGE_B = 0.0075    # Stage B: feeding pressure

# Текстура: 50×5 квадратных лунок по всему домену
N1_TEX = 50            # по φ
N2_TEX = 5             # по Z
S_FRAC = 0.20          # area fraction

HT0_VALUES = [0.15, 0.30, 0.45, 0.60, 0.75]


def make_grid(N_phi, N_Z):
    """Сетка φ ∈ [0, 2π), Z ∈ [-1, 1]."""
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_H_smooth(eps, Phi):
    """Чистый H без регуляризации шероховатости."""
    return 1.0 + eps * np.cos(Phi)


def add_square_dimples(H, Phi, Z, N1, N2, s_frac, ht0):
    """Добавить квадратные ступенчатые лунки.

    Домен разбит на N1×N2 ячеек. В центре каждой ячейки —
    квадратная лунка глубиной ht0.
    """
    H_out = H.copy()
    N_Z, N_phi = Phi.shape

    # Размер ячейки
    cell_phi = 2 * np.pi / N1      # в радианах
    cell_Z = 2.0 / N2              # в безразмерных Z

    # Сторона квадрата: area = s_frac × cell_area
    # В координатах: квадрат со стороной a_phi (рад) и a_Z (безразм.)
    # Для квадратных лунок: a_phi/cell_phi = a_Z/cell_Z = sqrt(s_frac)
    side_ratio = np.sqrt(s_frac)
    a_phi = side_ratio * cell_phi
    a_Z = side_ratio * cell_Z

    for i1 in range(N1):
        phi_c = (i1 + 0.5) * cell_phi
        for i2 in range(N2):
            Z_c = -1.0 + (i2 + 0.5) * cell_Z

            # Маска: внутри квадрата
            dphi = np.abs(Phi - phi_c)
            # Периодическая по φ
            dphi = np.minimum(dphi, 2 * np.pi - dphi)
            dz = np.abs(Z - Z_c)

            mask = (dphi < a_phi / 2) & (dz < a_Z / 2)
            H_out[mask] += ht0

    return H_out


def compute_forces(P, H, Phi, phi_1D, Z_1D, theta=None):
    """Несущая способность и трение (формула Ausas eq. 10).

    Friction T = ∫_Ω⁺ (1/h + 3h·∂p/∂φ) dΩ
    Интегрирование по Ω⁺ (full-film only, θ=1).
    """
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]

    # Load
    Fx = -np.trapz(np.trapz(P * np.cos(Phi), phi_1D, axis=1),
                    Z_1D, axis=0)
    Fy = -np.trapz(np.trapz(P * np.sin(Phi), phi_1D, axis=1),
                    Z_1D, axis=0)
    W = np.sqrt(Fx**2 + Fy**2)

    # Friction (Ausas eq. 10)
    dP_dphi = np.gradient(P, d_phi, axis=1)

    if theta is not None:
        # Ω⁺: full-film region (θ ≈ 1)
        full_film = theta > (1.0 - 1e-6)
    else:
        # HS: P > 0 region
        full_film = P > 0

    integrand = np.where(full_film, 1.0 / H + 3.0 * H * dP_dphi, 0.0)
    T = np.trapz(np.trapz(integrand, phi_1D, axis=1), Z_1D, axis=0)

    return W, T


def run_stage_a(out_dir):
    """Stage A: pa=0, качественная проверка."""
    print(f"\n{'=' * 80}")
    print(f"STAGE A: pa=0, ε={EPS}")
    print(f"{'=' * 80}")

    N_phi_s, N_Z_s = 500, 50     # smooth
    N_phi_t, N_Z_t = 750, 75     # textured

    # --- Smooth ---
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi_s, N_Z_s)
    H_s = make_H_smooth(EPS, Phi)

    # PS
    t0 = time.time()
    P_ps, theta_ps, res_ps, nit_ps = _ps_solver(
        H_s, dp, dz, 1.0, 1.0, tol=1e-6, max_iter=10_000_000)
    dt_ps = time.time() - t0
    W_ps, T_ps = compute_forces(P_ps, H_s, Phi, phi, Z, theta_ps)
    P_max_ps = np.max(P_ps)
    print(f"\n  Smooth PS: P_max={P_max_ps:.4f} (expect ~0.128), "
          f"W={W_ps:.4f}, T={T_ps:.4f}, {dt_ps:.1f}с")

    # HS
    t0 = time.time()
    P_hs, _, _, _ = solve_reynolds(
        H_s, dp, dz, 1.0, 1.0,
        closure="laminar", cavitation="half_sommerfeld",
        return_converged=True)
    dt_hs = time.time() - t0
    W_hs, T_hs = compute_forces(P_hs, H_s, Phi, phi, Z)
    print(f"  Smooth HS: P_max={np.max(P_hs):.4f}, "
          f"W={W_hs:.4f}, T={T_hs:.4f}, {dt_hs:.1f}с")

    # Midplane P(x) — smooth
    iz_mid = N_Z_s // 2
    x_norm = phi / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_norm, P_ps[iz_mid, :], "b-", linewidth=2, label="PS (Payvar-Salant)")
    ax.plot(x_norm, P_hs[iz_mid, :], "r--", linewidth=1.5, label="HS (Half-Sommerfeld)")
    ax.set_xlabel("x = φ/(2π)")
    ax.set_ylabel("P (безразмерное)")
    ax.set_title(f"Midplane P(x), smooth, ε={EPS} (cf. Ausas Fig. 4a)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_fig4a_smooth.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "ausas_fig4a_smooth.pdf"))
    plt.close(fig)
    print(f"  График: ausas_fig4a_smooth.png")

    # --- Textured 50×5, ht0=0.30 ---
    phi_t, Z_t, Phi_t, Zm_t, dp_t, dz_t = make_grid(N_phi_t, N_Z_t)
    H_t0 = make_H_smooth(EPS, Phi_t)
    ht0 = 0.30
    H_tex = add_square_dimples(H_t0, Phi_t, Zm_t, N1_TEX, N2_TEX, S_FRAC, ht0)

    # PS
    t0 = time.time()
    P_ps_t, theta_ps_t, _, _ = _ps_solver(
        H_tex, dp_t, dz_t, 1.0, 1.0, tol=1e-6, max_iter=10_000_000)
    dt = time.time() - t0
    W_ps_t, T_ps_t = compute_forces(P_ps_t, H_tex, Phi_t, phi_t, Z_t, theta_ps_t)
    print(f"\n  Textured PS (ht0={ht0}): P_max={np.max(P_ps_t):.4f}, "
          f"W={W_ps_t:.4f}, T={T_ps_t:.4f}, {dt:.1f}с")

    # HS
    t0 = time.time()
    P_hs_t, _, _, _ = solve_reynolds(
        H_tex, dp_t, dz_t, 1.0, 1.0,
        closure="laminar", cavitation="half_sommerfeld",
        return_converged=True)
    dt = time.time() - t0
    W_hs_t, T_hs_t = compute_forces(P_hs_t, H_tex, Phi_t, phi_t, Z_t)
    print(f"  Textured HS (ht0={ht0}): P_max={np.max(P_hs_t):.4f}, "
          f"W={W_hs_t:.4f}, T={T_hs_t:.4f}, {dt:.1f}с")

    # Midplane P(x) — textured
    iz_mid_t = N_Z_t // 2
    x_norm_t = phi_t / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_norm_t, P_ps_t[iz_mid_t, :], "b-", linewidth=1.5,
            label="PS (Payvar-Salant)")
    ax.plot(x_norm_t, P_hs_t[iz_mid_t, :], "r--", linewidth=1,
            label="HS (Half-Sommerfeld)")
    ax.set_xlabel("x = φ/(2π)")
    ax.set_ylabel("P (безразмерное)")
    ax.set_title(f"Midplane P(x), textured 50×5 ht0={ht0}, ε={EPS} "
                 f"(cf. Ausas Fig. 4b)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_fig4b_textured.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "ausas_fig4b_textured.pdf"))
    plt.close(fig)
    print(f"  График: ausas_fig4b_textured.png")

    # θ(x) — кавитационная зона
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_norm_t, theta_ps_t[iz_mid_t, :], "b-", linewidth=1.5,
            label="PS θ")
    ax.set_xlabel("x = φ/(2π)")
    ax.set_ylabel("θ")
    ax.set_title(f"θ(x) midplane, textured 50×5 ht0={ht0}, ε={EPS}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_theta_textured.png"), dpi=300)
    plt.close(fig)
    print(f"  График: ausas_theta_textured.png")

    # Контурная карта P(φ,Z) textured PS
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, P, title in [(axes[0], P_ps_t, "PS"), (axes[1], P_hs_t, "HS")]:
        c = ax.contourf(x_norm_t, Z_t, P, levels=50, cmap="hot_r")
        fig.colorbar(c, ax=ax, label="P")
        ax.set_xlabel("x = φ/(2π)")
        ax.set_title(f"{title}, textured 50×5 ht0={ht0}")
    axes[0].set_ylabel("Z")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_P_contour.png"), dpi=300)
    plt.close(fig)
    print(f"  График: ausas_P_contour.png")

    return {"P_max_smooth_ps": P_max_ps, "T_smooth_ps": T_ps,
            "T_tex_ps": T_ps_t}


def run_stage_b(out_dir):
    """Stage B: friction vs ht0 (Table 1, Fig. 8).

    pa=0 (наш солвер пока не поддерживает ненулевой Dirichlet).
    Трактовать как semi-quantitative.
    """
    print(f"\n{'=' * 80}")
    print(f"STAGE B: Friction vs ht0 (Table 1 / Fig. 8)")
    print(f"pa = 0 (наш солвер), Ausas pa = {PA_STAGE_B}")
    print(f"{'=' * 80}")

    N_phi, N_Z = 750, 75
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)
    H_s = make_H_smooth(EPS, Phi)

    # Smooth reference
    P_ps, theta_ps, _, _ = _ps_solver(
        H_s, dp, dz, 1.0, 1.0, tol=1e-6, max_iter=10_000_000)
    _, T_smooth = compute_forces(P_ps, H_s, Phi, phi, Z, theta_ps)
    print(f"\n  Smooth: T={T_smooth:.4f} (Ausas Table 1: 1.201)")

    P_hs, _, _, _ = solve_reynolds(
        H_s, dp, dz, 1.0, 1.0,
        closure="laminar", cavitation="half_sommerfeld",
        return_converged=True)
    _, T_smooth_hs = compute_forces(P_hs, H_s, Phi, phi, Z)
    print(f"  Smooth HS: T={T_smooth_hs:.4f}")

    # Sweep по ht0
    print(f"\n  {'ht0':>6} {'T_PS':>8} {'T_HS':>8} {'T/T_s PS':>10} {'T/T_s HS':>10}")
    print("  " + "-" * 50)

    T_ps_arr = []
    T_hs_arr = []
    for ht0 in HT0_VALUES:
        H_tex = add_square_dimples(H_s.copy(), Phi, Zm, N1_TEX, N2_TEX,
                                    S_FRAC, ht0)
        # PS
        P_t, theta_t, _, _ = _ps_solver(
            H_tex, dp, dz, 1.0, 1.0, tol=1e-6, max_iter=10_000_000)
        _, T_t = compute_forces(P_t, H_tex, Phi, phi, Z, theta_t)
        T_ps_arr.append(T_t)

        # HS
        P_hs_t, _, _, _ = solve_reynolds(
            H_tex, dp, dz, 1.0, 1.0,
            closure="laminar", cavitation="half_sommerfeld",
            return_converged=True)
        _, T_hs_t = compute_forces(P_hs_t, H_tex, Phi, phi, Z)
        T_hs_arr.append(T_hs_t)

        ratio_ps = T_t / T_smooth if T_smooth > 0 else 0
        ratio_hs = T_hs_t / T_smooth_hs if T_smooth_hs > 0 else 0
        print(f"  {ht0:6.2f} {T_t:8.4f} {T_hs_t:8.4f} "
              f"{ratio_ps:10.4f} {ratio_hs:10.4f}")

    # График T(ht0) — Fig. 8 analogue
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0] + HT0_VALUES, [T_smooth] + T_ps_arr,
            "bo-", linewidth=2, markersize=6, label="PS (Payvar-Salant)")
    ax.plot([0] + HT0_VALUES, [T_smooth_hs] + T_hs_arr,
            "rs--", linewidth=2, markersize=6, label="HS (Half-Sommerfeld)")
    ax.set_xlabel("ht0 (глубина лунки)")
    ax.set_ylabel("T (friction)")
    ax.set_title(f"Friction vs dimple depth, ε={EPS}, 50×5 square, s={S_FRAC}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_fig8_friction.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "ausas_fig8_friction.pdf"))
    plt.close(fig)
    print(f"\n  График: ausas_fig8_friction.png")

    # Acceptance checklist
    print(f"\n{'=' * 80}")
    print("ACCEPTANCE CHECKLIST")
    print(f"{'=' * 80}")
    print(f"  [{'✓' if abs(T_smooth - 1.201) / 1.201 < 0.20 else '✗'}] "
          f"Smooth T={T_smooth:.4f} (Ausas: 1.201, pa=0 vs pa=0.0075)")
    print(f"  Note: pa=0 может дать другой T — semi-quantitative")

    # Тренд: PS friction с ht0 — слабый рост?
    trend_ps = T_ps_arr[-1] > T_ps_arr[0]
    trend_hs = T_hs_arr[-1] > T_hs_arr[0]
    print(f"  [{'✓' if trend_ps else '✗'}] PS friction растёт с ht0 "
          f"(Fig. 8 trend)")
    print(f"  [{'✓' if trend_hs else '✗'}] HS friction растёт с ht0")
    print(f"{'=' * 80}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "validation_ausas")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ВАЛИДАЦИЯ PAYVAR-SALANT vs AUSAS et al. (2006)")
    print(f"ε={EPS}, B={B_AUSAS}, texture 50×5, s={S_FRAC}")
    print(f"Результаты → {out_dir}")
    print("=" * 80)

    stage_a_results = run_stage_a(out_dir)
    run_stage_b(out_dir)

    print(f"\nВсе результаты → {out_dir}")


if __name__ == "__main__":
    main()
