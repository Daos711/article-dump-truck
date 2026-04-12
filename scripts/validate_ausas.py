#!/usr/bin/env python3
"""Валидация PS-солвера против Ausas et al. (2006).

x₁ = φ ∈ [0, 2π], x₂ ∈ [0, B] → Z ∈ [-1, 1]: dx₂ = B·dZ/2.
h = 1 + X·cos(φ) + Y·sin(φ) + h_t

Stage A: фиксированный ε, качественная проверка P(x), θ(x).
Stage B: поиск равновесия для 5 значений Wa — количественная.
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

B_AUSAS = 0.1
R_AUSAS = 1.0
L_AUSAS = R_AUSAS * B_AUSAS  # = 0.1

N1_TEX = 50
N2_TEX = 5
S_FRAC = 0.20
HT0_VALUES = [0.15, 0.30, 0.45, 0.60, 0.75]
WA_VALUES = [0.002, 0.0048, 0.0076, 0.0104, 0.0132]


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_H_XY(X, Y, Phi):
    return 1.0 + X * np.cos(Phi) + Y * np.sin(Phi)


def add_square_dimples(H, Phi, Zm, N1, N2, s_frac, ht0):
    H_out = H.copy()
    cell_phi = 2 * np.pi / N1
    cell_Z = 2.0 / N2
    side_ratio = np.sqrt(s_frac)
    a_phi = side_ratio * cell_phi
    a_Z = side_ratio * cell_Z
    for i1 in range(N1):
        phi_c = (i1 + 0.5) * cell_phi
        for i2 in range(N2):
            Z_c = -1.0 + (i2 + 0.5) * cell_Z
            dphi = np.abs(Phi - phi_c)
            dphi = np.minimum(dphi, 2 * np.pi - dphi)
            dz = np.abs(Zm - Z_c)
            mask = (dphi < a_phi / 2) & (dz < a_Z / 2)
            H_out[mask] += ht0
    return H_out


def compute_forces_ausas(P, H, Phi, phi_1D, Z_1D, theta=None):
    """Нагрузка и трение (Couette + Poiseuille раздельно).

    Returns: WX, WY, W, T, T_couette, T_poiseuille
    """
    d_phi = phi_1D[1] - phi_1D[0]
    scale = B_AUSAS / 2.0

    WX = np.trapezoid(np.trapezoid(P * np.cos(Phi), phi_1D, axis=1),
                       Z_1D, axis=0) * scale
    WY = np.trapezoid(np.trapezoid(P * np.sin(Phi), phi_1D, axis=1),
                       Z_1D, axis=0) * scale
    W = np.sqrt(WX**2 + WY**2)

    dP_dx1 = np.gradient(P, d_phi, axis=1)

    if theta is not None:
        full_film = theta > (1.0 - 1e-6)
    else:
        full_film = P > 0

    couette_int = np.where(full_film, 1.0 / H, 0.0)
    poiseuille_int = np.where(full_film, 3.0 * H * dP_dx1, 0.0)

    T_c = np.trapezoid(np.trapezoid(couette_int, phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    T_p = np.trapezoid(np.trapezoid(poiseuille_int, phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    T = T_c + T_p

    return WX, WY, W, T, T_c, T_p


def solve_ps(H, dp, dz):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, R_AUSAS, L_AUSAS, tol=1e-6, max_iter=10_000_000,
        hs_warmup_iter=500_000, hs_warmup_omega=1.5)
    return P, theta


def solve_hs(H, dp, dz):
    P, _, _, _ = solve_reynolds(
        H, dp, dz, R_AUSAS, L_AUSAS,
        closure="laminar", cavitation="half_sommerfeld",
        return_converged=True)
    return P


def find_equilibrium(Wa, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                     textured=False, ht0=0.0, solver="ps",
                     max_iter=30, tol=1e-4):
    """Newton-Raphson: find (X, Y) such that W_hydro = Wa (vertical)."""
    X, Y = 0.0, -0.3
    dXY = 1e-5

    T_eq = 0.0
    T_c_eq = 0.0
    T_p_eq = 0.0
    theta_eq = None

    for it in range(max_iter):
        H = make_H_XY(X, Y, Phi)
        if textured:
            H = add_square_dimples(H, Phi, Zm, N1_TEX, N2_TEX, S_FRAC, ht0)

        if solver == "ps":
            P, theta = solve_ps(H, d_phi, d_Z)
            WX, WY, W, T, T_c, T_p = compute_forces_ausas(
                P, H, Phi, phi_1D, Z_1D, theta)
            theta_eq = theta
        else:
            P = solve_hs(H, d_phi, d_Z)
            WX, WY, W, T, T_c, T_p = compute_forces_ausas(
                P, H, Phi, phi_1D, Z_1D)

        T_eq, T_c_eq, T_p_eq = T, T_c, T_p

        Rx = WX
        Ry = WY - Wa
        err = np.sqrt(Rx**2 + Ry**2)

        if err < tol * Wa:
            eps = np.sqrt(X**2 + Y**2)
            cav = float(np.mean(theta_eq < 1.0 - 1e-6)) if theta_eq is not None else 0
            return X, Y, eps, W, T_eq, T_c_eq, T_p_eq, cav, it + 1

        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0), (0, dXY)]):
            H_p = make_H_XY(X + dX_, Y + dY_, Phi)
            if textured:
                H_p = add_square_dimples(H_p, Phi, Zm, N1_TEX, N2_TEX,
                                          S_FRAC, ht0)
            if solver == "ps":
                Pp, thp = solve_ps(H_p, d_phi, d_Z)
                WXp, WYp, _, _, _, _ = compute_forces_ausas(
                    Pp, H_p, Phi, phi_1D, Z_1D, thp)
            else:
                Pp = solve_hs(H_p, d_phi, d_Z)
                WXp, WYp, _, _, _, _ = compute_forces_ausas(
                    Pp, H_p, Phi, phi_1D, Z_1D)
            J[0, col] = (WXp - WX) / dXY
            J[1, col] = (WYp - WY) / dXY

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-20:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det

        step = min(1.0, 0.1 / max(abs(dX), abs(dY), 1e-10))
        X += step * dX
        Y += step * dY

    eps = np.sqrt(X**2 + Y**2)
    cav = float(np.mean(theta_eq < 1.0 - 1e-6)) if theta_eq is not None else 0
    return X, Y, eps, W, T_eq, T_c_eq, T_p_eq, cav, max_iter


def compute_stage_a(out_dir):
    EPS = 0.6
    print(f"\n{'=' * 80}")
    print(f"STAGE A: ε={EPS}, pa=0")
    print(f"{'=' * 80}")

    N_phi_s, N_Z_s = 500, 50
    N_phi_t, N_Z_t = 750, 75

    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi_s, N_Z_s)
    H_s = make_H_XY(EPS, 0.0, Phi)

    P_ps, theta_ps = solve_ps(H_s, dp, dz)
    _, _, W_ps, T_ps, Tc_ps, Tp_ps = compute_forces_ausas(
        P_ps, H_s, Phi, phi, Z, theta_ps)
    print(f"\n  Smooth PS: P_max={np.max(P_ps):.4f}, W={W_ps:.4f}, "
          f"T={T_ps:.4f} (Tc={Tc_ps:.4f} + Tp={Tp_ps:.4f})")

    T_analytical = 2 * np.pi * B_AUSAS / np.sqrt(1 - EPS**2)
    print(f"  T_couette analytical (full domain) = {T_analytical:.4f}")

    P_hs = solve_hs(H_s, dp, dz)
    _, _, W_hs, T_hs, Tc_hs, Tp_hs = compute_forces_ausas(
        P_hs, H_s, Phi, phi, Z)
    print(f"  Smooth HS: P_max={np.max(P_hs):.4f}, W={W_hs:.4f}, "
          f"T={T_hs:.4f} (Tc={Tc_hs:.4f} + Tp={Tp_hs:.4f})")

    phi_t, Z_t, Phi_t, Zm_t, dp_t, dz_t = make_grid(N_phi_t, N_Z_t)
    H0 = make_H_XY(EPS, 0.0, Phi_t)
    ht0 = 0.30
    H_tex = add_square_dimples(H0, Phi_t, Zm_t, N1_TEX, N2_TEX, S_FRAC, ht0)

    P_ps_t, theta_t = solve_ps(H_tex, dp_t, dz_t)
    _, _, W_t, T_t, Tc_t, Tp_t = compute_forces_ausas(
        P_ps_t, H_tex, Phi_t, phi_t, Z_t, theta_t)
    print(f"\n  Tex PS (ht0={ht0}): P_max={np.max(P_ps_t):.4f}, W={W_t:.4f}, "
          f"T={T_t:.4f} (Tc={Tc_t:.4f} + Tp={Tp_t:.4f})")

    P_hs_t = solve_hs(H_tex, dp_t, dz_t)
    _, _, W_ht, T_ht, Tc_ht, Tp_ht = compute_forces_ausas(
        P_hs_t, H_tex, Phi_t, phi_t, Z_t)
    print(f"  Tex HS (ht0={ht0}): P_max={np.max(P_hs_t):.4f}, W={W_ht:.4f}, "
          f"T={T_ht:.4f} (Tc={Tc_ht:.4f} + Tp={Tp_ht:.4f})")

    np.savez(os.path.join(out_dir, "stage_a.npz"),
             EPS=EPS, ht0=ht0,
             phi=phi, Z=Z, P_ps=P_ps, P_hs=P_hs,
             phi_t=phi_t, Z_t=Z_t, P_ps_t=P_ps_t, P_hs_t=P_hs_t,
             N_Z_s=N_Z_s, N_Z_t=N_Z_t)
    print(f"  stage_a.npz сохранён")


def plot_stage_a(out_dir):
    npz_path = os.path.join(out_dir, "stage_a.npz")
    if not os.path.exists(npz_path):
        print(f"  stage_a.npz не найден — пропуск графиков Stage A")
        return
    d = np.load(npz_path)
    EPS = float(d["EPS"])
    ht0 = float(d["ht0"])
    phi, Z = d["phi"], d["Z"]
    P_ps, P_hs = d["P_ps"], d["P_hs"]
    phi_t, Z_t = d["phi_t"], d["Z_t"]
    P_ps_t, P_hs_t = d["P_ps_t"], d["P_hs_t"]
    N_Z_s = int(d["N_Z_s"])
    N_Z_t = int(d["N_Z_t"])

    iz = N_Z_s // 2
    x = phi / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, P_ps[iz, :], "b-", lw=2, label="PS")
    ax.plot(x, P_hs[iz, :], "r--", lw=1.5, label="HS")
    ax.set_xlabel("x₁ = φ/(2π)")
    ax.set_ylabel("p")
    ax.set_title(f"Smooth midplane, ε={EPS} (cf. Ausas Fig. 4a)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_fig4a_smooth.png"), dpi=300)
    plt.close(fig)

    iz_t = N_Z_t // 2
    x_t = phi_t / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_t, P_ps_t[iz_t, :], "b-", lw=1.5, label="PS")
    ax.plot(x_t, P_hs_t[iz_t, :], "r--", lw=1, label="HS")
    ax.set_xlabel("x₁ = φ/(2π)")
    ax.set_ylabel("p")
    ax.set_title(f"Textured 50×5 ht0={ht0}, ε={EPS} (cf. Fig. 4b)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_fig4b_textured.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, P, title in [(axes[0], P_ps_t, "PS"), (axes[1], P_hs_t, "HS")]:
        c = ax.contourf(x_t, Z_t, P, levels=50, cmap="hot_r")
        fig.colorbar(c, ax=ax, label="p")
        ax.set_xlabel("x₁")
        ax.set_title(f"{title}, ht0={ht0}")
    axes[0].set_ylabel("Z")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_P_contour.png"), dpi=300)
    plt.close(fig)
    print(f"  Stage A графики: ausas_fig4a_smooth, ausas_fig4b_textured, ausas_P_contour")


def compute_stage_b(out_dir):
    print(f"\n{'=' * 80}")
    print(f"STAGE B: Все Wa, smooth + textured, с равновесием")
    print(f"{'=' * 80}")

    N_phi, N_Z = 750, 75
    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)

    # --- Smooth для всех Wa ---
    print(f"\n  SMOOTH (PS):")
    print(f"  {'Wa':>8} {'ε_eq':>6} {'T':>8} {'T_c':>8} {'T_p':>8} "
          f"{'T_c_an':>8} {'cav%':>6} {'nit':>4}")
    print("  " + "-" * 65)

    smooth_data = {}
    for Wa in WA_VALUES:
        t0 = time.time()
        X, Y, eps, W, T, Tc, Tp, cav, nit = find_equilibrium(
            Wa, Phi, Zm, phi, Z, dp, dz, solver="ps")
        dt = time.time() - t0
        Tc_an = 2 * np.pi * B_AUSAS / np.sqrt(1 - eps**2)
        smooth_data[Wa] = {"eps": eps, "T": T, "Tc": Tc, "Tp": Tp}
        print(f"  {Wa:8.4f} {eps:6.4f} {T:8.4f} {Tc:8.4f} {Tp:8.4f} "
              f"{Tc_an:8.4f} {cav*100:6.1f} {nit:4d}  ({dt:.0f}с)")

    # Ausas reference
    print(f"\n  Ausas Fig. 6 ref: T(0.002)≈0.88, T(0.0076)≈1.10, T(0.0132)≈1.30")

    # Textured Stage B включается флагом AUSAS_TEX=1
    if os.environ.get("AUSAS_TEX", "0") != "1":
        print(f"\n  [textured Stage B skipped — set AUSAS_TEX=1 to enable]")
        return

    # --- Textured ht0=0.30 для всех Wa ---
    ht0 = 0.30
    print(f"\n  TEXTURED ht0={ht0} (PS):")
    print(f"  {'Wa':>8} {'ε_eq':>6} {'T_tex':>8} {'T_sm':>8} "
          f"{'T_t/T_s':>8} {'cav%':>6} {'nit':>4}")
    print("  " + "-" * 60)

    tex_data = {}
    for Wa in WA_VALUES:
        t0 = time.time()
        X, Y, eps, W, T, Tc, Tp, cav, nit = find_equilibrium(
            Wa, Phi, Zm, phi, Z, dp, dz,
            textured=True, ht0=ht0, solver="ps")
        dt = time.time() - t0
        T_s = smooth_data[Wa]["T"]
        ratio = T / T_s if T_s > 0 else 0
        tex_data[Wa] = {"eps": eps, "T": T, "ratio": ratio}
        print(f"  {Wa:8.4f} {eps:6.4f} {T:8.4f} {T_s:8.4f} "
              f"{ratio:8.4f} {cav*100:6.1f} {nit:4d}  ({dt:.0f}с)")

    print(f"\n  Ausas Table 1: T_smooth=1.201, T_square=1.223, ratio=1.018")

    # --- T(ht0) для Wa=0.0076 ---
    Wa_ref = 0.0076
    print(f"\n  T(ht0) при Wa={Wa_ref}:")
    print(f"  {'ht0':>6} {'ε_eq':>6} {'T':>8} {'T/T_s':>7}")
    print("  " + "-" * 35)

    T_s_ref = smooth_data[Wa_ref]["T"]
    T_arr = []
    for ht0_v in HT0_VALUES:
        X, Y, eps, W, T, Tc, Tp, cav, nit = find_equilibrium(
            Wa_ref, Phi, Zm, phi, Z, dp, dz,
            textured=True, ht0=ht0_v, solver="ps")
        T_arr.append(T)
        ratio = T / T_s_ref if T_s_ref > 0 else 0
        print(f"  {ht0_v:6.2f} {eps:6.4f} {T:8.4f} {ratio:7.4f}")

    # Сохранить данные для --plot-only
    T_smooth_arr = np.array([smooth_data[w]["T"] for w in WA_VALUES])
    np.savez(os.path.join(out_dir, "stage_b.npz"),
             Wa_values=np.array(WA_VALUES), T_smooth=T_smooth_arr,
             ht0_values=np.array(HT0_VALUES),
             T_s_ref=T_s_ref, T_arr=np.array(T_arr), Wa_ref=Wa_ref)


def plot_stage_b(out_dir):
    npz_path = os.path.join(out_dir, "stage_b.npz")
    if not os.path.exists(npz_path):
        print(f"  stage_b.npz не найден — пропуск графиков Stage B")
        return
    d = np.load(npz_path)
    Wa_values = d["Wa_values"]
    T_smooth_arr = d["T_smooth"]
    ht0_values = d["ht0_values"]
    T_s_ref = float(d["T_s_ref"])
    T_arr = d["T_arr"]
    Wa_ref = float(d["Wa_ref"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0] + list(ht0_values), [T_s_ref] + list(T_arr),
            "bo-", lw=2, markersize=6, label=f"PS (Wa={Wa_ref})")
    ax.set_xlabel("ht0")
    ax.set_ylabel("T (friction)")
    ax.set_title(f"Friction vs ht0, 50×5 square, s={S_FRAC}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_fig8_equilibrium.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Wa_values, T_smooth_arr, "bo-", lw=2, markersize=6, label="PS smooth")
    ax.set_xlabel("Wa")
    ax.set_ylabel("T (friction)")
    ax.set_title("Friction vs Wa (smooth)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ausas_T_vs_Wa.png"), dpi=300)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Загрузить npz и построить графики")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Папка с npz")
    args = parser.parse_args()

    default_dir = os.path.join(os.path.dirname(__file__), "..",
                                "results", "validation_ausas")
    out_dir = args.data_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ВАЛИДАЦИЯ PS vs AUSAS et al. (2006)")
    print(f"B={B_AUSAS}, texture 50×5, s={S_FRAC}")
    print(f"Результаты → {out_dir}")
    print("=" * 80)

    if args.plot_only:
        print("--plot-only: загрузка npz")
    else:
        compute_stage_a(out_dir)
        compute_stage_b(out_dir)

    print("\nПостроение графиков:")
    plot_stage_a(out_dir)
    plot_stage_b(out_dir)
    print(f"\nВсе результаты → {out_dir}")


if __name__ == "__main__":
    main()
