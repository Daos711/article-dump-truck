#!/usr/bin/env python3
"""Wedge-bottom текстура (T2/T3 Manser 2019).

ЭТАП A: воспроизведение Manser параметров (D=L=40мм, ε=0.3-0.8, groove BC)
        для валидации T2-профиля. Smooth + T2 + T3, Full и Partial зоны.

ЭТАП B: перенос на насосный подшипник (R=35, L=56мм, periodic BC).
        Выполняется только если этап A подтвердил gain_W > 1 для T2.

Профили:
  T2 (convergent): Δh = depth*(1 − x_local), max на входе (+θ дырка)
  T3 (divergent):  Δh = depth*(1 + x_local), max на выходе
  SQ (flat):       Δh = depth

Кавитация: Payvar-Salant.
"""
import sys
import os
import time
import csv
import argparse

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

R_SOLVER = 1.0
L_SOLVER = 2.0  # α² = (2R/L)² = 1.0 для L/D=1 (Manser)

# ─── Stage A: Manser ──────────────────────────────────────────────
A_R = 0.020
A_L = 0.040
A_C = 50e-6
A_OMEGA = 2 * np.pi * 3000 / 60
A_MU = 0.05
A_P_SCALE = 6 * A_MU * A_OMEGA * A_R**2 / A_C**2

A_RX = 3.0e-3    # радиус текстуры по θ (полуось)
A_RZ = 3.0e-3
A_RY = 15e-6     # глубина

A_N_PHI = 441
A_N_Z = 121
A_EPS_VALUES = [0.3, 0.5, 0.6, 0.8]

# Ptex → (nCθ, nCz)
A_PTEX_MAP = {
    20: (10, 3),
    40: (14, 4),
    60: (17, 5),
    80: (18, 6),
}

A_ZONES = [("full", 0, 360), ("partial", 180, 360)]

# ─── Stage B: насос ───────────────────────────────────────────────
B_R = 0.035
B_L = 0.056
B_C = 50e-6
B_N = 3000
B_OMEGA = 2 * np.pi * B_N / 60

B_OILS = [
    ("mineral", "Минеральное", 0.022),
    ("rapeseed", "Рапсовое", 0.025),
]
B_R_SOLVER = 1.0
# α² = (2R/L)² при R=35, L=56: (2·35/56)² = 1.5625 → L_s при R_s=1:
#   (2/L_s)² = 1.5625 → L_s = 1.6
B_L_SOLVER = 1.6

B_HP_VALUES = [5e-6, 10e-6, 15e-6, 20e-6, 25e-6]
B_SIZES = [(1.5e-3, 1.2e-3), (3.0e-3, 2.4e-3)]
B_ZONES = [("full", 0, 360), ("part_180_360", 180, 360),
           ("part_0_90", 0, 90), ("part_0_180", 0, 180)]
B_PTEX_VALUES = [20, 40, 60, 80]
B_EPS_VALUES = [0.3, 0.5, 0.6, 0.7]
B_PROFILES = ["T2", "SQ"]

B_N_PHI = 441
B_N_Z = 121


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
    """Центры лунок с отступом от краёв зоны.

    a_phi_margin: отступ от границ зоны (в рад). Для groove — ≥ a_phi.
    """
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
    """T2 (convergent) / T3 (divergent) wedge-bottom лунки."""
    H_out = H.copy()
    for phi_c, Z_c in zip(centers_phi, centers_Z):
        dphi = Phi - phi_c
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dz = np.abs(Zm - Z_c)
        mask = (np.abs(dphi) < a_phi) & (dz < a_Z)

        x_local = dphi[mask] / a_phi  # ∈ [-1, +1]

        if direction == "convergent":
            # T2: max глубина на входе (x=-1), 0 на выходе (x=+1)
            H_out[mask] += depth * (1.0 - x_local)
        elif direction == "divergent":
            # T3: 0 на входе, max на выходе
            H_out[mask] += depth * (1.0 + x_local)
        elif direction == "flat":
            # SQ: flat-bottom
            H_out[mask] += depth
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return H_out


def solve_ps(H, dp, dz, R_s, L_s, phi_bc="periodic"):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, R_s, L_s, tol=1e-6, max_iter=10_000_000,
        hs_warmup_iter=200_000, hs_warmup_omega=1.5,
        phi_bc=phi_bc)
    return P, theta


def compute_metrics(P, H, Phi, phi_1D, Z_1D):
    scale = 0.5
    W_theta = -np.trapezoid(np.trapezoid(P * np.cos(Phi), phi_1D, axis=1),
                             Z_1D, axis=0) * scale
    W_Z = np.trapezoid(np.trapezoid(P * np.sin(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * scale
    W = np.sqrt(W_theta**2 + W_Z**2)

    d_phi = phi_1D[1] - phi_1D[0]
    h_nondim = H  # безразмерный зазор
    tau_c = 1.0 / h_nondim
    dP_dphi = np.gradient(P, d_phi, axis=1)
    tau_p = 0.5 * h_nondim * dP_dphi
    friction_int = np.trapezoid(np.trapezoid(tau_c + tau_p,
                                              phi_1D, axis=1),
                                 Z_1D, axis=0) * scale
    # Коэфф. трения по W (упрощённо)
    f_coef = abs(friction_int) / max(W, 1e-10)
    P_max = float(np.max(P))
    return W, f_coef, P_max


# ===================================================================
#  ЭТАП A: воспроизведение Manser
# ===================================================================

def run_stage_a(out_dir):
    print("=" * 90)
    print("ЭТАП A: воспроизведение Manser T2/T3 (groove BC)")
    print(f"D=L=40мм, C={A_C*1e6:.0f}мкм, r_x=r_z={A_RX*1e3:.0f}мм, "
          f"r_y={A_RY*1e6:.0f}мкм")
    print(f"Сетка: {A_N_PHI}×{A_N_Z}")
    print("=" * 90)

    phi, Z, Phi, Zm, dp, dz = make_grid(A_N_PHI, A_N_Z)

    a_phi = A_RX / A_R       # полуось по φ в рад
    a_Z = A_RZ / (A_L / 2)   # полуось по Z безразм.
    depth = A_RY / A_C

    results = []

    for eps in A_EPS_VALUES:
        H_s = make_H_smooth(eps, Phi)
        P_s, _ = solve_ps(H_s, dp, dz, R_SOLVER, L_SOLVER, phi_bc="groove")
        W_s, f_s, Pmax_s = compute_metrics(P_s, H_s, Phi, phi, Z)
        print(f"\n  ε={eps}: Smooth W={W_s:.4f}, P_max={Pmax_s:.4f}")

        for zone_name, phi_start, phi_end in A_ZONES:
            for Ptex in sorted(A_PTEX_MAP.keys()):
                nCth, nCz = A_PTEX_MAP[Ptex]
                phi_c, Z_c = make_dimple_centers(
                    nCth, nCz, phi_start, phi_end,
                    a_phi_margin=a_phi)
                if len(phi_c) == 0:
                    continue

                for profile in ["T2", "T3"]:
                    direction = ("convergent" if profile == "T2"
                                 else "divergent")
                    H_t = add_wedge_dimples(
                        H_s, Phi, Zm, phi_c, Z_c,
                        a_phi, a_Z, depth, direction=direction)
                    t0 = time.time()
                    P_t, _ = solve_ps(H_t, dp, dz,
                                       R_SOLVER, L_SOLVER, phi_bc="groove")
                    dt = time.time() - t0
                    W_t, f_t, Pmax_t = compute_metrics(P_t, H_t, Phi, phi, Z)
                    gain_W = W_t / W_s if W_s > 0 else 0
                    gain_f = f_t / f_s if f_s > 0 else 0
                    marker = " <<<" if gain_W > 1.05 else (
                        " ✓" if gain_W > 1.0 else "")
                    print(f"    {zone_name:>7s} {profile} Ptex={Ptex:>2d}% "
                          f"({nCth}×{nCz}): W={W_t:.4f} "
                          f"gain_W={gain_W:.4f} gain_f={gain_f:.4f} "
                          f"({dt:.1f}с){marker}")

                    r = dict(stage="A", eps=eps, zone=zone_name,
                              phi_start=phi_start, phi_end=phi_end,
                              Ptex=Ptex, nCth=nCth, nCz=nCz,
                              profile=profile,
                              W_smooth=W_s, W_tex=W_t, gain_W=gain_W,
                              f_smooth=f_s, f_tex=f_t, gain_f=gain_f,
                              Pmax_smooth=Pmax_s, Pmax_tex=Pmax_t)
                    results.append(r)

                    # Профили при ε=0.6, Ptex=40%
                    if eps == 0.6 and Ptex == 40 and zone_name == "full":
                        iz = A_N_Z // 2
                        key = f"stage_a_profile_{profile}"
                        np.savez(os.path.join(out_dir, f"{key}.npz"),
                                 phi=phi, P=P_t, iz=iz)

        # Save smooth profile for comparison
        if eps == 0.6:
            iz = A_N_Z // 2
            np.savez(os.path.join(out_dir, "stage_a_profile_smooth.npz"),
                     phi=phi, P=P_s, iz=iz)

    # CSV + CSV
    csv_path = os.path.join(out_dir, "stage_a_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            row = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                   for k, v in r.items()}
            w.writerow(row)
    print(f"\n  CSV: {csv_path}")

    # Check: T2 full > 1 anywhere?
    t2_full = [r for r in results if r["profile"] == "T2"
               and r["zone"] == "full"]
    max_gain_T2 = max((r["gain_W"] for r in t2_full), default=0)
    print(f"\n  Max gain_W (T2, full): {max_gain_T2:.4f}")
    print(f"  Ожидание Manser: ~1.43 при ε=0.6, Ptex=40%")

    # Plot midplane профилей при ε=0.6, Ptex=40%, full
    try:
        d_s = np.load(os.path.join(out_dir, "stage_a_profile_smooth.npz"))
        d_t2 = np.load(os.path.join(out_dir, "stage_a_profile_T2.npz"))
        d_t3 = np.load(os.path.join(out_dir, "stage_a_profile_T3.npz"))
        iz = int(d_s["iz"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(d_s["phi"], d_s["P"][iz, :], "k-", lw=2, label="Smooth")
        ax.plot(d_t2["phi"], d_t2["P"][iz, :], "b-", lw=1.5, label="T2 (wedge conv.)")
        ax.plot(d_t3["phi"], d_t3["P"][iz, :], "r-", lw=1.5, label="T3 (wedge div.)")
        ax.set_xlabel("θ (рад)")
        ax.set_ylabel("P (безразмерное)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage_a_midplane_T2_T3.png"),
                    dpi=300)
        plt.close(fig)
        print(f"  График: stage_a_midplane_T2_T3.png")
    except Exception as e:
        print(f"  Ошибка построения графика: {e}")

    return max_gain_T2, results


# ===================================================================
#  ЭТАП B: перенос на насос
# ===================================================================

def run_stage_b(out_dir):
    print("\n" + "=" * 90)
    print("ЭТАП B: перенос на насосный подшипник")
    print(f"R={B_R*1e3:.0f}мм, L={B_L*1e3:.0f}мм, C={B_C*1e6:.0f}мкм")
    print(f"Сетка: {B_N_PHI}×{B_N_Z}, phi_bc=periodic")
    print("=" * 90)

    phi, Z, Phi, Zm, dp, dz = make_grid(B_N_PHI, B_N_Z)

    # Оценка общего числа расчётов
    total = (len(B_OILS) * len(B_HP_VALUES) * len(B_SIZES) * len(B_ZONES)
             * len(B_PTEX_VALUES) * len(B_EPS_VALUES) * len(B_PROFILES))
    print(f"  Всего комбинаций: ~{total} (с учётом пропусков)")

    results = []
    smooth_cache = {}  # (oil, eps) -> W_s, f_s, Pmax_s
    done = 0
    t0_all = time.time()

    # Smooth baselines (кэшируем)
    for oil_key, oil_name, mu in B_OILS:
        for eps in B_EPS_VALUES:
            H_s = make_H_smooth(eps, Phi)
            P_s, _ = solve_ps(H_s, dp, dz, B_R_SOLVER, B_L_SOLVER,
                               phi_bc="periodic")
            W_s, f_s, Pmax_s = compute_metrics(P_s, H_s, Phi, phi, Z)
            smooth_cache[(oil_key, eps)] = (W_s, f_s, Pmax_s)
            print(f"  Smooth {oil_name} ε={eps}: W={W_s:.4f}")

    for oil_key, oil_name, mu in B_OILS:
        for a_dim, b_dim in B_SIZES:
            a_phi = b_dim / B_R       # полуось по φ
            a_Z = a_dim / (B_L / 2)   # полуось по Z (2b и 2a, dim a тут по Z)

            for hp in B_HP_VALUES:
                depth = hp / B_C
                for zone_name, phi_start, phi_end in B_ZONES:
                    # Count dimples from Ptex (approx):
                    # Area_dimple = 4·a·b, zone area = (phi_end-phi_start)·L
                    zone_span_rad = np.deg2rad(phi_end - phi_start)
                    zone_area = zone_span_rad * B_R * B_L
                    dimple_area = 4 * a_dim * b_dim
                    for Ptex in B_PTEX_VALUES:
                        N_total = int(Ptex / 100.0 * zone_area / dimple_area)
                        if N_total < 1:
                            continue
                        # Распределение: пропорционально сторонам
                        phi_span = zone_span_rad
                        aspect = phi_span * B_R / B_L
                        nCth = max(1, int(np.sqrt(N_total * aspect)))
                        nCz = max(1, N_total // nCth)

                        phi_c, Z_c = make_dimple_centers(
                            nCth, nCz, phi_start, phi_end, a_phi_margin=0.0)
                        if len(phi_c) == 0:
                            continue

                        for profile in B_PROFILES:
                            direction = {"T2": "convergent",
                                         "T3": "divergent",
                                         "SQ": "flat"}[profile]
                            for eps in B_EPS_VALUES:
                                H_s_eps = make_H_smooth(eps, Phi)
                                H_t = add_wedge_dimples(
                                    H_s_eps, Phi, Zm, phi_c, Z_c,
                                    a_phi, a_Z, depth, direction=direction)
                                P_t, _ = solve_ps(
                                    H_t, dp, dz, B_R_SOLVER, B_L_SOLVER,
                                    phi_bc="periodic")
                                W_t, f_t, Pmax_t = compute_metrics(
                                    P_t, H_t, Phi, phi, Z)
                                W_s, f_s, Pmax_s = smooth_cache[(oil_key, eps)]
                                gw = W_t / W_s if W_s > 0 else 0
                                gf = f_t / f_s if f_s > 0 else 0
                                done += 1

                                results.append(dict(
                                    oil=oil_key, mu=mu, eps=eps,
                                    zone=zone_name,
                                    phi_start=phi_start, phi_end=phi_end,
                                    a_mm=a_dim * 1e3, b_mm=b_dim * 1e3,
                                    hp_um=hp * 1e6, Ptex=Ptex,
                                    nCth=nCth, nCz=nCz,
                                    n_dimples=len(phi_c),
                                    profile=profile,
                                    W_s=W_s, W_t=W_t, gain_W=gw,
                                    f_s=f_s, f_t=f_t, gain_f=gf,
                                ))

                                if done % 40 == 0:
                                    el = time.time() - t0_all
                                    print(f"    [{done}] "
                                          f"{oil_name[:3]} hp={hp*1e6:.0f} "
                                          f"zone={zone_name} Ptex={Ptex}% "
                                          f"eps={eps:.1f} {profile} "
                                          f"gW={gw:.4f} ({el:.0f}с)")

    dt_all = time.time() - t0_all
    print(f"\n  Время: {dt_all/60:.1f} мин, расчётов: {done}")

    # CSV
    csv_path = os.path.join(out_dir, "stage_b_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            row = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                   for k, v in r.items()}
            w.writerow(row)
    print(f"  CSV: {csv_path}")

    # Топ-10 по gain_W для каждого ε
    for eps in B_EPS_VALUES:
        subset = [r for r in results if r["eps"] == eps]
        subset.sort(key=lambda x: x["gain_W"], reverse=True)
        print(f"\n  ТОП-10 при ε={eps}:")
        print(f"  {'#':>3} {'масло':>8} {'prof':>4} {'zone':>13} "
              f"{'hp':>4} {'a/b':>7} {'Ptex':>5} {'nCth×nCz':>8} "
              f"{'gain_W':>8} {'gain_f':>8}")
        for i, r in enumerate(subset[:10]):
            print(f"  {i+1:3d} {r['oil'][:7]:>8s} {r['profile']:>4s} "
                  f"{r['zone']:>13s} {r['hp_um']:4.0f} "
                  f"{r['a_mm']:.1f}/{r['b_mm']:.1f} {r['Ptex']:>4d}% "
                  f"{r['nCth']}×{r['nCz']:<4d} "
                  f"{r['gain_W']:8.4f} {r['gain_f']:8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="A",
                        help="A, B, или AB (default: A)")
    parser.add_argument("--skip-check", action="store_true",
                        help="Stage B запустить даже если Stage A не прошёл")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "wedge_texture")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Результаты → {out_dir}\n")

    max_gain_T2 = 0.0
    if "A" in args.stage.upper():
        max_gain_T2, _ = run_stage_a(out_dir)

    if "B" in args.stage.upper():
        if max_gain_T2 <= 1.0 and not args.skip_check:
            print(f"\n!!! Stage A max gain_W (T2) = {max_gain_T2:.4f} ≤ 1.0")
            print(f"Stage B пропущен. Используй --skip-check чтобы запустить.")
        else:
            run_stage_b(out_dir)


if __name__ == "__main__":
    main()
