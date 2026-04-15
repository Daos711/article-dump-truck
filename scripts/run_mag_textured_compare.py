#!/usr/bin/env python3
"""Textured vs smooth под одинаковым mag_share.

Одна текстура (фиксированная): зона 0°–90°, a/b=1.5/1.2мм, hp=30мкм,
sf=1.5, profile=smoothcap. Равновесие через тот же 2D Newton-Raphson.
"""
import sys
import os
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

from models.magnetic_force import (
    MagneticForceModel, calibrate_Kmag, sanity_checks
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Расчёт ──────────────────────────────────────────────────────
N_PHI = 800
N_Z = 200
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * params.n / 60.0
P_SCALE = 6 * ETA * OMEGA * (params.R / params.c) ** 2
F0 = P_SCALE * params.R * params.L

W_APPLIED_Y_SHARE = 0.25         # соответствует baseline smooth case
MAG_SHARE_TARGETS = [0.0, 0.05, 0.10, 0.20, 0.30]

# Фиксированная текстура
TEX_PHI_START = 0.0
TEX_PHI_END = 90.0
TEX_A_MM = 1.5
TEX_B_MM = 1.2
TEX_HP_UM = 30
TEX_SF = 1.5
TEX_PROFILE = "smoothcap"


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_smooth_H0(X, Y, Phi):
    H0 = 1.0 + X * np.cos(Phi) + Y * np.sin(Phi)
    return np.sqrt(H0**2 + (params.sigma / params.c) ** 2)


def setup_texture(Phi, Zm, d_phi, d_Z):
    """Центры лунок в зоне [TEX_PHI_START, TEX_PHI_END]."""
    a_phi = (TEX_B_MM * 1e-3) / params.R         # полуось по φ в рад
    a_Z = 2 * (TEX_A_MM * 1e-3) / params.L        # полуось по Z безразм.
    depth = (TEX_HP_UM * 1e-6) / params.c

    phi_s = np.deg2rad(TEX_PHI_START)
    phi_e = np.deg2rad(TEX_PHI_END)
    phi_span = phi_e - phi_s

    N_phi_tex = max(1, int(phi_span / (TEX_SF * 2 * a_phi)))
    N_Z_tex = max(1, int(2.0 / (TEX_SF * 2 * a_Z)))

    # Центры
    margin = a_phi * 1.1
    usable = phi_span - 2 * margin
    if N_phi_tex == 1:
        phi_c = np.array([phi_s + phi_span / 2])
    else:
        phi_c = phi_s + margin + np.linspace(0, usable, N_phi_tex)

    margin_Z = a_Z * 1.1
    usable_Z = 2.0 - 2 * margin_Z
    if N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_c = -1.0 + margin_Z + np.linspace(0, usable_Z, N_Z_tex)

    pg, zg = np.meshgrid(phi_c, Z_c)
    return pg.ravel(), zg.ravel(), a_phi, a_Z, depth


def make_textured_H(X, Y, Phi, Zm, phi_c, Z_c, a_phi, a_Z, depth):
    H0 = make_smooth_H0(X, Y, Phi)
    H = create_H_with_ellipsoidal_depressions(
        H0, depth, Phi, Zm, phi_c, Z_c, a_Z, a_phi, profile=TEX_PROFILE)
    return H


def solve_ps(H, dp, dz):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, params.R, params.L, tol=1e-6, max_iter=10_000_000)
    return P, theta


def compute_hydro_force(P, Phi, phi_1D, Z_1D):
    P_dim = P * P_SCALE
    Fx = -np.trapezoid(np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * params.R * params.L / 2
    Fy = -np.trapezoid(np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * params.R * params.L / 2
    return float(Fx), float(Fy)


def compute_metrics(P, H, theta=None):
    P_dim = P * P_SCALE
    h_dim = H * params.c
    h_min = float(np.min(h_dim))
    p_max = float(np.max(P_dim))
    if theta is not None:
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    else:
        cav_frac = 0.0
    tau_c = ETA * OMEGA * params.R / h_dim
    F_friction = float(np.sum(tau_c) * params.R * (2 * np.pi / H.shape[1])
                       * params.L * (2 / H.shape[0]) / 2)
    return h_min, p_max, cav_frac, F_friction


def make_H_for(case, X, Y, Phi, Zm, tex_params):
    if case == "smooth":
        return make_smooth_H0(X, Y, Phi)
    return make_textured_H(X, Y, Phi, Zm, **tex_params)


def find_equilibrium(case, W_applied, mag_model, Phi, Zm, phi_1D, Z_1D,
                     d_phi, d_Z, tex_params, X0=0.0, Y0=-0.4,
                     max_iter=30, tol=1e-4):
    X, Y = X0, Y0
    dXY = 1e-5
    P_last, theta_last = None, None
    for it in range(max_iter):
        H = make_H_for(case, X, Y, Phi, Zm, tex_params)
        P, theta = solve_ps(H, d_phi, d_Z)
        Fx_h, Fy_h = compute_hydro_force(P, Phi, phi_1D, Z_1D)
        Fx_m, Fy_m = mag_model.force(X, Y)
        Rx = Fx_h + Fx_m - W_applied[0]
        Ry = Fy_h + Fy_m - W_applied[1]
        norm_R = np.sqrt(Rx**2 + Ry**2)
        rel_R = norm_R / max(np.linalg.norm(W_applied), 1e-20)
        if rel_R < tol:
            P_last, theta_last = P, theta
            break

        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0), (0, dXY)]):
            H_p = make_H_for(case, X + dX_, Y + dY_, Phi, Zm, tex_params)
            P_p, _ = solve_ps(H_p, d_phi, d_Z)
            Fxp, Fyp = compute_hydro_force(P_p, Phi, phi_1D, Z_1D)
            Fxm_p, Fym_p = mag_model.force(X + dX_, Y + dY_)
            J[0, col] = ((Fxp + Fxm_p) - (Fx_h + Fx_m)) / dXY
            J[1, col] = ((Fyp + Fym_p) - (Fy_h + Fy_m)) / dXY

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det
        step = min(1.0, 0.1 / max(abs(dX), abs(dY), 1e-10))
        X += step * dX
        Y += step * dY
        P_last, theta_last = P, theta

    eps = np.sqrt(X**2 + Y**2)
    H = make_H_for(case, X, Y, Phi, Zm, tex_params)
    h_min, p_max, cav_frac, fr = compute_metrics(P_last, H, theta_last)
    Fx_h, Fy_h = compute_hydro_force(P_last, Phi, phi_1D, Z_1D)
    Fx_m, Fy_m = mag_model.force(X, Y)
    Rx = Fx_h + Fx_m - W_applied[0]
    Ry = Fy_h + Fy_m - W_applied[1]
    rel_R = np.sqrt(Rx**2 + Ry**2) / max(np.linalg.norm(W_applied), 1e-20)
    attitude = np.rad2deg(np.arctan2(Y, X))
    return dict(
        case=case,
        X=float(X), Y=float(Y), eps=float(eps),
        attitude_deg=float(attitude),
        Fx_hydro=float(Fx_h), Fy_hydro=float(Fy_h),
        Fx_mag=float(Fx_m), Fy_mag=float(Fy_m),
        h_min=h_min, p_max=p_max, cav_frac=cav_frac, friction=fr,
        n_iter=it + 1, rel_residual=float(rel_R),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phi", type=int, default=N_PHI)
    parser.add_argument("--n-z", type=int, default=N_Z)
    parser.add_argument("--W-y-share", type=float, default=W_APPLIED_Y_SHARE)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "magnetic_pump")
    os.makedirs(out_dir, exist_ok=True)

    ok, _ = sanity_checks(verbose=True)
    if not ok:
        print("FAIL sanity")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("TEXTURED vs SMOOTH + MAGNETIC UNLOADING")
    print(f"Сетка {args.n_phi}×{args.n_z}")
    print(f"Texture: zone {TEX_PHI_START}-{TEX_PHI_END}°, "
          f"a/b={TEX_A_MM}/{TEX_B_MM}мм, hp={TEX_HP_UM}мкм, sf={TEX_SF}")
    print(f"mag_share targets: {MAG_SHARE_TARGETS}")
    print("=" * 72)

    W_applied = np.array([0.0, -args.W_y_share * F0])
    W_norm = float(np.linalg.norm(W_applied))
    e_load = -W_applied / W_norm

    phi, Z, Phi, Zm, dp, dz = make_grid(args.n_phi, args.n_z)
    phi_c, Z_c, a_phi, a_Z, depth = setup_texture(Phi, Zm, dp, dz)
    n_dimples = len(phi_c)
    tex_params = dict(phi_c=phi_c, Z_c=Z_c, a_phi=a_phi, a_Z=a_Z, depth=depth)
    print(f"\nТекстура: {n_dimples} лунок (полуоси в Ausas: "
          f"a_phi={a_phi:.4f} рад, a_Z={a_Z:.4f})")

    # Baseline smooth для калибровки K_mag
    print("\n[Baseline smooth]")
    mag0 = MagneticForceModel(K_mag=0.0)
    baseline = find_equilibrium("smooth", W_applied, mag0,
                                  Phi, Zm, phi, Z, dp, dz, tex_params)
    print(f"  X={baseline['X']:.4f}, Y={baseline['Y']:.4f}, "
          f"ε={baseline['eps']:.4f}, h_min={baseline['h_min']*1e6:.2f}мкм")

    results = {"smooth": [], "textured": []}
    X_prev = {"smooth": baseline["X"], "textured": baseline["X"]}
    Y_prev = {"smooth": baseline["Y"], "textured": baseline["Y"]}

    for target in MAG_SHARE_TARGETS:
        if target == 0.0:
            mag = MagneticForceModel(K_mag=0.0)
        else:
            mag = calibrate_Kmag(baseline["X"], baseline["Y"],
                                  W_applied, target)
        K_out = mag.K_mag

        for case in ["smooth", "textured"]:
            t0 = time.time()
            r = find_equilibrium(case, W_applied, mag, Phi, Zm, phi, Z,
                                  dp, dz, tex_params,
                                  X0=X_prev[case], Y0=Y_prev[case])
            dt_r = time.time() - t0
            X_prev[case], Y_prev[case] = r["X"], r["Y"]
            hs = ((r["Fx_hydro"] * e_load[0] + r["Fy_hydro"] * e_load[1])
                   / W_norm)
            ms = ((r["Fx_mag"] * e_load[0] + r["Fy_mag"] * e_load[1])
                   / W_norm)
            r.update(dict(mag_share_target=target, K_mag=K_out,
                          hydro_load_share=float(hs),
                          mag_load_share=float(ms)))
            results[case].append(r)
            print(f"  [{case:>8s}] share={target*100:4.1f}%: "
                  f"ε={r['eps']:.4f}, h_min={r['h_min']*1e6:.2f}мкм, "
                  f"p_max={r['p_max']/1e6:.2f}МПа, res={r['rel_residual']:.1e}, "
                  f"{dt_r:.1f}с")

    # Сравнение
    print("\n" + "=" * 72)
    print("СРАВНЕНИЕ textured/smooth")
    print("=" * 72)
    print(f"  {'share':>6} {'Δε':>8} {'h_t/h_s':>9} {'p_t/p_s':>9} "
          f"{'cav_t-s':>9} {'fr_t/fr_s':>10}")
    for i, target in enumerate(MAG_SHARE_TARGETS):
        rs = results["smooth"][i]
        rt = results["textured"][i]
        dE = rt["eps"] - rs["eps"]
        hr = rt["h_min"] / max(rs["h_min"], 1e-12)
        pr = rt["p_max"] / max(rs["p_max"], 1e-12)
        cd = rt["cav_frac"] - rs["cav_frac"]
        fr = rt["friction"] / max(rs["friction"], 1e-12)
        print(f"  {target*100:>5.1f}% {dE:>+8.4f} {hr:>9.4f} "
              f"{pr:>9.4f} {cd:>+9.4f} {fr:>10.4f}")

    # CSV
    csv_path = os.path.join(out_dir, "mag_textured_equilibrium.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results["smooth"][0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for case in ["smooth", "textured"]:
            for r in results[case]:
                w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                            for k, v in r.items()})
    print(f"\nCSV: {csv_path}")

    # JSON
    json_path = os.path.join(out_dir, "mag_textured_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "N_phi": args.n_phi, "N_Z": args.n_z,
                "W_applied_N": [float(w) for w in W_applied],
                "texture": {
                    "zone_deg": [TEX_PHI_START, TEX_PHI_END],
                    "a_mm": TEX_A_MM, "b_mm": TEX_B_MM,
                    "hp_um": TEX_HP_UM, "sf": TEX_SF,
                    "profile": TEX_PROFILE,
                    "n_dimples": int(n_dimples),
                },
                "mag_share_targets": MAG_SHARE_TARGETS,
            },
            "smooth": [{k: (v if isinstance(v, (int, float)) else float(v))
                        for k, v in r.items()} for r in results["smooth"]],
            "textured": [{k: (v if isinstance(v, (int, float)) else float(v))
                          for k, v in r.items()} for r in results["textured"]],
        }, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
