#!/usr/bin/env python3
"""Textured vs smooth сравнение с магнитной разгрузкой.

Только на accepted targets из smooth run. Использует тот же shared
driver (models/magnetic_equilibrium) — идентичный NR и acceptance.
"""
import sys
import os
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

from models.magnetic_force import (
    RadialUnloadForceModel, sanity_checks,
    calibrate_Kmag_from_baseline_projection,
)
from models.magnetic_equilibrium import find_equilibrium
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Config ──────────────────────────────────────────────────────
N_PHI = 800
N_Z = 200
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * params.n / 60.0
P_SCALE = 6 * ETA * OMEGA * (params.R / params.c) ** 2
F0 = P_SCALE * params.R * params.L

# Reference texture
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


def setup_texture(Phi, Zm):
    a_phi = (TEX_B_MM * 1e-3) / params.R
    a_Z = 2 * (TEX_A_MM * 1e-3) / params.L
    depth = (TEX_HP_UM * 1e-6) / params.c

    phi_s = np.deg2rad(TEX_PHI_START)
    phi_e = np.deg2rad(TEX_PHI_END)
    phi_span = phi_e - phi_s
    N_phi_tex = max(1, int(phi_span / (TEX_SF * 2 * a_phi)))
    N_Z_tex = max(1, int(2.0 / (TEX_SF * 2 * a_Z)))

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


def make_H_and_force_factory(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                              textured=False, tex=None):
    """Замыкание: build_H(X,Y) [+ texture] + PS solve + metrics."""

    def H_and_force(X, Y):
        H0 = make_smooth_H0(X, Y, Phi)
        if textured and tex is not None:
            H = create_H_with_ellipsoidal_depressions(
                H0, tex["depth"], Phi, Zm,
                tex["phi_c"], tex["Z_c"],
                tex["a_Z"], tex["a_phi"], profile=TEX_PROFILE)
        else:
            H = H0
        P, theta, _, _ = _ps_solver(
            H, d_phi, d_Z, params.R, params.L,
            tol=1e-6, max_iter=10_000_000)
        P_dim = P * P_SCALE
        Fx = -np.trapezoid(
            np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * params.R * params.L / 2
        Fy = -np.trapezoid(
            np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * params.R * params.L / 2
        h_dim = H * params.c
        h_min = float(np.min(h_dim))
        p_max = float(np.max(P_dim))
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
        tau_c = ETA * OMEGA * params.R / h_dim
        friction = float(
            np.sum(tau_c) * params.R * (2 * np.pi / H.shape[1])
            * params.L * (2 / H.shape[0]) / 2)
        return (float(Fx), float(Fy), h_min, p_max, cav_frac,
                friction, P, theta)

    return H_and_force


def result_to_dict(r, case=""):
    d = dict(
        case=case,
        X=r.X, Y=r.Y, eps=r.eps, attitude_deg=r.attitude_deg,
        Fx_hydro=r.Fx_hydro, Fy_hydro=r.Fy_hydro,
        Fx_mag=r.Fx_mag, Fy_mag=r.Fy_mag,
        h_min=r.h_min, p_max=r.p_max, cav_frac=r.cav_frac,
        friction=r.friction,
        rel_residual=r.rel_residual, n_iter=r.n_iter,
        converged=bool(r.converged),
        unload_share_target=r.unload_share_target,
        unload_share_actual=r.unload_share_actual,
        hydro_share_actual=r.hydro_share_actual,
        K_mag=r.K_mag,
    )
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phi", type=int, default=N_PHI)
    parser.add_argument("--n-z", type=int, default=N_Z)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "magnetic_pump")
    # Загрузить smooth summary
    smooth_json = os.path.join(out_dir, "mag_smooth_summary.json")
    if not os.path.exists(smooth_json):
        print(f"Сначала запусти run_mag_smooth_continuation.py "
              f"(нет {smooth_json})")
        sys.exit(1)
    with open(smooth_json, "r", encoding="utf-8") as f:
        sm = json.load(f)

    ok, _ = sanity_checks(verbose=True)
    if not ok:
        print("FAIL sanity")
        sys.exit(1)

    # Accepted smooth points
    accepted = sm["continuation"]
    if not accepted:
        print("В smooth нет accepted targets")
        sys.exit(1)

    W_applied = np.array(sm["config"]["W_applied_N"])
    baseline_X = sm["baseline"]["X"]
    baseline_Y = sm["baseline"]["Y"]

    print("\n" + "=" * 72)
    print("TEXTURED vs SMOOTH + MAGNETIC UNLOAD (accepted targets only)")
    acc_str = ", ".join(f"{r['unload_share_target']*100:.1f}%"
                         for r in accepted)
    print(f"Accepted: [{acc_str}]")
    print("=" * 72)

    phi, Z, Phi, Zm, dp, dz = make_grid(args.n_phi, args.n_z)
    phi_c, Z_c, a_phi, a_Z, depth = setup_texture(Phi, Zm)
    n_dimples = len(phi_c)
    tex = dict(phi_c=phi_c, Z_c=Z_c, a_phi=a_phi, a_Z=a_Z, depth=depth)
    print(f"Texture: {n_dimples} лунок, zone "
          f"{TEX_PHI_START}-{TEX_PHI_END}°")

    H_and_force_smooth = make_H_and_force_factory(
        Phi, Zm, phi, Z, dp, dz, textured=False)
    H_and_force_tex = make_H_and_force_factory(
        Phi, Zm, phi, Z, dp, dz, textured=True, tex=tex)

    results = []  # list of dict
    X_prev_s, Y_prev_s = baseline_X, baseline_Y
    X_prev_t, Y_prev_t = baseline_X, baseline_Y

    for sm_entry in accepted:
        target = sm_entry["unload_share_target"]
        # Калибровка K_mag на baseline (smooth)
        if target == 0.0:
            m_smooth = RadialUnloadForceModel(K_mag=0.0)
            m_tex = RadialUnloadForceModel(K_mag=0.0)
        else:
            template = RadialUnloadForceModel(
                n_mag=3, H_reg=0.05, H_floor=0.02)
            m_smooth = calibrate_Kmag_from_baseline_projection(
                template, baseline_X, baseline_Y, W_applied, target)
            m_tex = calibrate_Kmag_from_baseline_projection(
                template, baseline_X, baseline_Y, W_applied, target)

        # Smooth solve (для consistency, хотя уже есть в JSON)
        t0 = time.time()
        rs = find_equilibrium(
            H_and_force_smooth, m_smooth, W_applied,
            X0=X_prev_s, Y0=Y_prev_s,
            tol=1e-3, step_cap=0.05, eps_max=0.90)
        dt_s = time.time() - t0
        rs.unload_share_target = target

        # Textured solve только если smooth сошёлся
        if rs.converged and rs.rel_residual < 1e-3:
            t0 = time.time()
            rt = find_equilibrium(
                H_and_force_tex, m_tex, W_applied,
                X0=X_prev_t, Y0=Y_prev_t,
                tol=1e-3, step_cap=0.05, eps_max=0.90)
            dt_t = time.time() - t0
            rt.unload_share_target = target
            accepted_flag = (rt.converged and rt.rel_residual < 1e-3)
            X_prev_s, Y_prev_s = rs.X, rs.Y
            if accepted_flag:
                X_prev_t, Y_prev_t = rt.X, rt.Y
        else:
            rt = None
            accepted_flag = False
            dt_t = 0.0

        ds = result_to_dict(rs, "smooth")
        dt_d = result_to_dict(rt, "textured") if rt is not None else None

        hr = rt.h_min / max(rs.h_min, 1e-12) if rt else None
        pr = rt.p_max / max(rs.p_max, 1e-12) if rt else None
        fr_r = rt.friction / max(rs.friction, 1e-12) if rt else None
        dcav = rt.cav_frac - rs.cav_frac if rt else None
        deps = rt.eps - rs.eps if rt else None

        print(f"  target={target*100:5.2f}%: "
              f"smooth ε={rs.eps:.4f} (res={rs.rel_residual:.1e}), "
              f"{'tex ε=' + f'{rt.eps:.4f}' if rt else 'tex SKIPPED'} "
              f"({dt_s+dt_t:.1f}с) "
              f"{'✓' if accepted_flag else '✗'}")
        if rt:
            print(f"    h_t/h_s={hr:.4f}, p_t/p_s={pr:.4f}, "
                  f"fr_t/fr_s={fr_r:.4f}, Δcav={dcav:+.4f}, "
                  f"Δε={deps:+.4f}")

        results.append(dict(
            unload_share_target=target,
            smooth=ds,
            textured=dt_d,
            ratios=dict(
                h_ratio=float(hr) if hr is not None else None,
                p_ratio=float(pr) if pr is not None else None,
                f_ratio=float(fr_r) if fr_r is not None else None,
                delta_cav=float(dcav) if dcav is not None else None,
                delta_eps=float(deps) if deps is not None else None,
            ),
            accepted=accepted_flag,
        ))

    # CSV: flatten
    csv_path = os.path.join(out_dir, "mag_textured_equilibrium.csv")
    rows = []
    for r in results:
        base = dict(unload_share_target=r["unload_share_target"],
                     accepted=r["accepted"])
        if r["smooth"]:
            for k, v in r["smooth"].items():
                base[f"s_{k}"] = v
        if r["textured"]:
            for k, v in r["textured"].items():
                base[f"t_{k}"] = v
        for k, v in r["ratios"].items():
            base[k] = v if v is not None else ""
        rows.append(base)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\nCSV: {csv_path}")

    json_path = os.path.join(out_dir, "mag_textured_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "N_phi": args.n_phi, "N_Z": args.n_z,
                "W_applied_N": [float(w_) for w_ in W_applied],
                "texture": dict(
                    zone_deg=[TEX_PHI_START, TEX_PHI_END],
                    a_mm=TEX_A_MM, b_mm=TEX_B_MM,
                    hp_um=TEX_HP_UM, sf=TEX_SF, profile=TEX_PROFILE,
                    n_dimples=int(n_dimples)),
            },
            "pairs": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
