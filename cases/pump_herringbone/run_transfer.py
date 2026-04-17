#!/usr/bin/env python3
"""Stage 2 — transfer herringbone design to pump geometry.

Two modes:
  * scaled_nondim — preserve w_g/D, L_g/D, d_g/c (N_g=10 fixed)
  * same_mm — preserve absolute groove dimensions (mm)

Same eps sweep, same COF metric. No magnets.

Requires: Stage 1 manifest with overall_pass=True.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps = solve_payvar_salant_cpu

from models.texture_geometry import (
    create_H_with_straight_grooves,
    create_H_with_herringbone_grooves,
    gu_groove_params_nondim,
    transfer_groove_params,
)
from cases.gu_2020 import config_gu as gu_cfg
from config import pump_params as pump
from config.oil_properties import MINERAL_OIL

SCHEMA = "herringbone_pump_v1"
TEXTURE_TYPES = ["conventional", "straight_grooves", "herringbone_grooves"]

# Pump constants
ETA_PUMP = MINERAL_OIL["eta_pump"]
D_PUMP = 2 * pump.R
R_PUMP = pump.R
L_PUMP = pump.L
C_PUMP = pump.c
N_PUMP = pump.n
SIGMA_PUMP = pump.sigma

EPS_LIST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def make_grid(N_phi: int, N_Z: int):
    phi = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def build_H(eps, Phi, Z, texture_type, groove, c, sigma):
    H0 = 1.0 + float(eps) * np.cos(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
    if texture_type == "conventional":
        return H0
    if texture_type == "straight_grooves":
        return create_H_with_straight_grooves(
            H0, groove["depth_nondim"], Phi, Z,
            groove["N_g"], groove["w_g_nondim"], groove["L_g_nondim"])
    if texture_type == "herringbone_grooves":
        return create_H_with_herringbone_grooves(
            H0, groove["depth_nondim"], Phi, Z,
            groove["N_g"], groove["w_g_nondim"], groove["L_g_nondim"],
            groove["beta_deg"])
    raise ValueError(texture_type)


def solve_metrics(H, Phi, phi_1D, Z_1D, d_phi, d_Z,
                   R, L, c, eta, n_rpm):
    omega = 2.0 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    P, theta, _, _ = _ps(H, d_phi, d_Z, R, L,
                          tol=1e-6, max_iter=10_000_000)
    P_dim = P * p_scale
    Fx = -np.trapezoid(
        np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R * L / 2.0
    Fy = -np.trapezoid(
        np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R * L / 2.0
    W = float(math.sqrt(Fx ** 2 + Fy ** 2))
    h_dim = H * c
    tau_c = eta * omega * R / h_dim
    F_fr = float(np.sum(tau_c) * R * (2.0 * math.pi / H.shape[1])
                 * L * (2.0 / H.shape[0]) / 2.0)
    COF = F_fr / max(W, 1e-20)
    return dict(W=W, F_friction=F_fr, COF=COF,
                h_min=float(np.min(h_dim)),
                p_max=float(np.max(P_dim)),
                cav_frac=float(np.mean(theta < 1.0 - 1e-6)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="800x200")
    parser.add_argument("--mode", choices=["scaled_nondim", "same_mm", "both"],
                        default="both")
    parser.add_argument("--skip-validation-check", action="store_true")
    args = parser.parse_args()

    # Check Stage 1 PASS
    if not args.skip_validation_check:
        val_manifest = os.path.join(ROOT, "results", "herringbone_gu_v1",
                                    "gu_validation_manifest.json")
        if not os.path.exists(val_manifest):
            print(f"FAIL: Stage 1 не пройден (нет {val_manifest})")
            sys.exit(1)
        with open(val_manifest) as f:
            vm = json.load(f)
        if not vm.get("overall_pass"):
            print(f"FAIL: Stage 1 overall_pass=False. Fix validation first.")
            sys.exit(1)

    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    out_dir = os.path.join(ROOT, "results", "herringbone_pump_v1")
    os.makedirs(out_dir, exist_ok=True)

    gu_ratios = gu_groove_params_nondim(
        gu_cfg.D, gu_cfg.L, gu_cfg.c, gu_cfg.R,
        gu_cfg.w_g, gu_cfg.L_g, gu_cfg.d_g, gu_cfg.beta_deg, gu_cfg.N_g)
    gu_ratios["D_source_m"] = gu_cfg.D
    gu_ratios["c_source_m"] = gu_cfg.c

    modes = (["scaled_nondim", "same_mm"] if args.mode == "both"
             else [args.mode])
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)

    all_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for mode in modes:
        groove = transfer_groove_params(
            gu_ratios, D_PUMP, L_PUMP, C_PUMP, R_PUMP,
            N_g=gu_cfg.N_g, beta_deg=gu_cfg.beta_deg, mode=mode)
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"Groove nondim: w_g={groove['w_g_nondim']:.4f} rad, "
              f"L_g={groove['L_g_nondim']:.4f}, "
              f"depth={groove['depth_nondim']:.4f}")
        print(f"{'='*60}")

        for tt in TEXTURE_TYPES:
            for eps in EPS_LIST:
                t0 = time.time()
                H = build_H(eps, Phi, Zm, tt, groove, C_PUMP, SIGMA_PUMP)
                m = solve_metrics(H, Phi, phi_1D, Z_1D, d_phi, d_Z,
                                  R_PUMP, L_PUMP, C_PUMP, ETA_PUMP, N_PUMP)
                dt = time.time() - t0
                row = dict(mode=mode, grid=f"{N_phi}x{N_Z}",
                           texture_type=tt, eps=eps, **m,
                           elapsed_sec=dt)
                all_rows.append(row)
                print(f"  {tt:>22s}  eps={eps:.1f}  COF={m['COF']:.6f}  "
                      f"W={m['W']:.0f}N  {dt:.1f}s")

    total = time.time() - t_global
    manifest = dict(
        schema_version=SCHEMA,
        stage="pump_transfer",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        bearing=dict(D_mm=D_PUMP * 1e3, L_mm=L_PUMP * 1e3,
                     c_um=C_PUMP * 1e6, n_rpm=N_PUMP,
                     eta_Pa_s=ETA_PUMP, sigma_um=SIGMA_PUMP * 1e6),
        friction_model="couette_only",
        modes=modes,
        eps_list=EPS_LIST,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        total_time_sec=total,
    )
    with open(os.path.join(out_dir, "pump_transfer_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "pump_transfer_curves.csv")
    fields = sorted({k for r in all_rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nArtifacts: {out_dir}/")
    print(f"Total: {total:.1f}s")


if __name__ == "__main__":
    main()
