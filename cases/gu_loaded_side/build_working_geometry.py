#!/usr/bin/env python3
"""Stage A — freeze working geometry L/D=0.40 + build canonical loadcases.

Reads existing ld_sweep results to verify continuity, then solves
conventional fixed-ε cases on confirm grid to produce canonical anchors.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps = solve_payvar_salant_cpu

from models.texture_geometry import gu_groove_params_nondim
from cases.gu_loaded_side.schema import SCHEMA, classify_status
from cases.gu_loaded_side.common import (
    D, R, L, c, n_rpm, eta, sigma, LD_RATIO,
    w_g, L_g, d_g, beta_deg, N_g,
    EPS_REF, LOADCASE_NAMES, GRID_CONFIRM,
    working_geometry_dict, groove_geometry_dict,
)


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def solve_conventional(eps, phi_1D, Z_1D, Phi, Zm, d_phi, d_Z):
    H = 1.0 + float(eps) * np.cos(Phi)
    if sigma > 0:
        H = np.sqrt(H ** 2 + (sigma / c) ** 2)
    omega = 2 * math.pi * n_rpm / 60.0
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
    friction = float(
        np.sum(tau_c) * R * (2 * math.pi / H.shape[1])
        * L * (2.0 / H.shape[0]) / 2.0)
    COF = friction / max(W, 1e-20)
    # Pressure peak angle (loaded-side detection for Stage B)
    p_avg_phi = np.mean(P_dim, axis=0)
    phi_loaded_idx = int(np.argmax(p_avg_phi))
    phi_loaded_deg = float(np.degrees(phi_1D[phi_loaded_idx]))
    return dict(
        eps=float(eps), Fx_hydro=float(Fx), Fy_hydro=float(Fy),
        W_N=W, h_min=float(np.min(h_dim)),
        p_max=float(np.max(P_dim)),
        cav_frac=float(np.mean(theta < 1.0 - 1e-6)),
        friction=friction, COF=COF,
        phi_loaded_deg=phi_loaded_deg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-ld-sweep", type=str, default=None,
                        help="path to ld_sweep_v1 results for continuity check")
    parser.add_argument("--ratio", type=float, default=LD_RATIO)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N_phi, N_Z = GRID_CONFIRM
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)

    print(f"Stage A: working geometry L/D={args.ratio}")
    print(f"Grid: {N_phi}×{N_Z}")

    # Solve conventional at each eps_ref
    anchors: List[Dict[str, Any]] = []
    loadcases: Dict[str, Dict[str, Any]] = {}
    for eps, lc_name in zip(EPS_REF, LOADCASE_NAMES):
        t0 = time.time()
        m = solve_conventional(eps, phi_1D, Z_1D, Phi, Zm, d_phi, d_Z)
        dt = time.time() - t0
        m["loadcase"] = lc_name
        m["config"] = "conv_nomag"
        m["status"] = "anchor"
        m["elapsed_sec"] = dt
        anchors.append(m)
        loadcases[lc_name] = dict(
            eps_source=eps,
            applied_load_N=[-m["Fx_hydro"], -m["Fy_hydro"]],
            W_N=m["W_N"],
            phi_loaded_deg=m["phi_loaded_deg"],
        )
        print(f"  {lc_name}: eps={eps} W={m['W_N']:.1f}N "
              f"COF={m['COF']:.6f} phi_loaded={m['phi_loaded_deg']:.1f}° "
              f"{dt:.1f}s")

    # Continuity check vs ld_sweep
    continuity_pass = True
    if args.from_ld_sweep:
        ld_csv = os.path.join(args.from_ld_sweep, "ld_sweep_curves.csv")
        if os.path.exists(ld_csv):
            with open(ld_csv) as f:
                ld_rows = list(csv.DictReader(f))
            for a in anchors:
                match = [r for r in ld_rows
                         if abs(float(r["ratio_target"]) - args.ratio) < 0.01
                         and r["config"] == "conventional"
                         and abs(float(r["eps"]) - a["eps"]) < 1e-6
                         and r["grid_name"] == "confirm"]
                if match:
                    ld_cof = float(match[0]["COF"])
                    rel = abs(a["COF"] - ld_cof) / max(abs(ld_cof), 1e-20)
                    ok = rel < 0.01
                    continuity_pass = continuity_pass and ok
                    tag = "✓" if ok else "✗"
                    print(f"  [{tag}] continuity {a['loadcase']}: "
                          f"ΔCOF={rel:.4f} {'< 1%' if ok else '>= 1%'}")
        else:
            print("  [?] ld_sweep CSV not found — skipping continuity")

    overall = continuity_pass
    manifest = dict(
        schema_version=SCHEMA,
        stage="A_working_geometry",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        working_geometry=working_geometry_dict(),
        groove_geometry=groove_geometry_dict(),
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        loadcases=loadcases,
        continuity_pass=bool(continuity_pass),
        overall_pass=bool(overall),
    )
    with open(os.path.join(args.out, "working_geometry_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Anchor CSV
    csv_path = os.path.join(args.out, "anchor_cases.csv")
    fields = sorted({k for a in anchors for k in a.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in anchors:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in a.items()})

    # Loadcases CSV
    lc_csv = os.path.join(args.out, "loadcases.csv")
    with open(lc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "loadcase", "eps_source", "Fx_applied", "Fy_applied",
            "W_N", "phi_loaded_deg"])
        w.writeheader()
        for name, lc in loadcases.items():
            w.writerow(dict(
                loadcase=name,
                eps_source=lc["eps_source"],
                Fx_applied=f"{lc['applied_load_N'][0]:.8e}",
                Fy_applied=f"{lc['applied_load_N'][1]:.8e}",
                W_N=f"{lc['W_N']:.8e}",
                phi_loaded_deg=f"{lc['phi_loaded_deg']:.2f}",
            ))

    print(f"\nStage A: {'PASS' if overall else 'FAIL'}")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
