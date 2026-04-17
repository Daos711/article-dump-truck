#!/usr/bin/env python3
"""Stage C -- sector magnets only (conv_mag).

For each loadcase, use the best partial-groove pattern's active cells
as magnet positions. Sweep B_ref and solve conventional + sector magnet
equilibrium.
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

from models.texture_geometry import (
    get_herringbone_cell_centers,
    gu_groove_params_nondim,
)
from models.groove_sector_magnet import make_sector_magnet_model
from models.magnetic_equilibrium import (
    find_equilibrium, is_accepted, result_to_dict, result_status,
)
from cases.gu_loaded_side.schema import SCHEMA, classify_status, TOL_HARD
from cases.gu_loaded_side.common import (
    D, R, L, c, n_rpm, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    LOADCASE_NAMES,
    A_MAG_M2, N_MAG_EXP, G_REG_M, T_COVER_M, D_SUB_M,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
    BREF_SWEEP,
)


# ── Grid ─────────────────────────────────────────────────────────────

def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


# ── H_and_force closure (conventional — no grooves) ─────────────────

def _make_H_and_force_conv(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z):
    """Closure for conventional (smooth) bearing with PS solve."""
    omega = 2.0 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    def H_and_force(X, Y):
        H = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
        if sigma > 0:
            H = np.sqrt(H ** 2 + (sigma / c) ** 2)
        P, theta, _, _ = _ps(H, d_phi, d_Z, R, L,
                              tol=1e-6, max_iter=10_000_000)
        P_dim = P * p_scale
        Fx = -np.trapezoid(
            np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * R * L / 2.0
        Fy = -np.trapezoid(
            np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * R * L / 2.0
        h_dim = H * c
        h_min = float(np.min(h_dim))
        p_max = float(np.max(P_dim))
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
        tau_c = eta * omega * R / h_dim
        friction = float(
            np.sum(tau_c) * R * (2.0 * math.pi / H.shape[1])
            * L * (2.0 / H.shape[0]) / 2.0)
        return (float(Fx), float(Fy), h_min, p_max, cav_frac,
                friction, P, theta)

    return H_and_force


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage C: sector magnets (conv_mag)")
    parser.add_argument("--stageA", type=str, required=True,
                        help="path to Stage A output directory")
    parser.add_argument("--stageB", type=str, required=True,
                        help="path to Stage B output directory")
    parser.add_argument("--bref", type=float, nargs="+", default=None,
                        help="B_ref values (default: BREF_SWEEP from common)")
    parser.add_argument("--grid", type=str, default="1200x400",
                        help="grid NxM (default 1200x400)")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()

    bref_list = args.bref if args.bref is not None else BREF_SWEEP

    # ── Validate Stage A manifest ────────────────────────────────
    manifest_A_path = os.path.join(args.stageA,
                                   "working_geometry_manifest.json")
    with open(manifest_A_path, encoding="utf-8") as f:
        manifest_A = json.load(f)
    if manifest_A.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageA manifest "
              f"(expected {SCHEMA}, got {manifest_A.get('schema_version')})")
        sys.exit(1)

    loadcases = manifest_A["loadcases"]

    # ── Validate Stage B manifest ────────────────────────────────
    manifest_B_path = os.path.join(args.stageB, "partial_manifest.json")
    with open(manifest_B_path, encoding="utf-8") as f:
        manifest_B = json.load(f)
    if manifest_B.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageB manifest "
              f"(expected {SCHEMA}, got {manifest_B.get('schema_version')})")
        sys.exit(1)

    # ── Read best partial patterns ───────────────────────────────
    best_path = os.path.join(args.stageB, "partial_best_by_loadcase.json")
    with open(best_path, encoding="utf-8") as f:
        best_by_lc = json.load(f)

    # ── Grid setup ───────────────────────────────────────────────
    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    cell_centers = get_herringbone_cell_centers(N_g)

    print(f"Stage C: sector magnets (conv_mag)")
    print(f"Grid: {N_phi}x{N_Z}")
    print(f"B_ref sweep: {bref_list}")

    flat_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for lc_name in LOADCASE_NAMES:
        lc = loadcases[lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        Wa_norm = float(np.linalg.norm(W_applied))

        best = best_by_lc.get(lc_name)
        if best is None:
            print(f"\n  {lc_name}: no best pattern from Stage B — skipping")
            continue

        active_cells = [int(i) for i in best["active_cells"]]
        active_phi = np.array([float(cell_centers[i]) for i in active_cells])

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wa_norm:.1f}N  "
              f"active_cells={active_cells}")
        print(f"{'='*60}")

        H_and_force = _make_H_and_force_conv(
            Phi, Zm, phi_1D, Z_1D, d_phi, d_Z)

        for B_ref in bref_list:
            mag_model = make_sector_magnet_model(
                active_phi_centers=active_phi,
                c_m=c, d_sub_m=D_SUB_M,
                B_ref_T=B_ref,
                t_cover_m=T_COVER_M,
                A_mag_m2=A_MAG_M2,
                n_mag=N_MAG_EXP,
                g_reg_m=G_REG_M,
            )

            t0 = time.time()
            r = find_equilibrium(
                H_and_force, mag_model, W_applied,
                X0=0.0, Y0=-0.4,
                tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
                tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
            dt = time.time() - t0

            COF_eq = r.friction / max(Wa_norm, 1e-20)

            d = result_to_dict(r)
            d.update(dict(
                loadcase=lc_name,
                config="conv_mag",
                B_ref_T=float(B_ref),
                N_active=len(active_cells),
                active_cells=active_cells,
                COF_eq=float(COF_eq),
                elapsed_sec=float(dt),
                Fx_mag=float(r.Fx_mag),
                Fy_mag=float(r.Fy_mag),
                F_mag_N=float(math.sqrt(r.Fx_mag**2 + r.Fy_mag**2)),
            ))
            flat_rows.append(d)

            print(f"  B={B_ref:.2f}T: eps={r.eps:.4f} "
                  f"h_min={r.h_min*1e6:.1f}um "
                  f"p_max={r.p_max/1e6:.2f}MPa "
                  f"COF={COF_eq:.6f} "
                  f"|F_mag|={d['F_mag_N']:.3f}N "
                  f"res={r.rel_residual:.1e} "
                  f"[{r.status}] {dt:.1f}s")

    total_time = time.time() - t_global

    # ── Write sector_magnet_results.csv ──────────────────────────
    csv_path = os.path.join(args.out, "sector_magnet_results.csv")
    skip_keys = {"active_cells"}
    fields = sorted({k for r in flat_rows for k in r.keys()
                     if k not in skip_keys
                     and not isinstance(r.get(k), (dict, list, np.ndarray))})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})

    # ── Write sector_magnet_manifest.json ────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        stage="C_sector_magnets",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        stageA_manifest=manifest_A_path,
        stageB_manifest=manifest_B_path,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        bref_sweep=bref_list,
        magnet_params=dict(
            A_mag_mm2=A_MAG_M2 * 1e6,
            n_mag=N_MAG_EXP,
            g_reg_um=G_REG_M * 1e6,
            t_cover_um=T_COVER_M * 1e6,
            d_sub_um=D_SUB_M * 1e6,
        ),
        tol_accept=TOL_HARD,
        max_iter_NR=MAX_ITER_NR,
        n_solves=len(flat_rows),
        total_time_sec=total_time,
    )
    with open(os.path.join(args.out, "sector_magnet_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStage C complete: {len(flat_rows)} solves")
    print(f"Total: {total_time:.1f}s")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
