#!/usr/bin/env python3
"""Stage C' -- unloaded-side sector magnets (conv_mag + groove_mag).

For each loadcase and B_ref, solve both conventional + sector magnet
(conv_mag) and full-coverage herringbone + sector magnet (groove_mag)
equilibria.  Magnet placement uses get_placement_center_deg to position
an active subset on the unloaded side.

B_ref continuation: solve in order 0.30, 0.50, 0.80, 1.00 with
warm-start from previous solution.

Optional --diagnostic flag runs a single loaded-side point at L50 B=0.50.

Schema: gu_loaded_side_v1_1
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
from typing import Any, Dict, List, Tuple

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
    create_H_with_herringbone_grooves,
    get_herringbone_cell_centers,
    gu_groove_params_nondim,
)
from models.groove_sector_magnet import (
    make_sector_magnet_model,
    build_active_subset,
    get_placement_center_deg,
)
from models.magnetic_equilibrium import (
    find_equilibrium, result_to_dict,
)
from cases.gu_loaded_side.schema import (
    SCHEMA, resolve_stage_dir, classify_status, TOL_HARD,
)
from cases.gu_loaded_side.common import (
    D, R, L, c, n_rpm, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    LOADCASE_NAMES,
    A_MAG_M2, N_MAG_EXP, G_REG_M, T_COVER_M, D_SUB_M,
    GRID_CONFIRM,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
    BREF_SWEEP,
)


# -- Grid -----------------------------------------------------------------

def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


# -- H_and_force closures ------------------------------------------------

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


def _make_H_and_force_groove(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                              groove_params):
    """Closure for full-coverage herringbone groove bearing with PS solve."""
    omega = 2.0 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    def H_and_force(X, Y):
        H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
        if sigma > 0:
            H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
        H = create_H_with_herringbone_grooves(
            H0, groove_params["depth_nondim"], Phi, Zm,
            groove_params["N_g"], groove_params["w_g_nondim"],
            groove_params["L_g_nondim"], groove_params["beta_deg"])
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


# -- Solve helper ---------------------------------------------------------

def _solve_one(H_and_force, mag_model, W_applied, X0, Y0):
    """Run find_equilibrium and return (result, elapsed_sec)."""
    t0 = time.time()
    r = find_equilibrium(
        H_and_force, mag_model, W_applied,
        X0=X0, Y0=Y0,
        tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
        tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
    return r, time.time() - t0


def _row_from_result(r, dt, lc_name, config, B_ref, active_phi, Wa_norm):
    """Build a flat dict from an EquilibriumResult."""
    COF_eq = r.friction / max(Wa_norm, 1e-20)
    status = classify_status(r.rel_residual, r.converged)
    d = result_to_dict(r)
    d.update(dict(
        loadcase=lc_name,
        config=config,
        B_ref_T=float(B_ref),
        N_active=len(active_phi),
        COF_eq=float(COF_eq),
        elapsed_sec=float(dt),
        classify_status=status,
        Fx_mag=float(r.Fx_mag),
        Fy_mag=float(r.Fy_mag),
        F_mag_N=float(math.sqrt(r.Fx_mag**2 + r.Fy_mag**2)),
    ))
    return d


# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage C': unloaded-side sector magnets")
    parser.add_argument("--stageA", type=str, required=True,
                        help="path to Stage A output directory")
    parser.add_argument("--stageB", type=str, required=True,
                        help="path to Stage B' output directory")
    parser.add_argument("--placement-mode", type=str, default="unloaded",
                        help="magnet placement mode (default: unloaded)")
    parser.add_argument("--N-mag-active", type=int, default=3,
                        help="number of active magnet cells (default: 3)")
    parser.add_argument("--bref", type=float, nargs="+", default=None,
                        help="B_ref values (default: 0.30 0.50 0.80 1.00)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="run loaded-side diagnostic at L50 B=0.50")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()

    bref_list = args.bref if args.bref is not None else BREF_SWEEP
    N_active = args.N_mag_active
    placement_mode = args.placement_mode

    # -- Validate Stage A manifest ----------------------------------------
    stageA_dir = resolve_stage_dir(args.stageA)
    manifest_A_path = os.path.join(stageA_dir,
                                   "working_geometry_manifest.json")
    with open(manifest_A_path, encoding="utf-8") as f:
        manifest_A = json.load(f)
    if manifest_A.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageA manifest "
              f"(expected {SCHEMA}, got {manifest_A.get('schema_version')})")
        sys.exit(1)

    loadcases = manifest_A["loadcases"]

    # -- Validate Stage B manifest ----------------------------------------
    stageB_dir = resolve_stage_dir(args.stageB)
    manifest_B_path = os.path.join(stageB_dir,
                                   "full_groove_anchor_manifest.json")
    with open(manifest_B_path, encoding="utf-8") as f:
        manifest_B = json.load(f)
    if manifest_B.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageB manifest "
              f"(expected {SCHEMA}, got {manifest_B.get('schema_version')})")
        sys.exit(1)

    # -- Grid setup -------------------------------------------------------
    N_phi, N_Z = GRID_CONFIRM
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    groove_params = gu_groove_params_nondim(
        D, L, c, R, w_g, L_g, d_g, beta_deg, N_g)
    cell_centers = get_herringbone_cell_centers(N_g)

    print(f"Stage C': sector magnets ({placement_mode}-side)")
    print(f"Grid: {N_phi}x{N_Z}")
    print(f"B_ref sweep: {bref_list}")
    print(f"N_active: {N_active}")

    H_and_force_conv = _make_H_and_force_conv(
        Phi, Zm, phi_1D, Z_1D, d_phi, d_Z)
    H_and_force_groove = _make_H_and_force_groove(
        Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, groove_params)

    flat_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for lc_name in LOADCASE_NAMES:
        lc = loadcases[lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        Wa_norm = float(np.linalg.norm(W_applied))

        phi_loaded_deg = float(lc["phi_loaded_deg"])
        phi_mag_center = get_placement_center_deg(phi_loaded_deg,
                                                   placement_mode)
        active_phi = build_active_subset(cell_centers, phi_mag_center,
                                          N_active)

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wa_norm:.1f}N  "
              f"phi_loaded={phi_loaded_deg:.1f}deg  "
              f"phi_mag_center={phi_mag_center:.1f}deg")
        print(f"Active phi (deg): "
              f"{[f'{math.degrees(p):.1f}' for p in active_phi]}")
        print(f"{'='*60}")

        # Warm-start seeds for B_ref continuation
        X_seed_conv, Y_seed_conv = 0.0, -0.4
        X_seed_groove, Y_seed_groove = 0.0, -0.4

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

            # -- conv_mag --
            r_cm, dt_cm = _solve_one(
                H_and_force_conv, mag_model, W_applied,
                X_seed_conv, Y_seed_conv)
            row_cm = _row_from_result(
                r_cm, dt_cm, lc_name, "conv_mag", B_ref,
                active_phi, Wa_norm)
            flat_rows.append(row_cm)

            # Warm-start from previous
            X_seed_conv, Y_seed_conv = r_cm.X, r_cm.Y

            print(f"  B={B_ref:.2f}T conv_mag: eps={r_cm.eps:.4f} "
                  f"h_min={r_cm.h_min*1e6:.1f}um "
                  f"COF={row_cm['COF_eq']:.6f} "
                  f"|F_mag|={row_cm['F_mag_N']:.3f}N "
                  f"res={r_cm.rel_residual:.1e} "
                  f"[{row_cm['classify_status']}] {dt_cm:.1f}s")

            # -- groove_mag --
            r_gm, dt_gm = _solve_one(
                H_and_force_groove, mag_model, W_applied,
                X_seed_groove, Y_seed_groove)
            row_gm = _row_from_result(
                r_gm, dt_gm, lc_name, "groove_mag", B_ref,
                active_phi, Wa_norm)
            flat_rows.append(row_gm)

            # Warm-start from previous
            X_seed_groove, Y_seed_groove = r_gm.X, r_gm.Y

            print(f"  B={B_ref:.2f}T groove_mag: eps={r_gm.eps:.4f} "
                  f"h_min={r_gm.h_min*1e6:.1f}um "
                  f"COF={row_gm['COF_eq']:.6f} "
                  f"|F_mag|={row_gm['F_mag_N']:.3f}N "
                  f"res={r_gm.rel_residual:.1e} "
                  f"[{row_gm['classify_status']}] {dt_gm:.1f}s")

    # -- Diagnostic: loaded-side at L50 B=0.50 ----------------------------
    if args.diagnostic:
        diag_lc = "L50"
        diag_B = 0.50
        lc_diag = loadcases[diag_lc]
        W_diag = np.array(lc_diag["applied_load_N"], dtype=float)
        Wa_diag = float(np.linalg.norm(W_diag))
        phi_loaded_deg_diag = float(lc_diag["phi_loaded_deg"])
        phi_diag_center = get_placement_center_deg(
            phi_loaded_deg_diag, "loaded")
        active_phi_diag = build_active_subset(
            cell_centers, phi_diag_center, N_active)

        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC: loaded-side at {diag_lc} B={diag_B}T")
        print(f"phi_mag_center={phi_diag_center:.1f}deg (loaded)")
        print(f"{'='*60}")

        mag_diag = make_sector_magnet_model(
            active_phi_centers=active_phi_diag,
            c_m=c, d_sub_m=D_SUB_M,
            B_ref_T=diag_B,
            t_cover_m=T_COVER_M,
            A_mag_m2=A_MAG_M2,
            n_mag=N_MAG_EXP,
            g_reg_m=G_REG_M,
        )

        for config_label, hf in [("conv_mag_loaded", H_and_force_conv),
                                  ("groove_mag_loaded", H_and_force_groove)]:
            r_d, dt_d = _solve_one(hf, mag_diag, W_diag, 0.0, -0.4)
            row_d = _row_from_result(
                r_d, dt_d, diag_lc, config_label, diag_B,
                active_phi_diag, Wa_diag)
            row_d["placement_mode"] = "loaded"
            flat_rows.append(row_d)

            print(f"  {config_label}: eps={r_d.eps:.4f} "
                  f"h_min={r_d.h_min*1e6:.1f}um "
                  f"COF={row_d['COF_eq']:.6f} "
                  f"|F_mag|={row_d['F_mag_N']:.3f}N "
                  f"res={r_d.rel_residual:.1e} "
                  f"[{row_d['classify_status']}] {dt_d:.1f}s")

    total_time = time.time() - t_global

    # -- Write sector_magnet_results.csv ----------------------------------
    csv_path = os.path.join(args.out, "sector_magnet_results.csv")
    _write_csv(csv_path, flat_rows)

    # -- Write sector_magnet_manifest.json --------------------------------
    manifest = dict(
        schema_version=SCHEMA,
        stage="Cp_sector_magnets_v11",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        stageA_manifest=manifest_A_path,
        stageB_manifest=manifest_B_path,
        placement_mode=placement_mode,
        N_active=N_active,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        bref_sweep=[float(b) for b in bref_list],
        groove_params=groove_params,
        magnet_params=dict(
            A_mag_mm2=A_MAG_M2 * 1e6,
            n_mag=N_MAG_EXP,
            g_reg_um=G_REG_M * 1e6,
            t_cover_um=T_COVER_M * 1e6,
            d_sub_um=D_SUB_M * 1e6,
        ),
        tol_accept=TOL_HARD,
        max_iter_NR=MAX_ITER_NR,
        diagnostic=args.diagnostic,
        n_solves=len(flat_rows),
        total_time_sec=total_time,
    )
    with open(os.path.join(args.out, "sector_magnet_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStage C' complete: {len(flat_rows)} solves")
    print(f"Total: {total_time:.1f}s")
    print(f"Artifacts: {args.out}/")


# -- Helpers --------------------------------------------------------------

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    skip_keys = {"active_cells"}
    fields = sorted({k for r in rows for k in r.keys()
                     if k not in skip_keys
                     and not isinstance(r.get(k), (dict, list, np.ndarray))})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})


if __name__ == "__main__":
    main()
