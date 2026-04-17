#!/usr/bin/env python3
"""Stage B' -- full-coverage herringbone groove anchor (no magnets).

Uses inline NR with backtracking (§5.2) and strict HS warmup (§5.3).
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
    gu_groove_params_nondim,
)
from cases.gu_loaded_side.schema import (
    SCHEMA, resolve_stage_dir, classify_status, TOL_HARD,
)
from cases.gu_loaded_side.common import (
    D, R, L, c, n_rpm, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    LOADCASE_NAMES,
    GRID_CONFIRM,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
    ps_solve,
)


def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def _eval(X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
          groove_params, omega, p_scale):
    """One H build + PS solve + force integration."""
    H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
    H = create_H_with_herringbone_grooves(
        H0, groove_params["depth_nondim"], Phi, Zm,
        groove_params["N_g"], groove_params["w_g_nondim"],
        groove_params["L_g_nondim"], groove_params["beta_deg"])
    P, theta, _, _ = ps_solve(_ps, H, d_phi, d_Z, R, L)
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
    return float(Fx), float(Fy), h_min, p_max, cav_frac, friction


def _residual(Fx_h, Fy_h, W_applied, Wa_norm):
    Rx = Fx_h - W_applied[0]
    Ry = Fy_h - W_applied[1]
    return math.sqrt(Rx ** 2 + Ry ** 2) / max(Wa_norm, 1e-20)


def solve_groove_equilibrium(
        W_applied, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        groove_params, X0=0.0, Y0=-0.4):
    """NR with backtracking (§5.2) + strict warmup via ps_solve."""
    omega = 2.0 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    Wa_norm = float(np.linalg.norm(W_applied))
    dXY = 1e-4

    X, Y = float(X0), float(Y0)
    Fx_h, Fy_h, h_min, p_max, cav, fr = _eval(
        X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        groove_params, omega, p_scale)
    rel_R = _residual(Fx_h, Fy_h, W_applied, Wa_norm)

    converged = False
    n_it = 0
    bt_count = 0

    for _ in range(MAX_ITER_NR):
        if rel_R < TOL_HARD:
            converged = True
            break

        # Central-difference Jacobian
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            Fxp, Fyp, *_ = _eval(
                X + dX_, Y + dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, omega, p_scale)
            Fxn, Fyn, *_ = _eval(
                X - dX_, Y - dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, omega, p_scale)
            J[0, col] = (Fxp - Fxn) / (2.0 * dXY)
            J[1, col] = (Fyp - Fyn) / (2.0 * dXY)

        Rx = Fx_h - W_applied[0]
        Ry = Fy_h - W_applied[1]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det

        # Step cap
        cap = STEP_CAP / max(abs(dX), abs(dY), 1e-20)
        if cap < 1.0:
            dX *= cap
            dY *= cap

        # Backtracking (§5.2)
        accepted = False
        for alpha in [1.0, 0.5, 0.25, 0.125]:
            X_try = X + alpha * dX
            Y_try = Y + alpha * dY
            eps_try = math.sqrt(X_try ** 2 + Y_try ** 2)
            if eps_try >= EPS_MAX:
                continue
            Fx_t, Fy_t, hm_t, pm_t, cv_t, fr_t = _eval(
                X_try, Y_try, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, omega, p_scale)
            rel_t = _residual(Fx_t, Fy_t, W_applied, Wa_norm)
            if rel_t < rel_R:
                if alpha < 1.0:
                    bt_count += 1
                    print(f"    [BT] alpha={alpha} "
                          f"res: {rel_R:.1e} → {rel_t:.1e}")
                X, Y = X_try, Y_try
                Fx_h, Fy_h = Fx_t, Fy_t
                h_min, p_max, cav, fr = hm_t, pm_t, cv_t, fr_t
                rel_R = rel_t
                accepted = True
                break
        if not accepted:
            break
        n_it += 1

    eps = math.sqrt(X ** 2 + Y ** 2)
    attitude = math.degrees(math.atan2(Y, X))
    COF_eq = fr / max(Wa_norm, 1e-20)
    status = classify_status(rel_R, rel_R <= 0.10)
    return dict(
        X=X, Y=Y, eps=eps, attitude_deg=attitude,
        Fx_hydro=Fx_h, Fy_hydro=Fy_h,
        h_min=h_min, p_max=p_max, cav_frac=cav,
        friction=fr, COF_eq=COF_eq,
        rel_residual=rel_R, n_iter=n_it,
        converged=(rel_R <= 0.10),
        status=status, bt_count=bt_count,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage B': full-coverage herringbone groove anchor")
    parser.add_argument("--stageA", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    stageA_dir = resolve_stage_dir(args.stageA)
    manifest_A_path = os.path.join(stageA_dir,
                                   "working_geometry_manifest.json")
    with open(manifest_A_path, encoding="utf-8") as f:
        manifest_A = json.load(f)
    if manifest_A.get("schema_version") != SCHEMA:
        print(f"FAIL: schema mismatch (expected {SCHEMA})")
        sys.exit(1)

    loadcases = manifest_A["loadcases"]

    anchor_csv = os.path.join(stageA_dir, "anchor_cases.csv")
    anchors_by_lc: Dict[str, Dict[str, Any]] = {}
    with open(anchor_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors_by_lc[row["loadcase"]] = {
                k: _try_float(v) for k, v in row.items()}

    N_phi, N_Z = GRID_CONFIRM
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    groove_params = gu_groove_params_nondim(
        D, L, c, R, w_g, L_g, d_g, beta_deg, N_g)

    print(f"Stage B': full-coverage herringbone groove anchor")
    print(f"Grid: {N_phi}x{N_Z}, MAX_ITER={MAX_ITER_NR}, "
          f"STEP_CAP={STEP_CAP}, TOL_HARD={TOL_HARD}")
    print(f"HS warmup: iter={200000}, tol={1e-5}")

    groove_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for lc_name in LOADCASE_NAMES:
        lc = loadcases[lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        Wa_norm = float(np.linalg.norm(W_applied))

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wa_norm:.1f}N")
        print(f"{'='*60}")

        t0 = time.time()
        d = solve_groove_equilibrium(
            W_applied, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
            groove_params)
        dt = time.time() - t0
        d["loadcase"] = lc_name
        d["config"] = "groove_nomag"
        d["elapsed_sec"] = dt
        groove_rows.append(d)

        print(f"  eps={d['eps']:.4f} h_min={d['h_min']*1e6:.1f}um "
              f"p_max={d['p_max']/1e6:.2f}MPa "
              f"COF={d['COF_eq']:.6f} res={d['rel_residual']:.1e} "
              f"bt={d['bt_count']} [{d['status']}] {dt:.1f}s")

    total_time = time.time() - t_global

    results_by_lc: Dict[str, Dict[str, Any]] = {}
    for lc_name in LOADCASE_NAMES:
        anchor = anchors_by_lc.get(lc_name, {})
        groove = next((r for r in groove_rows if r["loadcase"] == lc_name),
                      None)
        results_by_lc[lc_name] = dict(
            conv_nomag=dict(anchor),
            groove_nomag=dict(groove) if groove else None,
        )

    manifest = dict(
        schema_version=SCHEMA,
        stage="Bp_full_groove_anchor",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        stageA_manifest=manifest_A_path,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        groove_params=groove_params,
        tol_accept=TOL_HARD,
        max_iter_NR=MAX_ITER_NR,
        step_cap=STEP_CAP,
        hs_warmup_iter=200000,
        hs_warmup_tol=1e-5,
        results_by_loadcase=results_by_lc,
        total_time_sec=total_time,
    )
    with open(os.path.join(args.out, "full_groove_anchor_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = os.path.join(args.out, "groove_anchor_results.csv")
    fields = sorted({k for r in groove_rows for k in r.keys()
                     if not isinstance(r[k], (dict, list, np.ndarray))})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in groove_rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})

    print(f"\nStage B' complete: {len(groove_rows)} solves")
    print(f"Total: {total_time:.1f}s")
    print(f"Artifacts: {args.out}/")


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


if __name__ == "__main__":
    main()
