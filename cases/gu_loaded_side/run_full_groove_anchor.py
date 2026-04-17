#!/usr/bin/env python3
"""Stage B' -- full-coverage herringbone groove anchor (no magnets).

For each loadcase from Stage A, solve equilibrium with full-coverage
herringbone grooves (all N_g cells active) and zero magnetic force.
Produces groove_nomag anchors that complement the conv_nomag anchors
from Stage A.

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
    create_H_with_herringbone_grooves,
    gu_groove_params_nondim,
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
    GRID_CONFIRM,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
)


# -- Zero-force magnetic model (satisfies find_equilibrium interface) --

class _ZeroForceMagModel:
    """Trivial magnet model that produces zero force everywhere."""

    @property
    def scale(self) -> float:
        return 0.0

    @scale.setter
    def scale(self, value: float):
        pass

    def force(self, X: float, Y: float):
        return 0.0, 0.0


# -- Grid -----------------------------------------------------------------

def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


# -- H_and_force closure (full-coverage herringbone) ----------------------

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


# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage B': full-coverage herringbone groove anchor")
    parser.add_argument("--stageA", type=str, required=True,
                        help="path to Stage A output directory")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()

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

    # -- Read anchor CSV (conv_nomag from Stage A) ------------------------
    anchor_csv = os.path.join(stageA_dir, "anchor_cases.csv")
    anchors_by_lc: Dict[str, Dict[str, Any]] = {}
    with open(anchor_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors_by_lc[row["loadcase"]] = {
                k: (_try_float(v)) for k, v in row.items()
            }

    # -- Grid setup -------------------------------------------------------
    N_phi, N_Z = GRID_CONFIRM
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    groove_params = gu_groove_params_nondim(
        D, L, c, R, w_g, L_g, d_g, beta_deg, N_g)

    mag_model = _ZeroForceMagModel()

    print(f"Stage B': full-coverage herringbone groove anchor")
    print(f"Grid: {N_phi}x{N_Z}")

    groove_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    H_and_force = _make_H_and_force_groove(
        Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, groove_params)

    for lc_name in LOADCASE_NAMES:
        lc = loadcases[lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        Wa_norm = float(np.linalg.norm(W_applied))

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wa_norm:.1f}N")
        print(f"{'='*60}")

        t0 = time.time()
        r = find_equilibrium(
            H_and_force, mag_model, W_applied,
            X0=0.0, Y0=-0.4,
            tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
            tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
        dt = time.time() - t0

        COF_eq = r.friction / max(Wa_norm, 1e-20)
        status = classify_status(r.rel_residual, r.converged)

        d = result_to_dict(r)
        d.update(dict(
            loadcase=lc_name,
            config="groove_nomag",
            COF_eq=float(COF_eq),
            elapsed_sec=float(dt),
            classify_status=status,
        ))
        groove_rows.append(d)

        print(f"  eps={r.eps:.4f} h_min={r.h_min*1e6:.1f}um "
              f"p_max={r.p_max/1e6:.2f}MPa "
              f"COF={COF_eq:.6f} res={r.rel_residual:.1e} "
              f"[{status}] {dt:.1f}s")

    total_time = time.time() - t_global

    # -- Build manifest combining conv_nomag + groove_nomag ---------------
    results_by_lc: Dict[str, Dict[str, Any]] = {}
    for lc_name in LOADCASE_NAMES:
        anchor = anchors_by_lc.get(lc_name, {})
        groove = next((r for r in groove_rows if r["loadcase"] == lc_name),
                      None)
        results_by_lc[lc_name] = dict(
            conv_nomag=dict(anchor),
            groove_nomag=dict(groove) if groove else None,
        )

    # -- Write full_groove_anchor_manifest.json ---------------------------
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
        results_by_loadcase=results_by_lc,
        n_solves=len(groove_rows),
        total_time_sec=total_time,
    )
    with open(os.path.join(args.out, "full_groove_anchor_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStage B' complete: {len(groove_rows)} solves")
    print(f"Total: {total_time:.1f}s")
    print(f"Artifacts: {args.out}/")


# -- Helpers --------------------------------------------------------------

def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


if __name__ == "__main__":
    main()
