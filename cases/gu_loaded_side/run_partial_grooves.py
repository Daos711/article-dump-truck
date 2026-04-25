#!/usr/bin/env python3
"""Stage B -- partial groove coverage.

For each loadcase from Stage A, sweep N_active x shift_cells combos,
solve herringbone_grooves_subset equilibrium (NR with zero-force mag
model), and select best pattern per loadcase (min COF_eq among feasible).
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
    create_H_with_herringbone_grooves_subset,
    get_herringbone_cell_centers,
    gu_groove_params_nondim,
)
from models.magnetic_equilibrium import (
    find_equilibrium, is_accepted, result_to_dict, result_status,
)
from cases.gu_loaded_side.schema import SCHEMA, classify_status
from cases.gu_loaded_side.common import (
    D, R, L, c, n_rpm, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    EPS_REF, LOADCASE_NAMES,
    N_ACTIVE_LIST, SHIFT_CELLS_LIST,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
)
from cases.gu_loaded_side.schema import TOL_HARD


# ── Zero-force magnetic model (satisfies find_equilibrium interface) ──

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


# ── Grid ──────────────────────────────────────────────────────────────

def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


# ── H_and_force closure ──────────────────────────────────────────────

def _make_H_and_force(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                       groove_params, active_cells):
    """Build closure that solves PS on partial-groove geometry."""
    omega = 2.0 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    def H_and_force(X, Y):
        H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
        if sigma > 0:
            H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
        H = create_H_with_herringbone_grooves_subset(
            H0, groove_params["depth_nondim"], Phi, Zm,
            groove_params["N_g"], groove_params["w_g_nondim"],
            groove_params["L_g_nondim"], groove_params["beta_deg"],
            active_cells)
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


# ── Active cell computation ──────────────────────────────────────────

def compute_active_cells(phi_loaded_deg: float, N_active: int,
                          shift_cells: int, N_g: int) -> List[int]:
    """Contiguous window of N_active cells centered near phi_loaded,
    shifted by shift_cells positions."""
    cell_centers = get_herringbone_cell_centers(N_g)
    phi_loaded_rad = math.radians(phi_loaded_deg)
    # Find nearest cell to loaded angle
    dists = np.abs(np.mod(cell_centers - phi_loaded_rad + math.pi,
                          2 * math.pi) - math.pi)
    center_idx = int(np.argmin(dists))
    # Build contiguous window (with wrap-around)
    half = N_active // 2
    start = center_idx - half + shift_cells
    indices = [(start + i) % N_g for i in range(N_active)]
    return indices


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage B: partial groove coverage")
    parser.add_argument("--stageA", type=str, required=True,
                        help="path to Stage A output directory")
    parser.add_argument("--grid", type=str, default="1200x400",
                        help="grid NxM (default 1200x400)")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()

    # ── Validate Stage A manifest ─────────────────────────────────
    from cases.gu_loaded_side.schema import resolve_stage_dir
    stageA_dir = resolve_stage_dir(args.stageA)
    manifest_A_path = os.path.join(stageA_dir, "working_geometry_manifest.json")
    with open(manifest_A_path, encoding="utf-8") as f:
        manifest_A = json.load(f)
    if manifest_A.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageA manifest "
              f"(expected {SCHEMA}, got {manifest_A.get('schema_version')})")
        sys.exit(1)

    loadcases = manifest_A["loadcases"]

    # ── Read anchor CSV for feasibility thresholds ────────────────
    anchor_csv = os.path.join(stageA_dir, "anchor_cases.csv")
    anchors_by_lc: Dict[str, Dict[str, Any]] = {}
    with open(anchor_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors_by_lc[row["loadcase"]] = row

    # ── Grid setup ────────────────────────────────────────────────
    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)

    os.makedirs(args.out, exist_ok=True)

    groove_params = gu_groove_params_nondim(
        D, L, c, R, w_g, L_g, d_g, beta_deg, N_g)

    mag_model = _ZeroForceMagModel()

    print(f"Stage B: partial groove coverage")
    print(f"Grid: {N_phi}x{N_Z}")
    print(f"N_active sweep: {N_ACTIVE_LIST}")
    print(f"shift_cells sweep: {SHIFT_CELLS_LIST}")

    flat_rows: List[Dict[str, Any]] = []
    best_by_lc: Dict[str, Dict[str, Any]] = {}
    t_global = time.time()

    for lc_name in LOADCASE_NAMES:
        lc = loadcases[lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        phi_loaded_deg = float(lc["phi_loaded_deg"])

        anchor = anchors_by_lc[lc_name]
        h_min_conv = float(anchor["h_min"])
        p_max_conv = float(anchor["p_max"])

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  phi_loaded={phi_loaded_deg:.1f} deg")
        print(f"  anchor h_min={h_min_conv*1e6:.1f} um  "
              f"p_max={p_max_conv/1e6:.2f} MPa")
        print(f"{'='*60}")

        best_cof = float("inf")
        best_row = None

        for N_active in N_ACTIVE_LIST:
            for shift_cells in SHIFT_CELLS_LIST:
                active_cells = compute_active_cells(
                    phi_loaded_deg, N_active, shift_cells, N_g)

                H_and_force = _make_H_and_force(
                    Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                    groove_params, active_cells)

                t0 = time.time()
                r = find_equilibrium(
                    H_and_force, mag_model, W_applied,
                    X0=0.0, Y0=-0.4,
                    tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
                    tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
                dt = time.time() - t0

                Wa_norm = float(np.linalg.norm(W_applied))
                COF_eq = r.friction / max(Wa_norm, 1e-20)

                d = result_to_dict(r)
                d.update(dict(
                    loadcase=lc_name,
                    config="groove_nomag",
                    N_active=N_active,
                    shift_cells=shift_cells,
                    active_cells=active_cells,
                    COF_eq=float(COF_eq),
                    elapsed_sec=float(dt),
                    phi_loaded_deg=phi_loaded_deg,
                ))

                # Feasibility check
                feasible = (
                    r.converged
                    and r.h_min >= 0.95 * h_min_conv
                    and r.p_max <= 1.50 * p_max_conv
                )
                d["feasible"] = feasible

                flat_rows.append(d)

                tag = "F" if feasible else "x"
                print(f"  [{tag}] N={N_active} shift={shift_cells:+d}: "
                      f"eps={r.eps:.4f} h_min={r.h_min*1e6:.1f}um "
                      f"p_max={r.p_max/1e6:.2f}MPa "
                      f"COF={COF_eq:.6f} res={r.rel_residual:.1e} "
                      f"{dt:.1f}s")

                if feasible and COF_eq < best_cof:
                    best_cof = COF_eq
                    best_row = dict(d)

        if best_row is not None:
            best_by_lc[lc_name] = best_row
            print(f"  BEST: N={best_row['N_active']} "
                  f"shift={best_row['shift_cells']:+d} "
                  f"COF={best_row['COF_eq']:.6f}")
        else:
            print(f"  BEST: no feasible pattern found")

    total_time = time.time() - t_global

    # ── Write partial_results.csv ─────────────────────────────────
    csv_path = os.path.join(args.out, "partial_results.csv")
    # Exclude non-scalar fields
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

    # ── Write partial_best_by_loadcase.json ───────────────────────
    # Serialize active_cells as list of int
    best_serializable = {}
    for lc_name, row in best_by_lc.items():
        row_s = {k: v for k, v in row.items()}
        row_s["active_cells"] = [int(i) for i in row_s.get("active_cells", [])]
        best_serializable[lc_name] = row_s
    with open(os.path.join(args.out, "partial_best_by_loadcase.json"),
              "w", encoding="utf-8") as f:
        json.dump(best_serializable, f, indent=2, ensure_ascii=False)

    # ── Write partial_manifest.json ───────────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        stage="B_partial_grooves",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        stageA_manifest=manifest_A_path,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        N_active_sweep=N_ACTIVE_LIST,
        shift_cells_sweep=SHIFT_CELLS_LIST,
        groove_params=groove_params,
        tol_accept=TOL_HARD,
        max_iter_NR=MAX_ITER_NR,
        n_combos=len(flat_rows),
        n_feasible=sum(1 for r in flat_rows if r.get("feasible")),
        best_by_loadcase={lc: dict(N_active=b["N_active"],
                                    shift_cells=b["shift_cells"],
                                    active_cells=[int(i) for i in b["active_cells"]],
                                    COF_eq=b["COF_eq"])
                          for lc, b in best_by_lc.items()},
        total_time_sec=total_time,
    )
    with open(os.path.join(args.out, "partial_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStage B complete: {len(flat_rows)} combos, "
          f"{sum(1 for r in flat_rows if r.get('feasible'))} feasible")
    print(f"Total: {total_time:.1f}s")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
