#!/usr/bin/env python3
"""Stage D -- full 2x2 ablation: (conv/groove) x (nomag/mag).

For each loadcase and B_ref, assemble all 4 configs:
  conv_nomag   — from Stage A anchor (not re-solved)
  groove_nomag — from Stage B best (re-solve if not cached)
  conv_mag     — from Stage C (sector magnet, conventional H)
  groove_mag   — groove + sector magnet (solved here)

Compute headline (groove_mag vs conv_nomag) and contribution breakdown.
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
from models.groove_sector_magnet import make_sector_magnet_model
from models.magnetic_equilibrium import (
    find_equilibrium, result_to_dict, result_status,
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

CONFIGS = ["conv_nomag", "groove_nomag", "conv_mag", "groove_mag"]


# ── Zero-force magnetic model ────────────────────────────────────────

class _ZeroForceMagModel:
    @property
    def scale(self) -> float:
        return 0.0

    @scale.setter
    def scale(self, value: float):
        pass

    def force(self, X: float, Y: float):
        return 0.0, 0.0


# ── Grid ─────────────────────────────────────────────────────────────

def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


# ── H_and_force closures ────────────────────────────────────────────

def _make_H_and_force_conv(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z):
    """Closure for conventional (smooth) bearing."""
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
                              groove_params, active_cells):
    """Closure for partial-groove bearing."""
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage D: full 2x2 ablation")
    parser.add_argument("--stageA", type=str, required=True,
                        help="path to Stage A output directory")
    parser.add_argument("--stageB", type=str, required=True,
                        help="path to Stage B output directory")
    parser.add_argument("--stageC", type=str, required=True,
                        help="path to Stage C output directory")
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

    # ── Read anchor CSV (conv_nomag from Stage A) ────────────────
    anchor_csv = os.path.join(args.stageA, "anchor_cases.csv")
    anchors_by_lc: Dict[str, Dict[str, Any]] = {}
    with open(anchor_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors_by_lc[row["loadcase"]] = {
                k: (float(v) if _is_float(v) else v)
                for k, v in row.items()
            }

    # ── Validate Stage B manifest ────────────────────────────────
    manifest_B_path = os.path.join(args.stageB, "partial_manifest.json")
    with open(manifest_B_path, encoding="utf-8") as f:
        manifest_B = json.load(f)
    if manifest_B.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageB manifest "
              f"(expected {SCHEMA}, got {manifest_B.get('schema_version')})")
        sys.exit(1)

    best_path = os.path.join(args.stageB, "partial_best_by_loadcase.json")
    with open(best_path, encoding="utf-8") as f:
        best_by_lc = json.load(f)

    # ── Validate Stage C manifest ────────────────────────────────
    manifest_C_path = os.path.join(args.stageC,
                                   "sector_magnet_manifest.json")
    with open(manifest_C_path, encoding="utf-8") as f:
        manifest_C = json.load(f)
    if manifest_C.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageC manifest "
              f"(expected {SCHEMA}, got {manifest_C.get('schema_version')})")
        sys.exit(1)

    # ── Read Stage C results for conv_mag lookup ─────────────────
    stageC_csv = os.path.join(args.stageC, "sector_magnet_results.csv")
    stageC_rows: Dict[str, Dict[float, Dict[str, Any]]] = {}
    with open(stageC_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lc = row["loadcase"]
            b = float(row["B_ref_T"])
            stageC_rows.setdefault(lc, {})[b] = {
                k: (float(v) if _is_float(v) else v)
                for k, v in row.items()
            }

    # ── Grid setup ───────────────────────────────────────────────
    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    groove_params = gu_groove_params_nondim(
        D, L, c, R, w_g, L_g, d_g, beta_deg, N_g)
    cell_centers = get_herringbone_cell_centers(N_g)

    print(f"Stage D: full 2x2 ablation")
    print(f"Grid: {N_phi}x{N_Z}")
    print(f"B_ref sweep: {bref_list}")

    flat_rows: List[Dict[str, Any]] = []
    headline_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    # Cache for groove_nomag solves (keyed by loadcase)
    groove_nomag_cache: Dict[str, Dict[str, Any]] = {}

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

        for B_ref in bref_list:
            bkey = f"Bref_{B_ref:.2f}T"
            print(f"\n  B_ref = {B_ref:.2f} T")

            results_4: Dict[str, Dict[str, Any]] = {}

            # ── conv_nomag: from Stage A anchor (DO NOT re-solve) ────
            anchor = anchors_by_lc[lc_name]
            cn = dict(anchor)
            cn["config"] = "conv_nomag"
            cn["B_ref_T"] = 0.0
            cn["COF_eq"] = float(cn.get("COF",
                                 cn.get("COF_eq",
                                 float(cn["friction"]) / max(Wa_norm, 1e-20))))
            results_4["conv_nomag"] = cn

            # ── groove_nomag: from Stage B best (re-solve if needed) ─
            if lc_name in groove_nomag_cache:
                gn = dict(groove_nomag_cache[lc_name])
            else:
                H_and_force_g = _make_H_and_force_groove(
                    Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                    groove_params, active_cells)
                mag_zero = _ZeroForceMagModel()
                t0 = time.time()
                r_gn = find_equilibrium(
                    H_and_force_g, mag_zero, W_applied,
                    X0=0.0, Y0=-0.4,
                    tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
                    tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
                dt_gn = time.time() - t0
                gn = result_to_dict(r_gn)
                gn["COF_eq"] = float(r_gn.friction / max(Wa_norm, 1e-20))
                gn["elapsed_sec"] = float(dt_gn)
                groove_nomag_cache[lc_name] = dict(gn)
            gn["config"] = "groove_nomag"
            gn["B_ref_T"] = 0.0
            results_4["groove_nomag"] = gn

            # ── conv_mag: from Stage C ───────────────────────────────
            cm_lookup = stageC_rows.get(lc_name, {}).get(B_ref)
            if cm_lookup is not None:
                cm = dict(cm_lookup)
            else:
                # Solve conv_mag if not in Stage C
                H_and_force_c = _make_H_and_force_conv(
                    Phi, Zm, phi_1D, Z_1D, d_phi, d_Z)
                mag_sector = make_sector_magnet_model(
                    active_phi_centers=active_phi,
                    c_m=c, d_sub_m=D_SUB_M,
                    B_ref_T=B_ref,
                    t_cover_m=T_COVER_M,
                    A_mag_m2=A_MAG_M2,
                    n_mag=N_MAG_EXP,
                    g_reg_m=G_REG_M)
                t0 = time.time()
                r_cm = find_equilibrium(
                    H_and_force_c, mag_sector, W_applied,
                    X0=0.0, Y0=-0.4,
                    tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
                    tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
                dt_cm = time.time() - t0
                cm = result_to_dict(r_cm)
                cm["COF_eq"] = float(r_cm.friction / max(Wa_norm, 1e-20))
                cm["elapsed_sec"] = float(dt_cm)
            cm["config"] = "conv_mag"
            cm["B_ref_T"] = float(B_ref)
            results_4["conv_mag"] = cm

            # ── groove_mag: groove + sector magnet (solve here) ──────
            H_and_force_gm = _make_H_and_force_groove(
                Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, active_cells)
            mag_sector_gm = make_sector_magnet_model(
                active_phi_centers=active_phi,
                c_m=c, d_sub_m=D_SUB_M,
                B_ref_T=B_ref,
                t_cover_m=T_COVER_M,
                A_mag_m2=A_MAG_M2,
                n_mag=N_MAG_EXP,
                g_reg_m=G_REG_M)
            t0 = time.time()
            r_gm = find_equilibrium(
                H_and_force_gm, mag_sector_gm, W_applied,
                X0=0.0, Y0=-0.4,
                tol=TOL_HARD, step_cap=STEP_CAP, eps_max=EPS_MAX,
                tol_accept=TOL_HARD, max_iter=MAX_ITER_NR)
            dt_gm = time.time() - t0
            gm = result_to_dict(r_gm)
            gm["COF_eq"] = float(r_gm.friction / max(Wa_norm, 1e-20))
            gm["elapsed_sec"] = float(dt_gm)
            gm["config"] = "groove_mag"
            gm["B_ref_T"] = float(B_ref)
            results_4["groove_mag"] = gm

            # ── Print summary ────────────────────────────────────────
            for cfg in CONFIGS:
                d = results_4[cfg]
                status = d.get("status", "anchor" if cfg == "conv_nomag"
                               else "?")
                print(f"    {cfg:>16s}: eps={float(d.get('eps',0)):.4f} "
                      f"h_min={float(d.get('h_min',0))*1e6:.1f}um "
                      f"p_max={float(d.get('p_max',0))/1e6:.2f}MPa "
                      f"COF={float(d.get('COF_eq',0)):.6f} "
                      f"[{status}]")

            # ── Flatten rows ─────────────────────────────────────────
            for cfg in CONFIGS:
                row = dict(results_4[cfg])
                row["loadcase"] = lc_name
                row["B_ref_T"] = float(B_ref) if cfg != "conv_nomag" else 0.0
                flat_rows.append(row)

            # ── Ablation table row ───────────────────────────────────
            abl = dict(
                loadcase=lc_name,
                B_ref_T=B_ref,
            )
            for cfg in CONFIGS:
                d = results_4[cfg]
                abl[f"COF_{cfg}"] = float(d.get("COF_eq", 0))
                abl[f"h_min_{cfg}"] = float(d.get("h_min", 0))
                abl[f"p_max_{cfg}"] = float(d.get("p_max", 0))
                abl[f"eps_{cfg}"] = float(d.get("eps", 0))
            ablation_rows.append(abl)

            # ── Headline: groove_mag vs conv_nomag ───────────────────
            cn_d = results_4["conv_nomag"]
            gm_d = results_4["groove_mag"]
            cn_cof = float(cn_d.get("COF_eq", 0))
            gm_cof = float(gm_d.get("COF_eq", 0))
            cn_hmin = float(cn_d.get("h_min", 0))
            gm_hmin = float(gm_d.get("h_min", 0))
            cn_pmax = float(cn_d.get("p_max", 0))
            gm_pmax = float(gm_d.get("p_max", 0))

            dCOF_pct = ((gm_cof - cn_cof) / max(abs(cn_cof), 1e-20)
                        * 100.0)
            dh_pct = ((gm_hmin - cn_hmin) / max(abs(cn_hmin), 1e-20)
                      * 100.0)
            dp_pct = ((gm_pmax - cn_pmax) / max(abs(cn_pmax), 1e-20)
                      * 100.0)

            # Contribution breakdown
            gn_cof = float(results_4["groove_nomag"].get("COF_eq", 0))
            cm_cof = float(results_4["conv_mag"].get("COF_eq", 0))
            groove_contrib = cn_cof - gn_cof
            mag_contrib = cn_cof - cm_cof
            combined_effect = cn_cof - gm_cof
            synergy = combined_effect - groove_contrib - mag_contrib

            scenario = _classify_scenario(dCOF_pct, dh_pct)

            hl = dict(
                loadcase=lc_name,
                B_ref_T=B_ref,
                COF_conv_nomag=cn_cof,
                COF_groove_mag=gm_cof,
                dCOF_pct=dCOF_pct,
                dh_min_pct=dh_pct,
                dp_max_pct=dp_pct,
                groove_contrib=groove_contrib,
                mag_contrib=mag_contrib,
                combined_effect=combined_effect,
                synergy=synergy,
                scenario=scenario,
            )
            headline_rows.append(hl)

            print(f"    HEADLINE: dCOF={dCOF_pct:+.1f}%  "
                  f"dh_min={dh_pct:+.1f}%  dp_max={dp_pct:+.1f}%  "
                  f"scenario={scenario}")

    total_time = time.time() - t_global

    # ── Write combined_results.csv ───────────────────────────────
    _write_csv(os.path.join(args.out, "combined_results.csv"), flat_rows)

    # ── Write ablation_table.csv ─────────────────────────────────
    _write_csv(os.path.join(args.out, "ablation_table.csv"), ablation_rows)

    # ── Write headline_table.csv ─────────────────────────────────
    _write_csv(os.path.join(args.out, "headline_table.csv"), headline_rows)

    # ── Write combined_manifest.json ─────────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        stage="D_combined_ablation",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        stageA_manifest=manifest_A_path,
        stageB_manifest=manifest_B_path,
        stageC_manifest=manifest_C_path,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        bref_sweep=bref_list,
        configs=CONFIGS,
        tol_accept=TOL_HARD,
        max_iter_NR=MAX_ITER_NR,
        n_rows=len(flat_rows),
        scenarios={hl["loadcase"] + "_" + f"B{hl['B_ref_T']:.2f}":
                   hl["scenario"] for hl in headline_rows},
        total_time_sec=total_time,
    )
    with open(os.path.join(args.out, "combined_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStage D complete: {len(flat_rows)} rows")
    print(f"Total: {total_time:.1f}s")
    print(f"Artifacts: {args.out}/")


# ── Helpers ──────────────────────────────────────────────────────────

def _is_float(v):
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _classify_scenario(dCOF_pct: float, dh_pct: float) -> str:
    """Scenario tag from headline deltas."""
    if dCOF_pct < -5.0 and dh_pct > 5.0:
        return "win_win"
    if dCOF_pct < -5.0 and dh_pct >= -5.0:
        return "COF_gain_h_neutral"
    if dCOF_pct >= -5.0 and dh_pct > 5.0:
        return "h_gain_COF_neutral"
    if dCOF_pct > 5.0 or dh_pct < -5.0:
        return "adverse"
    return "marginal"


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
