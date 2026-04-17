#!/usr/bin/env python3
"""Stage M2 — 2×2 factorial ablation: (conv/groove) × (nomag/mag).

For each load case (L20, L50, L80) × B_ref sweep × 4 configurations:
  conv_nomag, groove_nomag, conv_mag, groove_mag

Uses find_equilibrium from magnetic_equilibrium (generic NR driver).
Magnet = EmbeddedGrooveMagnetModel (groove_magnet_force.py).
Groove = herringbone (texture_geometry.py).
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
    create_H_with_herringbone_grooves,
    gu_groove_params_nondim,
)
from models.groove_magnet_force import (
    EmbeddedGrooveMagnetModel,
    make_groove_magnet_model,
)
from models.magnetic_equilibrium import (
    find_equilibrium, is_accepted, result_to_dict, result_status,
)
from cases.gu_2020_magnet.config_gu_magnet import (
    SCHEMA, D, R, L, c, n, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    A_MAG_M2, N_MAG, G_REG_M, T_COVER_M,
    BREF_SWEEP_T, LOAD_CASE_NAMES,
    TOL_ACCEPT, STEP_CAP, EPS_MAX, MAX_ITER_NR, GRID,
)

CONFIGS = ["conv_nomag", "groove_nomag", "conv_mag", "groove_mag"]


def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def _make_H_and_force(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                       groove_params, use_groove: bool):
    """Closure: H(X,Y) + PS solve + metrics."""
    omega = 2.0 * math.pi * n / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    def H_and_force(X, Y):
        H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
        if sigma > 0:
            H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
        if use_groove:
            H = create_H_with_herringbone_grooves(
                H0, groove_params["depth_nondim"], Phi, Zm,
                groove_params["N_g"], groove_params["w_g_nondim"],
                groove_params["L_g_nondim"], groove_params["beta_deg"])
        else:
            H = H0
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


def solve_one_config(config_name: str,
                      W_applied: np.ndarray,
                      groove_params: Dict,
                      B_ref_T: float,
                      grid_tuple,
                      X0: float = 0.0, Y0: float = -0.4,
                      ) -> Dict[str, Any]:
    """Solve equilibrium for one of the four configurations."""
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = grid_tuple
    use_groove = "groove" in config_name
    use_mag = "mag" in config_name and "nomag" not in config_name

    H_and_force = _make_H_and_force(
        Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        groove_params, use_groove)

    if use_mag:
        mag = make_groove_magnet_model(
            N_g=N_g, c_m=c, d_g_m=d_g, t_cover_m=T_COVER_M,
            B_ref_T=B_ref_T, A_mag_m2=A_MAG_M2,
            n_mag=N_MAG, g_reg_m=G_REG_M)
    else:
        mag = make_groove_magnet_model(N_g=N_g, c_m=c, d_g_m=d_g,
                                        B_ref_T=0.0)

    t0 = time.time()
    r = find_equilibrium(
        H_and_force, mag, W_applied,
        X0=X0, Y0=Y0,
        tol=TOL_ACCEPT, step_cap=STEP_CAP, eps_max=EPS_MAX,
        tol_accept=TOL_ACCEPT, max_iter=MAX_ITER_NR)
    dt = time.time() - t0

    Wa_norm = float(np.linalg.norm(W_applied))
    e_load = W_applied / max(Wa_norm, 1e-20)
    COF_eq = r.friction / max(Wa_norm, 1e-20)

    d = result_to_dict(r)
    d.update(dict(
        config=config_name,
        B_ref_T=float(B_ref_T),
        use_groove=use_groove,
        use_mag=use_mag,
        COF_eq=float(COF_eq),
        elapsed_sec=float(dt),
    ))
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--grid", type=str, default=None)
    args = parser.parse_args()

    N_phi, N_Z = args.grid and tuple(int(x) for x in args.grid.split("x")) or GRID
    run_id = args.run_id or datetime.datetime.now().strftime(
        "%Y-%m-%d_%H%M%S_ablation")

    results_base = os.path.join(ROOT, "cases", "gu_2020_magnet", "results")
    out_dir = os.path.join(results_base, run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    # Load cases
    lc_path = os.path.join(results_base, "loadcases_gu_aligned.json")
    if not os.path.exists(lc_path):
        print(f"FAIL: запусти build_loadcases_from_fixed_eps.py сначала "
              f"(нет {lc_path})")
        sys.exit(1)
    with open(lc_path) as f:
        lc_doc = json.load(f)

    groove_params = gu_groove_params_nondim(
        D, L, c, R, w_g, L_g, d_g, beta_deg, N_g)
    grid_tuple = make_grid(N_phi, N_Z)

    print(f"Run ID: {run_id}")
    print(f"Grid: {N_phi}×{N_Z}")
    print(f"B_ref sweep: {BREF_SWEEP_T}")
    print(f"Groove: w_g={groove_params['w_g_nondim']:.4f} rad, "
          f"depth={groove_params['depth_nondim']:.4f}")

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    flat_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for lc_name in LOAD_CASE_NAMES:
        lc = lc_doc["loadcases"][lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        Wa_norm = float(np.linalg.norm(W_applied))
        print(f"\n{'='*60}")
        print(f"Load case: {lc_name} (W={Wa_norm:.1f} N, "
              f"eps_source={lc['eps_source']})")
        print(f"{'='*60}")

        all_results[lc_name] = {"applied_load_N": lc["applied_load_N"]}

        for B_ref in BREF_SWEEP_T:
            bkey = f"Bref_{B_ref:.2f}T"
            print(f"\n  B_ref = {B_ref:.2f} T")
            case_results = {}

            X_prev, Y_prev = 0.0, -0.4
            for cfg in CONFIGS:
                d = solve_one_config(
                    cfg, W_applied, groove_params, B_ref,
                    grid_tuple, X0=X_prev, Y0=Y_prev)
                case_results[cfg] = d
                if is_accepted(type("R", (), d)(), TOL_ACCEPT) if False else \
                   d.get("converged") and d.get("rel_residual", 1) < TOL_ACCEPT:
                    X_prev, Y_prev = d["X"], d["Y"]

                status = d.get("status", "?")
                print(f"    {cfg:>16s}: ε={d['eps']:.4f} "
                      f"h_min={d['h_min']*1e6:.1f}μm "
                      f"p_max={d['p_max']/1e6:.2f}MPa "
                      f"COF={d['COF_eq']:.6f} "
                      f"res={d['rel_residual']:.1e} "
                      f"[{status}] {d['elapsed_sec']:.1f}s")

                row = dict(d, loadcase=lc_name, B_ref_T=B_ref)
                flat_rows.append(row)

            all_results[lc_name][bkey] = case_results

    total_time = time.time() - t_global

    # ── Headline comparisons ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("HEADLINE: groove_mag vs conv_nomag")
    print(f"{'='*60}")
    for lc_name in LOAD_CASE_NAMES:
        for B_ref in BREF_SWEEP_T:
            bkey = f"Bref_{B_ref:.2f}T"
            cr = all_results[lc_name].get(bkey)
            if not cr:
                continue
            cn = cr.get("conv_nomag", {})
            gm = cr.get("groove_mag", {})
            if not cn or not gm:
                continue
            dh = (gm["h_min"] - cn["h_min"]) / max(cn["h_min"], 1e-20) * 100
            dp = (gm["p_max"] - cn["p_max"]) / max(cn["p_max"], 1e-20) * 100
            de = gm["eps"] - cn["eps"]
            dc = (gm["COF_eq"] - cn["COF_eq"]) / max(cn["COF_eq"], 1e-20) * 100
            print(f"  {lc_name} B={B_ref:.2f}T: "
                  f"Δε={de:+.4f}  Δh_min={dh:+.1f}%  "
                  f"Δp_max={dp:+.1f}%  ΔCOF={dc:+.1f}%")

    # ── Artifacts ─────────────────────────────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        run_id=run_id,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        gu_geometry=dict(D_mm=D * 1e3, L_mm=L * 1e3,
                         c_um=c * 1e6, n_rpm=n, eta_Pa_s=eta),
        groove_params=groove_params,
        magnet_model=dict(
            A_mag_mm2=A_MAG_M2 * 1e6,
            n_mag=N_MAG, g_reg_um=G_REG_M * 1e6,
            t_cover_um=T_COVER_M * 1e6),
        Bref_sweep_T=BREF_SWEEP_T,
        solver_tol_accept=TOL_ACCEPT,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        loadcases_file="loadcases_gu_aligned.json",
        friction_model="couette_only",
        total_time_sec=total_time,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "ablation_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(
            schema_version=SCHEMA,
            loadcases=all_results,
        ), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "ablation_table.csv")
    fields = sorted({k for r in flat_rows for k in r.keys()
                     if not isinstance(r.get(k), (dict, list, np.ndarray))})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})

    print(f"\nArtifacts: {out_dir}/")
    print(f"Total: {total_time:.1f}s "
          f"({len(flat_rows)} equilibrium solves)")


if __name__ == "__main__":
    main()
