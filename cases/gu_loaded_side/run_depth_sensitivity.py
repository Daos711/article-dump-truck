#!/usr/bin/env python3
"""Stage F — depth-sensitivity gate.

Sweep: 2 L/D × 3 d_g × 3 ε = 18 points.
Full-coverage herringbone, no magnets.
Tests hypothesis: high-ε harm is driven by d_g/h_min ratio.

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
from typing import Any, Dict, List, Optional, Tuple

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
    D, R, c, n_rpm, eta, sigma,
    w_g, L_g, beta_deg, N_g,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
    ps_solve,
)

# ── Geometry configs per L/D ─────────────────────────────────────

LD_CONFIGS = {
    0.30: dict(D=D, R=R, L=0.016, c=c, n_rpm=n_rpm, eta=eta, sigma=sigma),
    0.40: dict(D=D, R=R, L=D * 0.40, c=c, n_rpm=n_rpm, eta=eta, sigma=sigma),
}


def make_grid(N_phi, N_Z):
    phi = np.linspace(0.0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def _eval(X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
          groove_params, omega, p_scale, R_val, L_val):
    H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
    H = create_H_with_herringbone_grooves(
        H0, groove_params["depth_nondim"], Phi, Zm,
        groove_params["N_g"], groove_params["w_g_nondim"],
        groove_params["L_g_nondim"], groove_params["beta_deg"])
    P, theta, _, _ = ps_solve(_ps, H, d_phi, d_Z, R_val, L_val)
    P_dim = P * p_scale
    Fx = -np.trapezoid(
        np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R_val * L_val / 2.0
    Fy = -np.trapezoid(
        np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R_val * L_val / 2.0
    h_dim = H * c
    h_min = float(np.min(h_dim))
    p_max = float(np.max(P_dim))
    cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    tau_c = eta * omega * R_val / h_dim
    friction = float(
        np.sum(tau_c) * R_val * (2.0 * math.pi / H.shape[1])
        * L_val * (2.0 / H.shape[0]) / 2.0)
    return float(Fx), float(Fy), h_min, p_max, cav_frac, friction


def solve_groove_eq(W_applied, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                     groove_params, R_val, L_val, X0, Y0):
    omega = 2.0 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R_val / c) ** 2
    Wa_norm = float(np.linalg.norm(W_applied))
    dXY = 1e-4

    X, Y = float(X0), float(Y0)
    Fx_h, Fy_h, h_min, p_max, cav, fr = _eval(
        X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        groove_params, omega, p_scale, R_val, L_val)
    Rx = Fx_h - W_applied[0]
    Ry = Fy_h - W_applied[1]
    rel_R = math.sqrt(Rx ** 2 + Ry ** 2) / max(Wa_norm, 1e-20)

    converged = False
    n_it = 0
    bt_count = 0

    for _ in range(MAX_ITER_NR):
        if rel_R < TOL_HARD:
            converged = True
            break

        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            Fxp, Fyp, *_ = _eval(
                X + dX_, Y + dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, omega, p_scale, R_val, L_val)
            Fxn, Fyn, *_ = _eval(
                X - dX_, Y - dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, omega, p_scale, R_val, L_val)
            J[0, col] = (Fxp - Fxn) / (2.0 * dXY)
            J[1, col] = (Fyp - Fyn) / (2.0 * dXY)

        Rx = Fx_h - W_applied[0]
        Ry = Fy_h - W_applied[1]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det

        cap = STEP_CAP / max(abs(dX), abs(dY), 1e-20)
        if cap < 1.0:
            dX *= cap
            dY *= cap

        accepted = False
        for alpha in [1.0, 0.5, 0.25, 0.125]:
            X_try = X + alpha * dX
            Y_try = Y + alpha * dY
            eps_try = math.sqrt(X_try ** 2 + Y_try ** 2)
            if eps_try >= EPS_MAX:
                continue
            Fx_t, Fy_t, hm_t, pm_t, cv_t, fr_t = _eval(
                X_try, Y_try, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                groove_params, omega, p_scale, R_val, L_val)
            Rx_t = Fx_t - W_applied[0]
            Ry_t = Fy_t - W_applied[1]
            rel_t = math.sqrt(Rx_t ** 2 + Ry_t ** 2) / max(Wa_norm, 1e-20)
            if rel_t < rel_R:
                if alpha < 1.0:
                    bt_count += 1
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
    COF = fr / max(Wa_norm, 1e-20)
    status = classify_status(rel_R, rel_R <= 0.10)
    return dict(
        X=X, Y=Y, eps=eps,
        h_min=h_min, p_max=p_max, cav_frac=cav,
        friction=fr, COF=COF,
        rel_residual=rel_R, n_iter=n_it,
        status=status, bt_count=bt_count,
    )


def load_conv_anchors_ld030(source_path):
    """Load conventional anchors for L/D=0.30 from Gu validation CSV."""
    anchors = {}
    csv_path = os.path.join(source_path, "gu_validation_curves.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if (row.get("texture_type") == "conventional"
                        and row.get("grid") == "confirm"):
                    eps = float(row["eps"])
                    anchors[eps] = dict(
                        W=float(row["W"]), COF=float(row["COF"]),
                        h_min=float(row["h_min"]),
                        p_max=float(row["p_max"]),
                        Fx=float(row.get("Fx", 0)),
                        Fy=float(row.get("Fy", 0)),
                    )
        return anchors
    # Fallback: try ld_sweep
    csv2 = os.path.join(source_path, "ld_sweep_curves.csv")
    if os.path.exists(csv2):
        with open(csv2) as f:
            for row in csv.DictReader(f):
                if (row.get("config") == "conventional"
                        and row.get("grid_name") == "confirm"
                        and abs(float(row.get("ratio_target", 0)) - 0.30) < 0.01):
                    eps = float(row["eps"])
                    anchors[eps] = dict(
                        W=float(row["W_N"]), COF=float(row["COF"]),
                        h_min=float(row.get("h_min_um", 0)) * 1e-6,
                        p_max=float(row.get("p_max_MPa", 0)) * 1e6,
                        Fx=float(row.get("Fx", 0)),
                        Fy=float(row.get("Fy", 0)),
                    )
    return anchors


def load_conv_anchors_ld040(stageA_dir):
    """Load conventional anchors for L/D=0.40 from Stage A."""
    anchors = {}
    manifest_path = os.path.join(stageA_dir,
                                  "working_geometry_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    csv_path = os.path.join(stageA_dir, "anchor_cases.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                eps = float(row.get("eps", 0))
                anchors[eps] = dict(
                    W=float(row.get("W_N", 0)),
                    COF=float(row.get("COF", 0)),
                    h_min=float(row.get("h_min", 0)),
                    p_max=float(row.get("p_max", 0)),
                    Fx=float(row.get("Fx_hydro", 0)),
                    Fy=float(row.get("Fy_hydro", 0)),
                )
    # Also get applied loads from manifest
    for lc_name, lc in manifest.get("loadcases", {}).items():
        eps = float(lc.get("eps_source", 0))
        if eps in anchors:
            anchors[eps]["applied_load_N"] = lc["applied_load_N"]
    return anchors


def main():
    parser = argparse.ArgumentParser(
        description="Stage F: depth-sensitivity gate")
    parser.add_argument("--stageA-LD040", type=str, required=True)
    parser.add_argument("--stageA-LD030", type=str, required=True)
    parser.add_argument("--ld030-from-ld-sweep", type=str, default=None)
    parser.add_argument("--d-g-list", type=float, nargs="+",
                        default=[10, 25, 50])
    parser.add_argument("--eps-list", type=float, nargs="+",
                        default=[0.2, 0.5, 0.8])
    parser.add_argument("--grid", type=str, default="1200x400")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    os.makedirs(args.out, exist_ok=True)

    stageA_040 = resolve_stage_dir(args.stageA_LD040)
    stageA_030 = resolve_stage_dir(args.stageA_LD030)

    # Load conv anchors
    anchors_030 = load_conv_anchors_ld030(stageA_030)
    if not anchors_030 and args.ld030_from_ld_sweep:
        anchors_030 = load_conv_anchors_ld030(args.ld030_from_ld_sweep)
    anchors_040 = load_conv_anchors_ld040(stageA_040)

    all_anchors = {0.30: anchors_030, 0.40: anchors_040}

    print(f"Stage F: depth-sensitivity gate")
    print(f"Grid: {N_phi}x{N_Z}")
    print(f"d_g sweep: {args.d_g_list} μm")
    print(f"ε sweep: {args.eps_list}")
    print(f"L/D: [0.30, 0.40]")

    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    grid = (phi_1D, Z_1D, Phi, Zm, d_phi, d_Z)

    rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for ld_ratio in [0.30, 0.40]:
        geo = LD_CONFIGS[ld_ratio]
        R_val = geo["R"]
        L_val = geo["L"]
        anchors = all_anchors[ld_ratio]

        print(f"\n{'='*60}")
        print(f"L/D = {ld_ratio:.2f}  (L={L_val*1e3:.1f} mm)")
        print(f"{'='*60}")

        for d_g_um in args.d_g_list:
            d_g_m = d_g_um * 1e-6
            groove_params = gu_groove_params_nondim(
                D, L_val, c, R_val, w_g, L_g, d_g_m, beta_deg, N_g)

            for eps_ref in args.eps_list:
                anchor = anchors.get(eps_ref, {})
                W_anchor = anchor.get("W", 100.0)
                COF_anchor = anchor.get("COF", 0.01)
                h_min_anchor = anchor.get("h_min", 20e-6)

                # Build W_applied from anchor forces or estimate
                if "applied_load_N" in anchor:
                    W_applied = np.array(anchor["applied_load_N"],
                                          dtype=float)
                elif "Fx" in anchor and "Fy" in anchor:
                    W_applied = np.array([-anchor["Fx"], -anchor["Fy"]],
                                          dtype=float)
                else:
                    # Estimate: load in -Y direction
                    W_applied = np.array([0.0, -W_anchor], dtype=float)

                Wa_norm = float(np.linalg.norm(W_applied))

                # Warm-start from anchor eps
                Wa_dir = W_applied / max(Wa_norm, 1e-20)
                X0 = -eps_ref * float(Wa_dir[0])
                Y0 = -eps_ref * float(Wa_dir[1])

                t0 = time.time()
                d = solve_groove_eq(
                    W_applied, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                    groove_params, R_val, L_val, X0, Y0)
                dt = time.time() - t0

                dCOF_pct = ((d["COF"] - COF_anchor) / max(COF_anchor, 1e-20)
                            * 100) if COF_anchor > 0 else 0.0
                dh_pct = ((d["h_min"] - h_min_anchor)
                          / max(h_min_anchor, 1e-20)
                          * 100) if h_min_anchor > 0 else 0.0
                dg_hmin = (d_g_m / d["h_min"]) if d["h_min"] > 0 else float("inf")

                row = dict(
                    LD_ratio=ld_ratio,
                    d_g_um=d_g_um,
                    eps_ref=eps_ref,
                    eps_eq=d["eps"],
                    h_min_um=d["h_min"] * 1e6,
                    p_max_MPa=d["p_max"] / 1e6,
                    COF=d["COF"],
                    dCOF_pct=dCOF_pct,
                    dh_pct=dh_pct,
                    dg_over_hmin=dg_hmin,
                    status=d["status"],
                    rel_residual=d["rel_residual"],
                    bt_count=d["bt_count"],
                    elapsed_sec=dt,
                )
                rows.append(row)

                tag = "✓" if d["status"] in ("hard_converged", "soft_converged") else "✗"
                print(f"  [{tag}] d_g={d_g_um:2.0f}μm ε={eps_ref} → "
                      f"ε_eq={d['eps']:.4f} h_min={d['h_min']*1e6:.1f}μm "
                      f"ΔCOF={dCOF_pct:+.1f}% "
                      f"d_g/h_min={dg_hmin:.2f} "
                      f"res={d['rel_residual']:.1e} bt={d['bt_count']} "
                      f"[{d['status']}] {dt:.0f}s")

    total = time.time() - t_global

    # CSV
    csv_path = os.path.join(args.out, "depth_sensitivity_results.csv")
    fields = sorted(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # Manifest
    manifest = dict(
        schema_version=SCHEMA,
        stage="F_depth_sensitivity",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        LD_ratios=[0.30, 0.40],
        d_g_um_list=args.d_g_list,
        eps_list=args.eps_list,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        n_points=len(rows),
        total_time_sec=total,
    )
    with open(os.path.join(args.out, "depth_sensitivity_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary MD
    md_path = os.path.join(args.out, "stageF_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage F — depth-sensitivity gate\n\n")
        f.write(f"- {len(rows)} points computed in {total:.0f}s\n")
        f.write(f"- d_g sweep: {args.d_g_list} μm\n")
        f.write(f"- ε sweep: {args.eps_list}\n\n")

        for ld in [0.30, 0.40]:
            f.write(f"## L/D = {ld:.2f}\n\n")
            f.write("### ΔCOF vs conventional (%)\n\n")
            f.write("| ε \\ d_g | " + " | ".join(
                f"{d:.0f} μm" for d in args.d_g_list) + " |\n")
            f.write("|---" * (len(args.d_g_list) + 1) + "|\n")
            for eps in args.eps_list:
                vals = []
                for dg in args.d_g_list:
                    match = [r for r in rows
                             if abs(r["LD_ratio"] - ld) < 0.01
                             and abs(r["d_g_um"] - dg) < 0.1
                             and abs(r["eps_ref"] - eps) < 1e-6]
                    if match:
                        r = match[0]
                        s = "†" if r["status"] == "failed" else ""
                        vals.append(f"{r['dCOF_pct']:+.1f}%{s}")
                    else:
                        vals.append("—")
                f.write(f"| {eps} | " + " | ".join(vals) + " |\n")
            f.write("\n")

            f.write("### d_g / h_min ratio\n\n")
            f.write("| ε \\ d_g | " + " | ".join(
                f"{d:.0f} μm" for d in args.d_g_list) + " |\n")
            f.write("|---" * (len(args.d_g_list) + 1) + "|\n")
            for eps in args.eps_list:
                vals = []
                for dg in args.d_g_list:
                    match = [r for r in rows
                             if abs(r["LD_ratio"] - ld) < 0.01
                             and abs(r["d_g_um"] - dg) < 0.1
                             and abs(r["eps_ref"] - eps) < 1e-6]
                    if match:
                        vals.append(f"{match[0]['dg_over_hmin']:.2f}")
                    else:
                        vals.append("—")
                f.write(f"| {eps} | " + " | ".join(vals) + " |\n")
            f.write("\n")

        # Diagnostic gate
        f.write("## Diagnostic gate\n\n")
        for ld, gate_name in [(0.30, "GATE A"), (0.40, "GATE B")]:
            f.write(f"### {gate_name} (L/D={ld:.2f})\n\n")
            for dg in [50, 10]:
                match = [r for r in rows
                         if abs(r["LD_ratio"] - ld) < 0.01
                         and abs(r["d_g_um"] - dg) < 0.1
                         and abs(r["eps_ref"] - 0.8) < 1e-6]
                if match:
                    r = match[0]
                    s = " (FAILED)" if r["status"] == "failed" else ""
                    f.write(f"- d_g={dg} μm, ε=0.8: "
                            f"ΔCOF = {r['dCOF_pct']:+.1f}%{s}\n")
                else:
                    f.write(f"- d_g={dg} μm, ε=0.8: — (not computed)\n")
            f.write("\n")

        # Verdict
        gate_a_50 = next((r for r in rows
                          if abs(r["LD_ratio"] - 0.30) < 0.01
                          and abs(r["d_g_um"] - 50) < 0.1
                          and abs(r["eps_ref"] - 0.8) < 1e-6), None)
        gate_a_10 = next((r for r in rows
                          if abs(r["LD_ratio"] - 0.30) < 0.01
                          and abs(r["d_g_um"] - 10) < 0.1
                          and abs(r["eps_ref"] - 0.8) < 1e-6), None)
        gate_b_10 = next((r for r in rows
                          if abs(r["LD_ratio"] - 0.40) < 0.01
                          and abs(r["d_g_um"] - 10) < 0.1
                          and abs(r["eps_ref"] - 0.8) < 1e-6), None)

        a_works = (gate_a_10 is not None
                   and gate_a_10["status"] != "failed"
                   and gate_a_10["dCOF_pct"] < 0)
        b_works = (gate_b_10 is not None
                   and gate_b_10["status"] != "failed"
                   and gate_b_10["dCOF_pct"] < 0)

        f.write("## VERDICT\n\n")
        if a_works and b_works:
            f.write("**WORKS on both L/D** → shallow herringbone "
                    "is viable, no need for feed-consistent topology.\n")
        elif a_works and not b_works:
            f.write("**WORKS only at L/D=0.30** → L/D is critical; "
                    "for working geometry L/D=0.40, need different "
                    "topology or partial coverage.\n")
        else:
            f.write("**FAILS on both L/D** → depth alone does not "
                    "rescue high-ε regime. Feed-consistent topology "
                    "(smooth belt + branches) needed.\n")

    print(f"\nArtifacts: {args.out}/")
    print(f"Total: {total:.0f}s ({len(rows)} points)")


if __name__ == "__main__":
    main()
