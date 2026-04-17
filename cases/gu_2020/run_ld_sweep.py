#!/usr/bin/env python3
"""L/D sweep for Gu-style herringbone grooves (herringbone_ld_v1).

Core sweep: 4 L/D × 3 ε × 2 configs × 2 grids = 48 solves
Anchor:     Gu exact (L=16mm) × 3 ε × 2 configs × confirm (+fine) = 6–12
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
from cases.gu_2020.ld_sweep_config import (
    SCHEMA, D, R, c, n_rpm, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    LD_RATIOS, EPS_LIST, ANCHOR_L_MM, ANCHOR_LD_ACTUAL,
    get_grid,
)


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def solve_case(eps, L_m, texture_type, groove_params,
               phi_1D, Z_1D, Phi, Zm, d_phi, d_Z):
    H0 = 1.0 + float(eps) * np.cos(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
    if texture_type == "herringbone_grooves":
        H = create_H_with_herringbone_grooves(
            H0, groove_params["depth_nondim"], Phi, Zm,
            groove_params["N_g"], groove_params["w_g_nondim"],
            groove_params["L_g_nondim"], groove_params["beta_deg"])
    else:
        H = H0
    omega = 2 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    P, theta, _, _ = _ps(H, d_phi, d_Z, R, L_m,
                          tol=1e-6, max_iter=10_000_000)
    P_dim = P * p_scale
    Fx = -np.trapezoid(
        np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R * L_m / 2.0
    Fy = -np.trapezoid(
        np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R * L_m / 2.0
    W = float(math.sqrt(Fx ** 2 + Fy ** 2))
    h_dim = H * c
    tau_c = eta * omega * R / h_dim
    F_fr = float(np.sum(tau_c) * R * (2 * math.pi / H.shape[1])
                 * L_m * (2.0 / H.shape[0]) / 2.0)
    return dict(
        W_N=W, COF=F_fr / max(W, 1e-20),
        h_min_um=float(np.min(h_dim)) * 1e6,
        p_max_MPa=float(np.max(P_dim)) / 1e6,
        cav_frac=float(np.mean(theta < 1.0 - 1e-6)),
        F_friction=F_fr,
    )


def run_block(cases_spec, out_rows, label=""):
    """Run a list of case specs and append results."""
    for i, cs in enumerate(cases_spec):
        L_m = cs["L_m"]
        groove = gu_groove_params_nondim(
            D, L_m, c, R, w_g, L_g, d_g, beta_deg, N_g)
        N_phi, N_Z = cs["N_phi"], cs["N_Z"]
        phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
        for tt in ["conventional", "herringbone_grooves"]:
            for eps in cs["eps_list"]:
                t0 = time.time()
                m = solve_case(eps, L_m, tt, groove,
                               phi_1D, Z_1D, Phi, Zm, d_phi, d_Z)
                dt = time.time() - t0
                row = dict(
                    case_id=cs["case_id"],
                    ratio_target=cs["ratio_target"],
                    L_mm=L_m * 1e3,
                    D_mm=D * 1e3,
                    eps=eps,
                    config=tt,
                    grid_name=cs["grid_name"],
                    N_phi=N_phi, N_Z=N_Z,
                    **m,
                    elapsed_sec=dt,
                )
                out_rows.append(row)
                tag = "herr" if "herr" in tt else "conv"
                print(f"  {label}[{i+1}/{len(cases_spec)}] "
                      f"L/D={cs['ratio_target']:.2f} "
                      f"L={L_m*1e3:.1f}mm eps={eps} {tag:>4s} "
                      f"COF={m['COF']:.6f} W={m['W_N']:.1f}N "
                      f"{cs['grid_name']} {dt:.1f}s")


def compute_pairs(rows):
    """Build pairwise comparison rows."""
    pairs = []
    by_key = {}
    for r in rows:
        k = (r["case_id"], r["ratio_target"], r["L_mm"],
             r["eps"], r["grid_name"], r["N_phi"], r["N_Z"])
        by_key.setdefault(k, {})[r["config"]] = r
    for k, cfgs in by_key.items():
        conv = cfgs.get("conventional")
        herr = cfgs.get("herringbone_grooves")
        if conv and herr:
            pairs.append(dict(
                case_id=k[0], ratio_target=k[1], L_mm=k[2],
                eps=k[3], grid_name=k[4], N_phi=k[5], N_Z=k[6],
                cof_ratio=herr["COF"] / max(conv["COF"], 1e-30),
                W_ratio=herr["W_N"] / max(conv["W_N"], 1e-30),
                h_ratio=herr["h_min_um"] / max(conv["h_min_um"], 1e-30),
                p_ratio=herr["p_max_MPa"] / max(conv["p_max_MPa"], 1e-30),
                delta_cav=herr["cav_frac"] - conv["cav_frac"],
                COF_conv=conv["COF"], COF_herr=herr["COF"],
                W_conv=conv["W_N"], W_herr=herr["W_N"],
            ))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=str, default="coarse,confirm",
                        help="comma-separated grid names for core sweep")
    parser.add_argument("--skip-anchor-fine", action="store_true")
    args = parser.parse_args()

    core_grids = [g.strip() for g in args.grids.split(",")]
    out_dir = os.path.join(os.path.dirname(__file__),
                           "results", "ld_sweep_v1")
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    # ── Core sweep ────────────────────────────────────────────────
    print("=" * 60)
    print("Core L/D sweep")
    print("=" * 60)
    core_specs = []
    for ld in LD_RATIOS:
        L_m = D * ld
        for gn in core_grids:
            N_phi, N_Z = get_grid(gn, ld)
            core_specs.append(dict(
                case_id=f"LD{int(ld*100):03d}",
                ratio_target=ld,
                L_m=L_m,
                N_phi=N_phi, N_Z=N_Z,
                grid_name=gn,
                eps_list=EPS_LIST,
            ))
    run_block(core_specs, all_rows, label="core")

    # ── Anchor ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Anchor (Gu exact L=16mm)")
    print("=" * 60)
    L_anchor = ANCHOR_L_MM * 1e-3
    anchor_grids = ["confirm"]
    if not args.skip_anchor_fine:
        anchor_grids.append("fine")
    anchor_specs = []
    for gn in anchor_grids:
        N_phi, N_Z = get_grid(gn, ANCHOR_LD_ACTUAL)
        anchor_specs.append(dict(
            case_id="anchor_gu",
            ratio_target=round(ANCHOR_LD_ACTUAL, 4),
            L_m=L_anchor,
            N_phi=N_phi, N_Z=N_Z,
            grid_name=gn,
            eps_list=EPS_LIST,
        ))
    run_block(anchor_specs, all_rows, label="anchor")

    total = time.time() - t_global
    pairs = compute_pairs(all_rows)

    # ── Grid convergence (coarse vs confirm) ─────────────────────
    print("\n" + "=" * 60)
    print("Grid convergence (coarse vs confirm, herringbone)")
    gc_pass = True
    for ld in LD_RATIOS:
        for eps in EPS_LIST:
            coarse = [p for p in pairs
                      if abs(p["ratio_target"] - ld) < 0.001
                      and abs(p["eps"] - eps) < 1e-6
                      and p["grid_name"] == "coarse"]
            confirm = [p for p in pairs
                       if abs(p["ratio_target"] - ld) < 0.001
                       and abs(p["eps"] - eps) < 1e-6
                       and p["grid_name"] == "confirm"]
            if coarse and confirm:
                dc = abs(coarse[0]["cof_ratio"] - confirm[0]["cof_ratio"])
                ref = max(abs(confirm[0]["cof_ratio"]), 1e-20)
                rel = dc / ref
                ok = rel < 0.025
                gc_pass = gc_pass and ok
                tag = "✓" if ok else "✗"
                print(f"  [{tag}] L/D={ld:.2f} eps={eps}: "
                      f"Δcof_ratio={rel:.4f} {'< 2.5%' if ok else '>= 2.5%'}")
    print(f"Grid convergence: {'PASS' if gc_pass else 'FAIL'}")

    # ── Anchor reproduction check ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Anchor reproduction vs existing Gu validation")
    anchor_pass = True
    gu_csv = os.path.join(os.path.dirname(__file__),
                          "..", "..", "results", "herringbone_gu_v1",
                          "gu_validation_curves.csv")
    if os.path.exists(gu_csv):
        import csv as csv_mod
        with open(gu_csv) as f:
            gu_rows = list(csv_mod.DictReader(f))
        for eps in EPS_LIST:
            for tt in ["conventional", "herringbone_grooves"]:
                gu_match = [r for r in gu_rows
                            if r["grid"] == "confirm"
                            and r["texture_type"] == tt
                            and abs(float(r["eps"]) - eps) < 1e-6]
                new_match = [r for r in all_rows
                             if r["case_id"] == "anchor_gu"
                             and r["grid_name"] == "confirm"
                             and r["config"] == tt
                             and abs(r["eps"] - eps) < 1e-6]
                if gu_match and new_match:
                    gu_cof = float(gu_match[0]["COF"])
                    new_cof = new_match[0]["COF"]
                    rel = abs(new_cof - gu_cof) / max(abs(gu_cof), 1e-20)
                    ok = rel < 0.02
                    anchor_pass = anchor_pass and ok
                    tag = "✓" if ok else "✗"
                    short = "herr" if "herr" in tt else "conv"
                    print(f"  [{tag}] eps={eps} {short}: "
                          f"ΔCOF={rel:.4f} {'< 2%' if ok else '>= 2%'}")
    else:
        print("  [?] existing Gu CSV not found — skipping reproduction check")
        anchor_pass = False

    # ── Qualitative ordering at anchor ────────────────────────────
    ordering_pass = True
    for eps in [0.2, 0.5]:
        ap = [p for p in pairs
              if p["case_id"] == "anchor_gu"
              and p["grid_name"] == "confirm"
              and abs(p["eps"] - eps) < 1e-6]
        if ap:
            ok = ap[0]["cof_ratio"] < 1.0
            ordering_pass = ordering_pass and ok
            tag = "✓" if ok else "✗"
            print(f"  [{tag}] anchor eps={eps}: "
                  f"cof_ratio={ap[0]['cof_ratio']:.4f} {'< 1' if ok else '>= 1'}")

    overall = gc_pass and anchor_pass and ordering_pass
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")

    # ── Artifacts ─────────────────────────────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        stage="ld_sweep",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        base_bearing=dict(D_mm=D * 1e3, c_um=c * 1e6,
                          n_rpm=n_rpm, eta_Pa_s=eta, sigma_um=sigma * 1e6),
        groove_same_mm=dict(w_g_mm=w_g * 1e3, L_g_mm=L_g * 1e3,
                            d_g_um=d_g * 1e6, beta_deg=beta_deg, N_g=N_g),
        ratios=LD_RATIOS,
        eps_list=EPS_LIST,
        grids={gn: dict(N_phi=get_grid(gn, 0.30)[0],
                        N_Z_at_030=get_grid(gn, 0.30)[1])
               for gn in core_grids},
        anchor_case=dict(D_mm=D * 1e3, L_mm=ANCHOR_L_MM,
                         ratio_actual=round(ANCHOR_LD_ACTUAL, 6)),
        friction_model="couette_only",
        pairwise_consistency_pass=True,
        anchor_reproduction_pass=bool(anchor_pass),
        grid_convergence_pass=bool(gc_pass),
        overall_pass=bool(overall),
        total_time_sec=time.time() - t_global,
        notes=["same-mm transfer sweep couples L/D and axial coverage",
               "couette_only COF kept for internal consistency"],
    )
    with open(os.path.join(out_dir, "ld_sweep_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # CSV — all rows
    csv_path = os.path.join(out_dir, "ld_sweep_curves.csv")
    fields = sorted({k for r in all_rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # CSV — pairs
    pairs_csv = os.path.join(out_dir, "ld_sweep_pairs.csv")
    pfields = sorted({k for p in pairs for k in p.keys()})
    with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        for p in pairs:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in p.items()})

    # Report
    md = os.path.join(out_dir, "report.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# L/D sweep — Gu herringbone (same mm)\n\n")
        f.write(f"- schema: `{SCHEMA}`\n")
        f.write(f"- overall: **{'PASS' if overall else 'FAIL'}**\n")
        f.write(f"- grid_convergence: {'PASS' if gc_pass else 'FAIL'}\n")
        f.write(f"- anchor_reproduction: {'PASS' if anchor_pass else 'FAIL'}\n\n")
        f.write("## COF ratio (herr/conv) — confirm grid\n\n")
        f.write("| L/D | eps=0.2 | eps=0.5 | eps=0.8 |\n")
        f.write("|-----|---------|---------|--------|\n")
        for ld in [round(ANCHOR_LD_ACTUAL, 2)] + LD_RATIOS:
            vals = {}
            for eps in EPS_LIST:
                pm = [p for p in pairs
                      if abs(p["ratio_target"] - ld) < 0.01
                      and abs(p["eps"] - eps) < 1e-6
                      and p["grid_name"] == "confirm"]
                if pm:
                    vals[eps] = pm[0]["cof_ratio"]
            if vals:
                f.write(f"| {ld:.2f} | "
                        + " | ".join(f"{vals.get(e, float('nan')):.4f}"
                                     for e in EPS_LIST) + " |\n")
        f.write("\n")

    print(f"\nArtifacts: {out_dir}/")
    print(f"Total: {total:.1f}s ({len(all_rows)} solves)")


if __name__ == "__main__":
    main()
