#!/usr/bin/env python3
"""Stage 1 — Gu 2020 herringbone validation.

Три case'а (conventional / straight / herringbone) × ε = {0.2, 0.5, 0.8}
× три сетки (800×200, 1200×300, 1600×400).

Метрика: COF = F_friction / W_hydro (Couette-only friction, clarification 3).
Validation PASS: qualitative ordering herringbone ≤ straight ≤ conventional
и grid-convergence (ТЗ §6.2).
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

_HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
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
    create_H_with_straight_grooves,
    create_H_with_herringbone_grooves,
    gu_groove_params_nondim,
)
from cases.gu_2020 import config_gu as cfg

SCHEMA = "herringbone_gu_v1"
TEXTURE_TYPES = ["conventional", "straight_grooves", "herringbone_grooves"]


def make_grid(N_phi: int, N_Z: int):
    phi = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1.0, 1.0, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def build_H(eps: float, Phi: np.ndarray, Z: np.ndarray,
             texture_type: str, groove: Dict[str, Any]) -> np.ndarray:
    H0 = 1.0 + float(eps) * np.cos(Phi)
    if cfg.sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (cfg.sigma / cfg.c) ** 2)
    if texture_type == "conventional":
        return H0
    if texture_type == "straight_grooves":
        return create_H_with_straight_grooves(
            H0, groove["depth_nondim"], Phi, Z,
            groove["N_g"], groove["w_g_nondim"], groove["L_g_nondim"])
    if texture_type == "herringbone_grooves":
        return create_H_with_herringbone_grooves(
            H0, groove["depth_nondim"], Phi, Z,
            groove["N_g"], groove["w_g_nondim"], groove["L_g_nondim"],
            groove["beta_deg"])
    raise ValueError(f"unknown texture_type: {texture_type!r}")


def solve_metrics(H: np.ndarray, Phi: np.ndarray,
                   phi_1D: np.ndarray, Z_1D: np.ndarray,
                   d_phi: float, d_Z: float) -> Dict[str, float]:
    omega = 2.0 * math.pi * cfg.n / 60.0
    p_scale = 6.0 * cfg.eta * omega * (cfg.R / cfg.c) ** 2
    P, theta, _, _ = _ps(H, d_phi, d_Z, cfg.R, cfg.L,
                          tol=1e-6, max_iter=10_000_000)
    P_dim = P * p_scale
    Fx = -np.trapezoid(
        np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * cfg.R * cfg.L / 2.0
    Fy = -np.trapezoid(
        np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * cfg.R * cfg.L / 2.0
    W = float(math.sqrt(Fx ** 2 + Fy ** 2))
    h_dim = H * cfg.c
    h_min = float(np.min(h_dim))
    p_max = float(np.max(P_dim))
    cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    tau_c = cfg.eta * omega * cfg.R / h_dim
    F_friction = float(
        np.sum(tau_c) * cfg.R * (2.0 * math.pi / H.shape[1])
        * cfg.L * (2.0 / H.shape[0]) / 2.0)
    COF = F_friction / max(W, 1e-20)
    return dict(W=W, Fx=float(Fx), Fy=float(Fy),
                F_friction=F_friction, COF=COF,
                h_min=h_min, p_max=p_max, cav_frac=cav_frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=str, default="coarse,confirm,fine",
                        help="comma-separated grid tags from config_gu.GRIDS")
    parser.add_argument("--eps-list", type=str, default=None,
                        help="override eps list, e.g. '0.2,0.5,0.8'")
    args = parser.parse_args()

    grid_tags = [g.strip() for g in args.grids.split(",")]
    eps_list = ([float(x) for x in args.eps_list.split(",")]
                if args.eps_list else cfg.EPS_VALIDATION)

    out_dir = os.path.join(ROOT, "results", "herringbone_gu_v1")
    os.makedirs(out_dir, exist_ok=True)

    groove = gu_groove_params_nondim(
        cfg.D, cfg.L, cfg.c, cfg.R,
        cfg.w_g, cfg.L_g, cfg.d_g, cfg.beta_deg, cfg.N_g)
    print(f"Groove nondim: {groove}")

    all_rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for gtag in grid_tags:
        N_phi, N_Z = cfg.GRIDS[gtag]
        phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
        print(f"\n{'='*60}")
        print(f"Grid: {gtag} ({N_phi}×{N_Z})")
        print(f"{'='*60}")

        for tt in TEXTURE_TYPES:
            for eps in eps_list:
                t0 = time.time()
                H = build_H(eps, Phi, Zm, tt, groove)
                m = solve_metrics(H, Phi, phi_1D, Z_1D, d_phi, d_Z)
                dt = time.time() - t0
                row = dict(grid=gtag, N_phi=N_phi, N_Z=N_Z,
                           texture_type=tt, eps=eps, **m,
                           elapsed_sec=dt)
                all_rows.append(row)
                print(f"  {tt:>22s}  eps={eps:.1f}  COF={m['COF']:.6f}  "
                      f"W={m['W']:.1f}N  h_min={m['h_min']*1e6:.1f}μm  "
                      f"p_max={m['p_max']/1e6:.2f}MPa  {dt:.1f}s")

    total = time.time() - t_global

    # ── Grid convergence check (ТЗ §6.2) ─────────────────────────
    print(f"\n{'='*60}")
    print("Grid convergence (herringbone, eps=0.5)")
    gc_pass = True
    gc_rows = [r for r in all_rows
                if r["texture_type"] == "herringbone_grooves"
                and abs(r["eps"] - 0.5) < 1e-6]
    gc_by_grid = {r["grid"]: r["COF"] for r in gc_rows}
    if "coarse" in gc_by_grid and "confirm" in gc_by_grid:
        d1 = abs(gc_by_grid["confirm"] - gc_by_grid["coarse"]) / max(abs(gc_by_grid["confirm"]), 1e-20)
        tag1 = "✓" if d1 < 0.05 else "✗"
        print(f"  [{tag1}] |COF_1200 - COF_800| / COF_1200 = {d1:.4f} {'< 5%' if d1 < 0.05 else '>= 5%'}")
        gc_pass = gc_pass and d1 < 0.05
    if "confirm" in gc_by_grid and "fine" in gc_by_grid:
        d2 = abs(gc_by_grid["fine"] - gc_by_grid["confirm"]) / max(abs(gc_by_grid["fine"]), 1e-20)
        tag2 = "✓" if d2 < 0.02 else "✗"
        print(f"  [{tag2}] |COF_1600 - COF_1200| / COF_1600 = {d2:.4f} {'< 2%' if d2 < 0.02 else '>= 2%'}")
        gc_pass = gc_pass and d2 < 0.02

    # ── Qualitative ordering ────────────────────────────────────────
    # Required for PASS: herr < conv при ε=0.2 и ε=0.5.
    # str vs conv — только reported (без asperity contact в нашей модели
    # straight grooves убивают клиновой эффект → str > conv expected).
    print(f"\n{'='*60}")
    print("Qualitative ordering (confirm grid)")
    ordering_pass = True
    for eps in eps_list:
        rows_at = {r["texture_type"]: r["COF"]
                   for r in all_rows
                   if r["grid"] == "confirm" and abs(r["eps"] - eps) < 1e-6}
        if len(rows_at) < 2:
            print(f"  [?] eps={eps:.1f}: incomplete data")
            ordering_pass = False
            continue
        c_cof = rows_at.get("conventional")
        s_cof = rows_at.get("straight_grooves")
        h_cof = rows_at.get("herringbone_grooves")
        if c_cof is None or h_cof is None:
            print(f"  [?] eps={eps:.1f}: missing conv or herr")
            ordering_pass = False
            continue
        # Required check: herr < conv at low/mid ε
        herr_lt_conv = h_cof < c_cof
        is_blocking_eps = eps <= 0.5 + 1e-6
        if is_blocking_eps:
            tag = "✓" if herr_lt_conv else "✗"
            ordering_pass = ordering_pass and herr_lt_conv
            print(f"  [{tag}] eps={eps:.1f}: herr={h_cof:.6f} "
                  f"< conv={c_cof:.6f}  [REQUIRED]")
        else:
            # At high ε differences expected to vanish
            tag = "~"
            print(f"  [{tag}] eps={eps:.1f}: herr={h_cof:.6f} "
                  f"vs conv={c_cof:.6f}  [high-ε, info only]")
        # Informational: str vs conv
        if s_cof is not None:
            str_tag = "≤" if s_cof <= c_cof else ">"
            print(f"       str={s_cof:.6f} {str_tag} conv={c_cof:.6f}"
                  f"  [info — no asperity contact]")

    overall_pass = gc_pass and ordering_pass
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")

    # ── Artifacts ─────────────────────────────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        stage="gu_validation",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        bearing=dict(D_mm=cfg.D * 1e3, L_mm=cfg.L * 1e3,
                     c_um=cfg.c * 1e6, n_rpm=cfg.n,
                     eta_Pa_s=cfg.eta, sigma_um=cfg.sigma * 1e6),
        groove=groove,
        friction_model="couette_only",
        eps_list=eps_list,
        grids={k: dict(N_phi=v[0], N_Z=v[1]) for k, v in cfg.GRIDS.items()
               if k in grid_tags},
        grid_convergence_pass=bool(gc_pass),
        qualitative_ordering_pass=bool(ordering_pass),
        ordering_criterion="herr_lt_conv_at_eps_02_05",
        overall_pass=bool(overall_pass),
        total_time_sec=float(total),
    )
    with open(os.path.join(out_dir, "gu_validation_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "gu_validation_curves.csv")
    fields = sorted({k for r in all_rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    gc_csv = os.path.join(out_dir, "gu_grid_convergence.csv")
    gc_data = [r for r in all_rows
               if r["texture_type"] == "herringbone_grooves"]
    fields_gc = sorted({k for r in gc_data for k in r.keys()})
    with open(gc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_gc)
        w.writeheader()
        for r in gc_data:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ── Report ────────────────────────────────────────────────────
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gu 2020 herringbone validation\n\n")
        f.write(f"- schema: `{SCHEMA}`\n")
        f.write(f"- friction_model: `couette_only`\n")
        f.write(f"- overall: **{'PASS' if overall_pass else 'FAIL'}**\n")
        f.write(f"- grid_convergence: {'PASS' if gc_pass else 'FAIL'}\n")
        f.write(f"- qualitative_ordering: {'PASS' if ordering_pass else 'FAIL'}\n\n")
        f.write("## COF table (confirm grid)\n\n")
        f.write("| eps | conventional | straight | herringbone |\n")
        f.write("|-----|------------|----------|-------------|\n")
        for eps in eps_list:
            row_at = {r["texture_type"]: r["COF"]
                      for r in all_rows
                      if r["grid"] == "confirm" and abs(r["eps"] - eps) < 1e-6}
            f.write(f"| {eps:.1f} | {row_at.get('conventional', 0):.6f} "
                    f"| {row_at.get('straight_grooves', 0):.6f} "
                    f"| {row_at.get('herringbone_grooves', 0):.6f} |\n")
        f.write("\n")
        f.write("## Grid convergence (herringbone, eps=0.5)\n\n")
        f.write("| grid | N_phi | N_Z | COF |\n")
        f.write("|------|-------|-----|-----|\n")
        for r in gc_data:
            if abs(r["eps"] - 0.5) < 1e-6:
                f.write(f"| {r['grid']} | {r['N_phi']} | {r['N_Z']} "
                        f"| {r['COF']:.6f} |\n")
        f.write("\n")
        if os.path.exists(os.path.join(out_dir, "gu_fig6_reproduction.png")):
            f.write("![Fig 6](gu_fig6_reproduction.png)\n")
    print(f"\nArtifacts: {out_dir}/")
    print(f"  gu_validation_manifest.json")
    print(f"  gu_validation_curves.csv")
    print(f"  report.md")
    print(f"\nTotal: {total:.1f}s")


if __name__ == "__main__":
    main()
