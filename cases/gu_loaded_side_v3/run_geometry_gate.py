#!/usr/bin/env python3
"""Stage G2 — stationary geometry gate (no p_supply, no PV).

Hierarchical screening of feed-consistent geometry variants.
Pass 1: core set with default params.
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
import warnings
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_fn = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_fn = solve_payvar_salant_cpu

from cases.gu_loaded_side_v3.geometry_builders import build_relief
from cases.gu_loaded_side_v3.schema import (
    SCHEMA, resolve_stage_dir, classify_status, TOL_HARD,
)
from cases.gu_loaded_side_v3.common import (
    D, R, L, c, n_rpm, eta, sigma, w_g,
    GRID_MAIN, MAX_ITER_NR, STEP_CAP, EPS_MAX,
    HS_WARMUP_ITER, HS_WARMUP_TOL,
)

_warmup_warnings: List[Dict[str, Any]] = []


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def _ps_call(H, d_phi, d_Z, phi_bc, tag=""):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _ps_fn(H, d_phi, d_Z, R, L,
                         tol=1e-6, max_iter=10_000_000,
                         hs_warmup_iter=HS_WARMUP_ITER,
                         hs_warmup_tol=HS_WARMUP_TOL,
                         phi_bc=phi_bc)
    import re
    for w in caught:
        msg = str(w.message)
        if "warmup" in msg.lower():
            m = re.search(r"res=([0-9.e+-]+)", msg)
            _warmup_warnings.append(dict(
                config=tag,
                residual=float(m.group(1)) if m else None))
    return result


def _eval(X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
          relief, omega, p_scale, phi_bc, tag=""):
    H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
    H = H0 + relief
    P, theta, _, _ = _ps_call(H, d_phi, d_Z, phi_bc, tag)
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
    tau_couette = eta * omega * R / h_dim
    dP_dphi = np.gradient(P_dim, d_phi, axis=1)
    tau_pressure = h_dim / 2.0 * dP_dphi / R
    tau = tau_couette + tau_pressure
    friction = float(
        np.trapezoid(
            np.trapezoid(np.abs(tau), phi_1D, axis=1),
            Z_1D, axis=0) * R * L / 2.0)
    return float(Fx), float(Fy), h_min, p_max, cav_frac, friction


def solve_eq(W_applied, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
             relief, phi_bc, X0, Y0, tag=""):
    omega = 2 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    Wa_norm = float(np.linalg.norm(W_applied))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)

    Fx_h, Fy_h, h_min, p_max, cav, fr = _eval(
        X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        relief, omega, p_scale, phi_bc, tag)
    Rx = Fx_h - W_applied[0]
    Ry = Fy_h - W_applied[1]
    rel_R = math.sqrt(Rx ** 2 + Ry ** 2) / max(Wa_norm, 1e-20)

    for _ in range(MAX_ITER_NR):
        if rel_R < TOL_HARD:
            break
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            Fxp, Fyp, *_ = _eval(
                X + dX_, Y + dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                relief, omega, p_scale, phi_bc, tag)
            Fxn, Fyn, *_ = _eval(
                X - dX_, Y - dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                relief, omega, p_scale, phi_bc, tag)
            J[0, col] = (Fxp - Fxn) / (2.0 * dXY)
            J[1, col] = (Fyp - Fyn) / (2.0 * dXY)
        Rx = Fx_h - W_applied[0]
        Ry = Fy_h - W_applied[1]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det
        cap_f = STEP_CAP / max(abs(dX), abs(dY), 1e-20)
        if cap_f < 1.0:
            dX *= cap_f
            dY *= cap_f
        accepted = False
        for alpha in [1.0, 0.5, 0.25, 0.125]:
            Xt = X + alpha * dX
            Yt = Y + alpha * dY
            if math.sqrt(Xt ** 2 + Yt ** 2) >= EPS_MAX:
                continue
            Fxt, Fyt, hmt, pmt, cvt, frt = _eval(
                Xt, Yt, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                relief, omega, p_scale, phi_bc, tag)
            Rxt = Fxt - W_applied[0]
            Ryt = Fyt - W_applied[1]
            relt = math.sqrt(Rxt ** 2 + Ryt ** 2) / max(Wa_norm, 1e-20)
            if relt < rel_R:
                X, Y = Xt, Yt
                Fx_h, Fy_h = Fxt, Fyt
                h_min, p_max, cav, fr = hmt, pmt, cvt, frt
                rel_R = relt
                accepted = True
                break
        if not accepted:
            break

    eps = math.sqrt(X ** 2 + Y ** 2)
    COF = fr / max(Wa_norm, 1e-20)
    status = classify_status(rel_R, rel_R <= 0.10)
    return dict(eps=eps, h_min=h_min, p_max=p_max, cav_frac=cav,
                friction=fr, COF=COF, rel_residual=rel_R, status=status)


def main():
    parser = argparse.ArgumentParser(
        description="Stage G2: stationary geometry gate")
    parser.add_argument("--stageA", type=str, required=True)
    parser.add_argument("--variants", nargs="+",
                        default=["straight_ramped", "half_herringbone_ramped"])
    parser.add_argument("--dg-list", type=float, nargs="+", default=[10, 25])
    parser.add_argument("--belt-list", type=float, nargs="+", default=[0.15])
    parser.add_argument("--beta-list", type=float, nargs="+", default=[20])
    parser.add_argument("--ramp-frac", type=float, default=0.15)
    parser.add_argument("--taper-ratio", type=float, default=1.0)
    parser.add_argument("--apex-radius-frac", type=float, default=0.5)
    parser.add_argument("--coverage-modes", nargs="+",
                        default=["full_360", "partial_fixed"])
    parser.add_argument("--feed-modes", nargs="+",
                        default=["periodic", "groove"])
    parser.add_argument("--eps-list", type=float, nargs="+",
                        default=[0.5, 0.8])
    parser.add_argument("--grid", type=str, default="1200x400")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    stA = resolve_stage_dir(args.stageA)
    mp = os.path.join(stA, "working_geometry_manifest.json")
    with open(mp) as f:
        mA = json.load(f)
    anchors = {}
    for lc_name, lc in mA.get("loadcases", {}).items():
        anchors[float(lc["eps_source"])] = dict(
            W_applied=lc["applied_load_N"],
            phi_loaded_deg=lc.get("phi_loaded_deg", 140.0))

    print(f"Stage G2: geometry gate")
    print(f"Grid: {N_phi}x{N_Z}")
    print(f"Variants: {args.variants}")
    print(f"d_g: {args.dg_list} μm, belt: {args.belt_list}")
    print(f"Coverage: {args.coverage_modes}, Feed: {args.feed_modes}")

    _warmup_warnings.clear()
    rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for variant in args.variants:
        for dg_um in args.dg_list:
            for belt in args.belt_list:
                for beta in args.beta_list:
                    for cov_mode in args.coverage_modes:
                        # Pre-build relief (same for all eps/feed)
                        relief = build_relief(
                            Phi, Zm, variant=variant,
                            depth_nondim=dg_um * 1e-6 / c,
                            N_branch_per_side=3,
                            w_branch_nondim=w_g / R,
                            belt_half_nondim=belt,
                            beta_deg=beta,
                            ramp_frac=args.ramp_frac,
                            taper_ratio=args.taper_ratio,
                            apex_radius_frac=args.apex_radius_frac,
                            coverage_mode=cov_mode,
                            phi_loaded_deg=anchors.get(
                                0.5, {}).get("phi_loaded_deg", 140.0))
                        zero_relief = np.zeros_like(Phi)

                        for feed_mode in args.feed_modes:
                            phi_bc = feed_mode
                            for eps_ref in args.eps_list:
                                anc = anchors.get(eps_ref, {})
                                W_applied = np.array(
                                    anc.get("W_applied", [0, -100]),
                                    dtype=float)
                                Wa_norm = float(np.linalg.norm(W_applied))
                                Wa_dir = W_applied / max(Wa_norm, 1e-20)
                                X0 = -eps_ref * float(Wa_dir[0])
                                Y0 = -eps_ref * float(Wa_dir[1])

                                for use_vein, label in [(False, "smooth"),
                                                         (True, "veined")]:
                                    r = relief if use_vein else zero_relief
                                    feed_label = "fed" if phi_bc == "groove" else "nofeed"
                                    config = f"{label}_{feed_label}"
                                    tag = (f"{variant}_dg{dg_um}_b{belt}_"
                                           f"cov{cov_mode}_{config}_"
                                           f"eps{eps_ref}")

                                    t0 = time.time()
                                    d = solve_eq(
                                        W_applied, Phi, Zm, phi_1D, Z_1D,
                                        d_phi, d_Z, r, phi_bc, X0, Y0, tag)
                                    dt = time.time() - t0

                                    dg_hmin = (dg_um * 1e-6) / max(
                                        d["h_min"], 1e-12)
                                    row = dict(
                                        variant=variant, d_g_um=dg_um,
                                        belt=belt, beta_deg=beta,
                                        ramp_frac=args.ramp_frac,
                                        taper_ratio=args.taper_ratio,
                                        apex_radius_frac=args.apex_radius_frac,
                                        coverage_mode=cov_mode,
                                        feed_mode=feed_mode,
                                        config=config, eps_ref=eps_ref,
                                        eps_eq=d["eps"],
                                        h_min_um=d["h_min"] * 1e6,
                                        p_max_MPa=d["p_max"] / 1e6,
                                        COF=d["COF"],
                                        dg_over_hmin=dg_hmin,
                                        cav_frac=d["cav_frac"],
                                        status=d["status"],
                                        rel_residual=d["rel_residual"],
                                        elapsed_sec=dt)
                                    rows.append(row)

                                    st = "✓" if d["status"] in (
                                        "hard_converged",
                                        "soft_converged") else "✗"
                                    print(
                                        f"  [{st}] {tag[:60]:60s} "
                                        f"COF={d['COF']:.6f} "
                                        f"h={d['h_min']*1e6:.1f}μm "
                                        f"dg/h={dg_hmin:.2f} "
                                        f"res={d['rel_residual']:.1e} "
                                        f"[{d['status']}] {dt:.0f}s")

    total = time.time() - t_global

    # Pairwise: veined vs smooth
    pairs = []
    for r in rows:
        if "veined" not in r["config"]:
            continue
        sm_config = r["config"].replace("veined", "smooth")
        sm = next((s for s in rows
                    if s["config"] == sm_config
                    and s["variant"] == r["variant"]
                    and s["d_g_um"] == r["d_g_um"]
                    and s["belt"] == r["belt"]
                    and s["coverage_mode"] == r["coverage_mode"]
                    and s["feed_mode"] == r["feed_mode"]
                    and s["eps_ref"] == r["eps_ref"]), None)
        if sm:
            dCOF = (r["COF"] - sm["COF"]) / max(sm["COF"], 1e-20) * 100
            dh = (r["h_min_um"] - sm["h_min_um"]) / max(
                sm["h_min_um"], 1e-20) * 100
            dp = (r["p_max_MPa"] - sm["p_max_MPa"]) / max(
                sm["p_max_MPa"], 1e-20) * 100
            pairs.append(dict(
                variant=r["variant"], d_g_um=r["d_g_um"],
                belt=r["belt"], beta_deg=r["beta_deg"],
                coverage_mode=r["coverage_mode"],
                feed_mode=r["feed_mode"], eps_ref=r["eps_ref"],
                dCOF_pct=dCOF, dh_pct=dh, dp_pct=dp,
                status_veined=r["status"],
                status_smooth=sm["status"],
                COF_smooth=sm["COF"], COF_veined=r["COF"]))

    # CSVs
    if rows:
        csv_path = os.path.join(args.out, "geometry_gate_results.csv")
        fields = sorted(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    if pairs:
        pairs_csv = os.path.join(args.out, "geometry_gate_pairs.csv")
        pf = sorted(pairs[0].keys())
        with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pf)
            w.writeheader()
            for p in pairs:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in p.items()})

    # Summary
    print(f"\n{'='*60}")
    print(f"Pairwise: veined vs smooth (accepted only)")
    for p in pairs:
        if p["status_veined"] in ("hard_converged", "soft_converged"):
            print(f"  {p['variant']:30s} dg={p['d_g_um']:2.0f} "
                  f"cov={p['coverage_mode']:15s} "
                  f"feed={p['feed_mode']:8s} eps={p['eps_ref']}: "
                  f"ΔCOF={p['dCOF_pct']:+.1f}% "
                  f"Δh={p['dh_pct']:+.1f}% "
                  f"Δp={p['dp_pct']:+.1f}%")

    # Gate check
    weak_pass = any(
        p["dCOF_pct"] <= -3 and p["dh_pct"] >= -2
        and p["status_veined"] in ("hard_converged", "soft_converged")
        for p in pairs)
    strong_pass = any(
        p["dCOF_pct"] <= -5 and p["dh_pct"] >= 0
        and p["status_veined"] in ("hard_converged", "soft_converged")
        for p in pairs)

    verdict = "STRONG_PASS" if strong_pass else (
        "WEAK_PASS" if weak_pass else "NO_GO")
    print(f"\nGate verdict: {verdict}")

    ww_count = len(_warmup_warnings)
    ww_max = max((w["residual"] for w in _warmup_warnings
                   if w["residual"]), default=0.0)

    manifest = dict(
        schema_version=SCHEMA,
        stage="G2_geometry_gate",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        variants=args.variants,
        d_g_um_list=args.dg_list,
        belt_list=args.belt_list,
        beta_list=args.beta_list,
        ramp_frac=args.ramp_frac,
        taper_ratio=args.taper_ratio,
        apex_radius_frac=args.apex_radius_frac,
        coverage_modes=args.coverage_modes,
        feed_modes=args.feed_modes,
        eps_list=args.eps_list,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        gate_verdict=verdict,
        n_solves=len(rows),
        n_pairs=len(pairs),
        total_time_sec=total,
        hs_warmup_warnings=dict(count=ww_count, max_residual=ww_max),
    )
    with open(os.path.join(args.out, "geometry_gate_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total:.0f}s ({len(rows)} solves)")
    print(f"HS warmup warnings: {ww_count}")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
