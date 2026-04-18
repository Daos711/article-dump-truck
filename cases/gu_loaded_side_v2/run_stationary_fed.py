#!/usr/bin/env python3
"""Stage B_v2 — stationary fed geometry, no PV, no magnets.

Binary feed: phi_bc='periodic' (no feed) vs phi_bc='groove' (fed).
For each (feed_state, variant, d_g, belt, eps) solves equilibrium
and compares veined vs smooth.

Gate: stage-gates at d_g=10/belt=0.15/variant=straight first.
Expand only if first gate passes.
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
    _ps_fn = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_fn = solve_payvar_salant_cpu

from models.feed_geometry import (
    create_H_with_central_feed_branches,
    feed_geometry_params,
)
from cases.gu_loaded_side_v2.schema import (
    SCHEMA, resolve_stage_dir, classify_status, TOL_HARD,
)
from cases.gu_loaded_side_v2.common import (
    D, R, L, c, n_rpm, eta, sigma, w_g,
    EPS_REF, LOADCASE_NAMES, GRID_CONFIRM,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
    HS_WARMUP_ITER, HS_WARMUP_TOL,
)


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def _ps_call(H, d_phi, d_Z, phi_bc):
    return _ps_fn(H, d_phi, d_Z, R, L,
                   tol=1e-6, max_iter=10_000_000,
                   hs_warmup_iter=HS_WARMUP_ITER,
                   hs_warmup_tol=HS_WARMUP_TOL,
                   phi_bc=phi_bc)


def _eval(X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
          geo_params, omega, p_scale, phi_bc, use_branches):
    H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
    if use_branches and geo_params["depth_nondim"] > 0:
        H = create_H_with_central_feed_branches(
            H0, Phi=Phi, Z=Zm, **{
                k: geo_params[k]
                for k in ("depth_nondim", "N_branch_per_side",
                           "w_branch_nondim", "belt_half_nondim",
                           "beta_deg", "variant")})
    else:
        H = H0
    P, theta, _, _ = _ps_call(H, d_phi, d_Z, phi_bc)
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
        np.sum(tau_c) * R * (2 * math.pi / H.shape[1])
        * L * (2.0 / H.shape[0]) / 2.0)
    return float(Fx), float(Fy), h_min, p_max, cav_frac, friction


def solve_eq(W_applied, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
             geo_params, phi_bc, use_branches, X0, Y0):
    omega = 2 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    Wa_norm = float(np.linalg.norm(W_applied))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)

    Fx_h, Fy_h, h_min, p_max, cav, fr = _eval(
        X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        geo_params, omega, p_scale, phi_bc, use_branches)
    Rx = Fx_h - W_applied[0]
    Ry = Fy_h - W_applied[1]
    rel_R = math.sqrt(Rx ** 2 + Ry ** 2) / max(Wa_norm, 1e-20)
    n_it = 0

    for _ in range(MAX_ITER_NR):
        if rel_R < TOL_HARD:
            break
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            Fxp, Fyp, *_ = _eval(
                X + dX_, Y + dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                geo_params, omega, p_scale, phi_bc, use_branches)
            Fxn, Fyn, *_ = _eval(
                X - dX_, Y - dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                geo_params, omega, p_scale, phi_bc, use_branches)
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
            Xt = X + alpha * dX
            Yt = Y + alpha * dY
            if math.sqrt(Xt ** 2 + Yt ** 2) >= EPS_MAX:
                continue
            Fxt, Fyt, hmt, pmt, cvt, frt = _eval(
                Xt, Yt, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                geo_params, omega, p_scale, phi_bc, use_branches)
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
        n_it += 1

    eps = math.sqrt(X ** 2 + Y ** 2)
    COF = fr / max(Wa_norm, 1e-20)
    status = classify_status(rel_R, rel_R <= 0.10)
    return dict(X=X, Y=Y, eps=eps, h_min=h_min, p_max=p_max,
                cav_frac=cav, friction=fr, COF=COF,
                rel_residual=rel_R, n_iter=n_it, status=status)


def main():
    parser = argparse.ArgumentParser(
        description="Stage B_v2: stationary fed geometry")
    parser.add_argument("--stageA", type=str, default=None,
                        help="Stage A v1.1 manifest for anchor loadcases")
    parser.add_argument("--variant", choices=["straight", "half_herringbone"],
                        default="straight")
    parser.add_argument("--dg", type=float, nargs="+", default=[10])
    parser.add_argument("--belt", type=float, nargs="+", default=[0.15])
    parser.add_argument("--eps-list", type=float, nargs="+",
                        default=[0.2, 0.5, 0.8])
    parser.add_argument("--grid", type=str, default="1200x400")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    N_phi, N_Z = (int(x) for x in args.grid.split("x"))
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    os.makedirs(args.out, exist_ok=True)

    # Load anchors from Stage A if available (for warm-start + W_applied)
    anchors = {}
    if args.stageA:
        stA = resolve_stage_dir(args.stageA)
        mp = os.path.join(stA, "working_geometry_manifest.json")
        if os.path.exists(mp):
            with open(mp) as f:
                mA = json.load(f)
            for lc_name, lc in mA.get("loadcases", {}).items():
                anchors[float(lc["eps_source"])] = dict(
                    W_applied=lc["applied_load_N"],
                    eps_source=lc["eps_source"],
                    lc_name=lc_name,
                )

    print(f"Stage B_v2: stationary fed geometry")
    print(f"Grid: {N_phi}x{N_Z}, variant={args.variant}")
    print(f"d_g: {args.dg} μm, belt: {args.belt}")
    print(f"Feed: periodic (no feed) vs groove (fed)")

    rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for d_g_um in args.dg:
        for belt in args.belt:
            geo = feed_geometry_params(
                d_g_um * 1e-6, c, N_branch=3, w_g_m=w_g, R_m=R,
                belt_half_frac=belt, variant=args.variant)

            for phi_bc in ["periodic", "groove"]:
                feed_label = "fed" if phi_bc == "groove" else "no_feed"

                for use_branches in [False, True]:
                    config = ("veined" if use_branches else "smooth") + f"_{feed_label}"

                    for eps_ref in args.eps_list:
                        anchor = anchors.get(eps_ref)
                        if anchor:
                            W_applied = np.array(anchor["W_applied"],
                                                  dtype=float)
                        else:
                            # Fallback: estimate load for this eps
                            omega = 2 * math.pi * n_rpm / 60.0
                            p_scale = 6.0 * eta * omega * (R / c) ** 2
                            H0_test = 1.0 + eps_ref * np.cos(Phi)
                            P_t, _, _, _ = _ps_call(H0_test, d_phi, d_Z,
                                                     "periodic")
                            Pd = P_t * p_scale
                            Fx = -np.trapezoid(
                                np.trapezoid(Pd * np.cos(Phi), phi_1D, axis=1),
                                Z_1D, axis=0) * R * L / 2.0
                            Fy = -np.trapezoid(
                                np.trapezoid(Pd * np.sin(Phi), phi_1D, axis=1),
                                Z_1D, axis=0) * R * L / 2.0
                            W_applied = np.array([-float(Fx), -float(Fy)])

                        Wa_norm = float(np.linalg.norm(W_applied))
                        Wa_dir = W_applied / max(Wa_norm, 1e-20)
                        X0 = -eps_ref * float(Wa_dir[0])
                        Y0 = -eps_ref * float(Wa_dir[1])

                        t0 = time.time()
                        d = solve_eq(
                            W_applied, Phi, Zm, phi_1D, Z_1D,
                            d_phi, d_Z, geo, phi_bc, use_branches,
                            X0, Y0)
                        dt = time.time() - t0

                        row = dict(
                            d_g_um=d_g_um, belt=belt,
                            variant=args.variant,
                            phi_bc=phi_bc, feed=feed_label,
                            config=config,
                            eps_ref=eps_ref,
                            eps_eq=d["eps"],
                            h_min_um=d["h_min"] * 1e6,
                            p_max_MPa=d["p_max"] / 1e6,
                            COF=d["COF"],
                            cav_frac=d["cav_frac"],
                            friction=d["friction"],
                            status=d["status"],
                            rel_residual=d["rel_residual"],
                            elapsed_sec=dt,
                        )
                        rows.append(row)

                        tag = "✓" if d["status"] in (
                            "hard_converged", "soft_converged") else "✗"
                        print(f"  [{tag}] {config:>20s} dg={d_g_um:2.0f} "
                              f"belt={belt} eps={eps_ref} "
                              f"COF={d['COF']:.6f} "
                              f"h={d['h_min']*1e6:.1f}μm "
                              f"res={d['rel_residual']:.1e} "
                              f"[{d['status']}] {dt:.0f}s")

    total = time.time() - t_global

    # Pairwise: veined vs smooth at same (d_g, belt, feed, eps)
    pair_rows = []
    for r in rows:
        if "veined" not in r["config"]:
            continue
        smooth_key = r["config"].replace("veined", "smooth")
        smooth = next((s for s in rows
                       if s["config"] == smooth_key
                       and s["d_g_um"] == r["d_g_um"]
                       and s["belt"] == r["belt"]
                       and s["eps_ref"] == r["eps_ref"]), None)
        if smooth:
            dCOF = ((r["COF"] - smooth["COF"])
                    / max(smooth["COF"], 1e-20) * 100)
            dh = ((r["h_min_um"] - smooth["h_min_um"])
                  / max(smooth["h_min_um"], 1e-20) * 100)
            pair_rows.append(dict(
                d_g_um=r["d_g_um"], belt=r["belt"],
                feed=r["feed"], eps_ref=r["eps_ref"],
                COF_smooth=smooth["COF"], COF_veined=r["COF"],
                dCOF_pct=dCOF,
                h_smooth=smooth["h_min_um"], h_veined=r["h_min_um"],
                dh_pct=dh,
                status_smooth=smooth["status"],
                status_veined=r["status"],
            ))

    # CSV
    csv_path = os.path.join(args.out, "stationary_fed_results.csv")
    if rows:
        fields = sorted(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    pairs_csv = os.path.join(args.out, "stationary_fed_pairs.csv")
    if pair_rows:
        pf = sorted(pair_rows[0].keys())
        with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pf)
            w.writeheader()
            for r in pair_rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    # Summary
    print(f"\n{'='*60}")
    print(f"Pairwise: veined vs smooth")
    for p in pair_rows:
        print(f"  {p['feed']:>7s} dg={p['d_g_um']:2.0f} belt={p['belt']} "
              f"eps={p['eps_ref']}: "
              f"ΔCOF={p['dCOF_pct']:+.1f}% Δh={p['dh_pct']:+.1f}%")

    # Manifest
    manifest = dict(
        schema_version=SCHEMA,
        stage="B_v2_stationary_fed",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        feed_model="binary: periodic (no feed) vs groove (fed)",
        variant=args.variant,
        d_g_um_list=args.dg,
        belt_list=args.belt,
        eps_list=args.eps_list,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        n_solves=len(rows),
        n_pairs=len(pair_rows),
        total_time_sec=total,
    )
    with open(os.path.join(args.out, "stageB_v2_stationary_fed_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total:.0f}s ({len(rows)} solves)")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
