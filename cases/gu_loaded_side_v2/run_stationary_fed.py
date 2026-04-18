#!/usr/bin/env python3
"""Stage B_v2 — stationary fed geometry, no PV, no magnets.

Binary feed: phi_bc='periodic' (no feed) vs phi_bc='groove' (fed).
Fixes applied: beta passthrough, d_g/h_min metric, warmup warning
capture, Poiseuille friction component.
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

# Fix 3: warmup warning tracker
_warmup_warnings: List[Dict[str, Any]] = []


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def _ps_call(H, d_phi, d_Z, phi_bc, config_tag=""):
    """PS solve with warmup warning capture (Fix 3)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _ps_fn(H, d_phi, d_Z, R, L,
                         tol=1e-6, max_iter=10_000_000,
                         hs_warmup_iter=HS_WARMUP_ITER,
                         hs_warmup_tol=HS_WARMUP_TOL,
                         phi_bc=phi_bc)
    for w in caught:
        msg = str(w.message)
        if "warmup" in msg.lower():
            import re
            res_match = re.search(r"res=([0-9.e+-]+)", msg)
            _warmup_warnings.append(dict(
                config=config_tag,
                message=msg,
                residual=float(res_match.group(1)) if res_match else None,
            ))
    return result


def _eval(X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
          geo_params, omega, p_scale, phi_bc, use_branches,
          config_tag=""):
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
    P, theta, _, _ = _ps_call(H, d_phi, d_Z, phi_bc, config_tag)
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
    # Fix 4: full friction = Couette + Poiseuille
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
             geo_params, phi_bc, use_branches, X0, Y0,
             config_tag=""):
    omega = 2 * math.pi * n_rpm / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    Wa_norm = float(np.linalg.norm(W_applied))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)

    Fx_h, Fy_h, h_min, p_max, cav, fr = _eval(
        X, Y, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        geo_params, omega, p_scale, phi_bc, use_branches, config_tag)
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
                geo_params, omega, p_scale, phi_bc, use_branches, config_tag)
            Fxn, Fyn, *_ = _eval(
                X - dX_, Y - dY_, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                geo_params, omega, p_scale, phi_bc, use_branches, config_tag)
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
                geo_params, omega, p_scale, phi_bc, use_branches, config_tag)
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
    parser.add_argument("--stageA", type=str, default=None)
    parser.add_argument("--variant", choices=["straight", "half_herringbone"],
                        default="straight")
    # Fix 1: beta CLI arg
    parser.add_argument("--beta", type=float, default=20.0,
                        help="branch angle (deg), only for half_herringbone")
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

    # Fix 1: compute effective beta
    effective_beta = args.beta if args.variant == "half_herringbone" else 0.0

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
    print(f"Grid: {N_phi}x{N_Z}, variant={args.variant}"
          f"{f', beta={effective_beta}°' if args.variant == 'half_herringbone' else ''}")
    print(f"d_g: {args.dg} μm, belt: {args.belt}")
    print(f"Feed: periodic (no feed) vs groove (fed)")
    print(f"Friction: Couette + Poiseuille (full)")

    _warmup_warnings.clear()
    rows: List[Dict[str, Any]] = []
    t_global = time.time()

    for d_g_um in args.dg:
        for belt in args.belt:
            # Fix 1: pass beta_deg
            geo = feed_geometry_params(
                d_g_um * 1e-6, c, N_branch=3, w_g_m=w_g, R_m=R,
                belt_half_frac=belt,
                beta_deg=effective_beta,
                variant=args.variant)

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

                        config_tag = f"{config}_dg{d_g_um}_belt{belt}_eps{eps_ref}"
                        t0 = time.time()
                        d = solve_eq(
                            W_applied, Phi, Zm, phi_1D, Z_1D,
                            d_phi, d_Z, geo, phi_bc, use_branches,
                            X0, Y0, config_tag)
                        dt = time.time() - t0

                        # Fix 2: d_g/h_min metric
                        dg_over_hmin = (d_g_um * 1e-6) / max(d["h_min"], 1e-12)

                        row = dict(
                            d_g_um=d_g_um, belt=belt,
                            variant=args.variant,
                            beta_deg=effective_beta,
                            phi_bc=phi_bc, feed=feed_label,
                            config=config,
                            eps_ref=eps_ref,
                            eps_eq=d["eps"],
                            h_min_um=d["h_min"] * 1e6,
                            p_max_MPa=d["p_max"] / 1e6,
                            COF=d["COF"],
                            dg_over_hmin=dg_over_hmin,
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
                              f"dg/h={dg_over_hmin:.2f} "
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
                dg_over_hmin_veined=r["dg_over_hmin"],
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

    # Fix 3: warmup warnings summary in manifest
    ww_count = len(_warmup_warnings)
    ww_max_res = (max(w["residual"] for w in _warmup_warnings
                       if w["residual"] is not None)
                  if _warmup_warnings else 0.0)
    ww_configs = sorted({w["config"] for w in _warmup_warnings})

    # Manifest
    manifest = dict(
        schema_version=SCHEMA,
        stage="B_v2_stationary_fed",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        feed_model="binary: periodic (no feed) vs groove (fed)",
        friction_model="couette_plus_poiseuille",
        variant=args.variant,
        beta_deg=effective_beta,
        d_g_um_list=args.dg,
        belt_list=args.belt,
        eps_list=args.eps_list,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        n_solves=len(rows),
        n_pairs=len(pair_rows),
        total_time_sec=total,
        hs_warmup_warnings=dict(
            count=ww_count,
            max_residual=ww_max_res,
            affected_configs=ww_configs,
        ),
    )
    with open(os.path.join(args.out, "stageB_v2_stationary_fed_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total:.0f}s ({len(rows)} solves)")
    print(f"HS warmup warnings: {ww_count} (max res={ww_max_res:.2e})")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
