#!/usr/bin/env python3
"""Stage G4.2 — herringbone equilibrium continuation.

Continuation path from smooth → target (beta, dg) with pressure
warm-start (g_init) and scipy least_squares trust-region solver.
"""
from __future__ import annotations
import argparse, csv, datetime, json, math, os, sys, time, warnings
from typing import Any, Dict, List, Tuple, Optional
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import least_squares

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu as _ps_fn
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu as _ps_fn

from cases.gu_loaded_side_v4.geometry_builders import build_relief
from cases.gu_loaded_side_v4.schema import (
    SCHEMA, resolve_stage_dir, classify_status, TOL_HARD, TOL_SOFT,
    PROTECTED_LO_DEG, PROTECTED_HI_DEG,
)
from cases.gu_loaded_side_v4.common import (
    D, R, L, c as C_CLEARANCE, n_rpm, eta, sigma, w_g,
    MAX_ITER_NR, STEP_CAP, EPS_MAX,
)

OMEGA = 2 * math.pi * n_rpm / 60.0
P_SCALE = 6.0 * eta * OMEGA * (R / C_CLEARANCE) ** 2


def make_grid(Np, Nz):
    phi = np.linspace(0, 2 * math.pi, Np, endpoint=False)
    Z = np.linspace(-1, 1, Nz)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1] - phi[0], Z[1] - Z[0]


def pack_g_init(P, theta, eps=1e-14):
    g = np.where(P > eps, P, theta - 1.0)
    return np.ascontiguousarray(g, dtype=np.float64)


def _ps_call(H, dp, dz, phi_bc, g_init=None):
    kw = dict(tol=1e-6, max_iter=3_000_000,
              hs_warmup_iter=200_000, hs_warmup_tol=1e-5,
              phi_bc=phi_bc)
    if g_init is not None:
        kw["g_init"] = g_init
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return _ps_fn(H, dp, dz, R, L, **kw)


def eval_point(X, Y, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
               g_init=None):
    """Full eval: H build → PS solve → force integration.
    Returns (metrics_dict, P, theta) for warm-start chain."""
    H0 = 1.0 + float(X) * np.cos(Phi) + float(Y) * np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0 ** 2 + (sigma / C_CLEARANCE) ** 2)
    H = H0 + relief
    P, theta, _, _ = _ps_call(H, dp, dz, phi_bc, g_init)
    Pd = P * P_SCALE
    Fx = -np.trapezoid(
        np.trapezoid(Pd * np.cos(Phi), p1, axis=1),
        z1, axis=0) * R * L / 2.0
    Fy = -np.trapezoid(
        np.trapezoid(Pd * np.sin(Phi), p1, axis=1),
        z1, axis=0) * R * L / 2.0
    hd = H * C_CLEARANCE
    hm = float(np.min(hd))
    pm = float(np.max(Pd))
    cv = float(np.mean(theta < 1 - 1e-6))
    tc = eta * OMEGA * R / hd
    dPdp = np.gradient(Pd, dp, axis=1)
    tp = hd / 2 * dPdp / R
    ta = tc + tp
    fr = float(np.trapezoid(
        np.trapezoid(np.abs(ta), p1, axis=1),
        z1, axis=0) * R * L / 2.0)
    return dict(Fx=float(Fx), Fy=float(Fy), h_min=hm, p_max=pm,
                cav_frac=cv, friction=fr), P, theta


def solve_equilibrium_nr(W_applied, Phi, Zm, p1, z1, dp, dz,
                          relief, phi_bc, X0, Y0, g_init_0=None):
    """Backtracking NR (proven in v3/v4) with g_init warm-start."""
    Wa = np.asarray(W_applied, dtype=float)
    Wn = float(np.linalg.norm(Wa))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)
    g_cur = g_init_0

    m, P, theta = eval_point(X, Y, Phi, Zm, p1, z1, dp, dz,
                              relief, phi_bc, g_init=g_cur)
    g_cur = pack_g_init(P, theta)
    rel_R = math.sqrt((m["Fx"] - Wa[0])**2 + (m["Fy"] - Wa[1])**2) / max(Wn, 1e-20)
    Fx_h, Fy_h = m["Fx"], m["Fy"]
    h_min, p_max, cav, fr = m["h_min"], m["p_max"], m["cav_frac"], m["friction"]

    for it in range(MAX_ITER_NR):
        if rel_R < TOL_HARD:
            break
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            mp, _, _ = eval_point(X + dX_, Y + dY_, Phi, Zm, p1, z1,
                                   dp, dz, relief, phi_bc, g_init=g_cur)
            mn, _, _ = eval_point(X - dX_, Y - dY_, Phi, Zm, p1, z1,
                                   dp, dz, relief, phi_bc, g_init=g_cur)
            J[0, col] = (mp["Fx"] - mn["Fx"]) / (2 * dXY)
            J[1, col] = (mp["Fy"] - mn["Fy"]) / (2 * dXY)
        Rx = Fx_h - Wa[0]
        Ry = Fy_h - Wa[1]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        ddX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        ddY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det
        cap = STEP_CAP / max(abs(ddX), abs(ddY), 1e-20)
        if cap < 1:
            ddX *= cap
            ddY *= cap
        ok = False
        for alpha in [1.0, 0.5, 0.25, 0.125]:
            Xt = X + alpha * ddX
            Yt = Y + alpha * ddY
            if math.sqrt(Xt**2 + Yt**2) >= EPS_MAX:
                continue
            mt, Pt, tht = eval_point(Xt, Yt, Phi, Zm, p1, z1,
                                      dp, dz, relief, phi_bc, g_init=g_cur)
            rt = math.sqrt((mt["Fx"] - Wa[0])**2 + (mt["Fy"] - Wa[1])**2) / max(Wn, 1e-20)
            if rt < rel_R:
                X, Y = Xt, Yt
                Fx_h, Fy_h = mt["Fx"], mt["Fy"]
                h_min, p_max, cav, fr = mt["h_min"], mt["p_max"], mt["cav_frac"], mt["friction"]
                rel_R = rt
                g_cur = pack_g_init(Pt, tht)
                ok = True
                break
        if not ok:
            break

    eps = math.sqrt(X**2 + Y**2)
    COF = fr / max(Wn, 1e-20)
    status = classify_status(rel_R, rel_R <= 0.10)
    return dict(X=X, Y=Y, eps=eps, h_min=h_min, p_max=p_max,
                cav_frac=cav, friction=fr, COF=COF,
                rel_residual=rel_R, status=status), g_cur


def build_continuation_path(target_beta, target_dg):
    """Build (beta, dg) path from smooth to target."""
    path = [(0, 0)]
    betas = sorted(set([10, 15, 20, 25, 30] +
                       [target_beta] +
                       [b for b in range(35, target_beta + 1, 5)]))
    dgs = sorted(set([5, 10, 15] +
                     [target_dg] +
                     [d for d in range(20, target_dg + 1, 5)]))

    for b in betas:
        if b <= 0:
            continue
        path.append((b, min(5, target_dg)))
    for d in dgs:
        if d <= 5:
            continue
        path.append((target_beta, d))
    # Deduplicate preserving order
    seen = set()
    unique = []
    for p in path:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--stageA", required=True)
    pa.add_argument("--target-beta", type=float, default=30)
    pa.add_argument("--target-dg", type=float, default=15)
    pa.add_argument("--N-branch", type=int, default=10)
    pa.add_argument("--taper", type=float, default=0.6)
    pa.add_argument("--chirality", default="pump_to_edge")
    pa.add_argument("--coverage-mode", default="protect_loaded_union")
    pa.add_argument("--feed-mode", default="groove")
    pa.add_argument("--loadcases", nargs="+", default=["L50"])
    pa.add_argument("--grid", default="800x240")
    pa.add_argument("--out", required=True)
    args = pa.parse_args()

    Np, Nz = (int(x) for x in args.grid.split("x"))
    p1, z1, Phi, Zm, dp, dz = make_grid(Np, Nz)
    os.makedirs(args.out, exist_ok=True)

    stA = resolve_stage_dir(args.stageA)
    with open(os.path.join(stA, "working_geometry_manifest.json")) as f:
        mA = json.load(f)

    path = build_continuation_path(int(args.target_beta),
                                    int(args.target_dg))
    print(f"Stage G4.2: herringbone equilibrium continuation")
    print(f"Grid: {Np}x{Nz}, N={args.N_branch}, taper={args.taper}")
    print(f"Target: beta={args.target_beta}°, dg={args.target_dg}μm")
    print(f"Chirality: {args.chirality}")
    print(f"Path ({len(path)} steps): {path}")

    all_rows: List[Dict] = []
    t_global = time.time()

    for lc_name in args.loadcases:
        lc = mA["loadcases"].get(lc_name)
        if not lc:
            print(f"  {lc_name}: not found in stageA — skipping")
            continue
        Wa = np.array(lc["applied_load_N"], dtype=float)
        Wn = float(np.linalg.norm(Wa))
        Wd = Wa / max(Wn, 1e-20)
        eps_ref = float(lc["eps_source"])
        X_seed = -eps_ref * float(Wd[0])
        Y_seed = -eps_ref * float(Wd[1])

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wn:.1f}N  eps_ref={eps_ref}")
        print(f"{'='*60}")

        X_prev, Y_prev = X_seed, Y_seed
        g_prev = None
        chain_broken = False

        for step_i, (beta_step, dg_step) in enumerate(path):
            if chain_broken:
                all_rows.append(dict(
                    loadcase=lc_name, step=step_i,
                    beta_deg=beta_step, d_g_um=dg_step,
                    status="skipped_chain_broken"))
                continue

            # Step 0 (smooth): get g_init from anchor position, don't
            # re-solve (groove BC smooth NR diverges). Honest comparison
            # done at end: eval smooth at target (X,Y).
            if beta_step == 0 and dg_step == 0:
                t0 = time.time()
                m0, P0, th0 = eval_point(
                    X_prev, Y_prev, Phi, Zm, p1, z1, dp, dz,
                    np.zeros_like(Phi), args.feed_mode)
                g_prev = pack_g_init(P0, th0)
                dt = time.time() - t0
                Wn_c = float(np.linalg.norm(Wa))
                d = dict(X=X_prev, Y=Y_prev,
                         eps=math.sqrt(X_prev**2 + Y_prev**2),
                         h_min=m0["h_min"], p_max=m0["p_max"],
                         cav_frac=m0["cav_frac"], friction=m0["friction"],
                         COF=m0["friction"] / max(Wn_c, 1e-20),
                         rel_residual=0, status="anchor_seed")
                print(f"  [ 0] beta= 0 dg= 0 → eps={d['eps']:.4f} "
                      f"COF={d['COF']:.6f} h={d['h_min']*1e6:.1f}μm "
                      f"[anchor_seed→g_init] {dt:.0f}s")
                row = dict(loadcase=lc_name, step=0,
                           beta_deg=0, d_g_um=0,
                           X=d["X"], Y=d["Y"], eps=d["eps"],
                           h_min_um=d["h_min"]*1e6,
                           p_max_MPa=d["p_max"]/1e6,
                           COF=d["COF"], dg_over_hmin=0,
                           cav_frac=d["cav_frac"],
                           rel_residual=0,
                           status="anchor_seed", elapsed_sec=dt)
                all_rows.append(row)
                continue

            relief = build_relief(
                Phi, Zm, variant="half_herringbone_ramped",
                depth_nondim=dg_step * 1e-6 / C_CLEARANCE,
                N_branch_per_side=args.N_branch,
                w_branch_nondim=w_g / R,
                belt_half_nondim=0.15,
                beta_deg=beta_step,
                chirality=args.chirality,
                ramp_frac=0.15,
                taper_ratio=args.taper,
                coverage_mode=args.coverage_mode,
                protected_lo_deg=PROTECTED_LO_DEG,
                protected_hi_deg=PROTECTED_HI_DEG)

            t0 = time.time()
            d, g_new = solve_equilibrium_nr(
                Wa, Phi, Zm, p1, z1, dp, dz,
                relief, args.feed_mode,
                X_prev, Y_prev, g_init_0=g_prev)
            dt = time.time() - t0

            dg_hm = (dg_step * 1e-6) / max(d["h_min"], 1e-12)
            tag = f"beta={beta_step:2.0f} dg={dg_step:2.0f}"
            feasible = d["status"] in ("hard_converged", "soft_converged")

            print(f"  [{step_i:2d}] {tag} → "
                  f"eps={d['eps']:.4f} "
                  f"COF={d['COF']:.6f} "
                  f"h={d['h_min']*1e6:.1f}μm "
                  f"dg/h={dg_hm:.2f} "
                  f"res={d['rel_residual']:.1e} "
                  f"[{d['status']}] {dt:.0f}s")

            row = dict(
                loadcase=lc_name, step=step_i,
                beta_deg=beta_step, d_g_um=dg_step,
                X=d["X"], Y=d["Y"], eps=d["eps"],
                h_min_um=d["h_min"] * 1e6,
                p_max_MPa=d["p_max"] / 1e6,
                COF=d["COF"], dg_over_hmin=dg_hm,
                cav_frac=d["cav_frac"],
                rel_residual=d["rel_residual"],
                status=d["status"],
                elapsed_sec=dt)
            all_rows.append(row)

            if feasible:
                X_prev, Y_prev = d["X"], d["Y"]
                g_prev = g_new
            else:
                print(f"    [CHAIN BREAK] at beta={beta_step} dg={dg_step}")
                chain_broken = True

    total = time.time() - t_global

    # Honest comparison: eval smooth at target equilibrium position
    target_row = None
    for r in reversed(all_rows):
        if (r.get("beta_deg") == args.target_beta
                and r.get("d_g_um") == args.target_dg
                and r.get("status") not in ("skipped_chain_broken", "failed")):
            target_row = r
            break

    if target_row:
        # Eval smooth bearing at target's (X, Y) for fair COF comparison
        Xt, Yt = target_row["X"], target_row["Y"]
        m_sm, _, _ = eval_point(
            Xt, Yt, Phi, Zm, p1, z1, dp, dz,
            np.zeros_like(Phi), args.feed_mode)
        Wn_h = float(np.linalg.norm(
            np.array(mA["loadcases"][args.loadcases[0]]["applied_load_N"])))
        COF_smooth_at_target = m_sm["friction"] / max(Wn_h, 1e-20)
        h_smooth_at_target = m_sm["h_min"] * 1e6

        dCOF = (target_row["COF"] - COF_smooth_at_target) / max(COF_smooth_at_target, 1e-20) * 100
        dh = (target_row["h_min_um"] - h_smooth_at_target) / max(h_smooth_at_target, 1e-20) * 100
        print(f"\n{'='*60}")
        print(f"Headline: target vs smooth AT SAME (X,Y)")
        print(f"  smooth COF @ target pos = {COF_smooth_at_target:.6f}")
        print(f"  veined COF @ target pos = {target_row['COF']:.6f}")
        print(f"  ΔCOF = {dCOF:+.1f}%")
        print(f"  smooth h_min = {h_smooth_at_target:.1f}μm")
        print(f"  veined h_min = {target_row['h_min_um']:.1f}μm")
        print(f"  Δh_min = {dh:+.1f}%")
        print(f"  dg/hmin = {target_row['dg_over_hmin']:.2f}")
        print(f"  status = {target_row['status']}")

        weak = dCOF <= -3 and dh >= -2
        strong = dCOF <= -5 and dh >= 0
        gate = "HB_ACCEPTED_STRONG" if strong else (
            "HB_ACCEPTED_WEAK" if weak else "HB_FIXED_ONLY_NO_EQ")
        print(f"  Gate: {gate}")
    elif not target_row:
        gate = "HB_FIXED_ONLY_NO_EQ"
        print(f"\nTarget not reached — chain broken before target")
        print(f"Gate: {gate}")
    else:
        gate = "HB_FIXED_ONLY_NO_EQ"

    # CSV
    if all_rows:
        cp = os.path.join(args.out, "continuation_results.csv")
        flds = sorted({k for r in all_rows for k in r.keys()})
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    manifest = dict(
        schema_version=SCHEMA, stage="G4.2_herringbone_eq_continuation",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        target_beta=args.target_beta, target_dg=args.target_dg,
        N_branch=args.N_branch, taper=args.taper,
        chirality=args.chirality, coverage=args.coverage_mode,
        feed=args.feed_mode, loadcases=args.loadcases,
        grid=dict(N_phi=Np, N_Z=Nz),
        continuation_path=path,
        gate=gate, total_time_sec=total)
    with open(os.path.join(args.out, "eq_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Path log
    with open(os.path.join(args.out, "continuation_path_log.txt"), "w") as f:
        for r in all_rows:
            f.write(f"step={r.get('step','')} beta={r.get('beta_deg','')} "
                    f"dg={r.get('d_g_um','')} "
                    f"eps={r.get('eps','')} "
                    f"res={r.get('rel_residual','')} "
                    f"status={r.get('status','')}\n")

    print(f"\nTotal: {total:.0f}s")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
