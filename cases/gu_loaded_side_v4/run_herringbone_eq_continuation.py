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


def solve_equilibrium_trf(W_applied, Phi, Zm, p1, z1, dp, dz,
                           relief, phi_bc, X0, Y0, g_init_0=None):
    """Trust-region equilibrium solver with g_init warm-start."""
    Wa = np.asarray(W_applied, dtype=float)
    Wn = float(np.linalg.norm(Wa))
    last_g = [g_init_0]
    last_P = [None]
    last_theta = [None]

    def residual_fn(xy):
        X, Y = float(xy[0]), float(xy[1])
        m, P, theta = eval_point(
            X, Y, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
            g_init=last_g[0])
        last_P[0] = P
        last_theta[0] = theta
        last_g[0] = pack_g_init(P, theta)
        Rx = (m["Fx"] - Wa[0]) / max(Wn, 1e-20)
        Ry = (m["Fy"] - Wa[1]) / max(Wn, 1e-20)
        return np.array([Rx, Ry])

    try:
        result = least_squares(
            residual_fn, x0=[float(X0), float(Y0)],
            method="trf",
            bounds=([-(EPS_MAX - 0.01), -(EPS_MAX - 0.01)],
                    [EPS_MAX - 0.01, EPS_MAX - 0.01]),
            diff_step=[0.002, 0.002],
            max_nfev=200,
            ftol=TOL_HARD, xtol=1e-6, gtol=1e-6)
        X_sol, Y_sol = float(result.x[0]), float(result.x[1])
        rel_R = float(np.linalg.norm(result.fun))
    except Exception as e:
        print(f"    [TRF] exception: {e}")
        X_sol, Y_sol = float(X0), float(Y0)
        rel_R = 1.0

    m_final, P_final, theta_final = eval_point(
        X_sol, Y_sol, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
        g_init=last_g[0])
    rel_R_final = math.sqrt(
        (m_final["Fx"] - Wa[0]) ** 2 +
        (m_final["Fy"] - Wa[1]) ** 2) / max(Wn, 1e-20)

    eps = math.sqrt(X_sol ** 2 + Y_sol ** 2)
    COF = m_final["friction"] / max(Wn, 1e-20)
    status = classify_status(rel_R_final, rel_R_final <= 0.10)

    return dict(
        X=X_sol, Y=Y_sol, eps=eps,
        h_min=m_final["h_min"], p_max=m_final["p_max"],
        cav_frac=m_final["cav_frac"], friction=m_final["friction"],
        COF=COF, rel_residual=rel_R_final, status=status,
    ), pack_g_init(P_final, theta_final)


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
            d, g_new = solve_equilibrium_trf(
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

    # Compare final vs smooth anchor
    smooth_anchor = None
    for r in all_rows:
        if r.get("beta_deg") == 0 and r.get("d_g_um") == 0 and r.get("status") != "skipped_chain_broken":
            smooth_anchor = r
            break

    target_row = None
    for r in reversed(all_rows):
        if (r.get("beta_deg") == args.target_beta
                and r.get("d_g_um") == args.target_dg
                and r.get("status") not in ("skipped_chain_broken", "failed")):
            target_row = r
            break

    if smooth_anchor and target_row:
        dCOF = (target_row["COF"] - smooth_anchor["COF"]) / max(smooth_anchor["COF"], 1e-20) * 100
        dh = (target_row["h_min_um"] - smooth_anchor["h_min_um"]) / max(smooth_anchor["h_min_um"], 1e-20) * 100
        print(f"\n{'='*60}")
        print(f"Headline: target vs smooth anchor")
        print(f"  ΔCOF = {dCOF:+.1f}%")
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
