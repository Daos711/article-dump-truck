#!/usr/bin/env python3
"""Stage G4.0 — fixed-ε probe (no equilibrium search).

One PS call per point. Fast screening to identify which depths
produce real pumping before expensive equilibrium.
"""
from __future__ import annotations
import argparse, csv, datetime, json, math, os, sys, time
from typing import Any, Dict, List, Tuple
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu as _ps_fn
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu as _ps_fn
from cases.gu_loaded_side_v4.geometry_builders import build_relief
from cases.gu_loaded_side_v4.schema import (
    SCHEMA, resolve_stage_dir, PROTECTED_LO_DEG, PROTECTED_HI_DEG,
)
from cases.gu_loaded_side_v4.common import (
    D, R, L, c as C_CLEARANCE, n_rpm, eta, sigma, w_g,
)

def make_grid(Np, Nz):
    phi = np.linspace(0, 2*math.pi, Np, endpoint=False)
    Z = np.linspace(-1, 1, Nz)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1]-phi[0], Z[1]-Z[0]

_quiet_mode = False
_ww_count = 0

def _ps_call(H, dp, dz, phi_bc, ps_tol=1e-5, ps_max_iter=300_000):
    global _ww_count
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = _ps_fn(H, dp, dz, R, L, tol=ps_tol, max_iter=ps_max_iter,
                         phi_bc=phi_bc)
    for w in caught:
        if "warmup" in str(w.message).lower():
            _ww_count += 1
    return result

def eval_fixed(X, Y, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
               ps_tol=1e-5, ps_max_iter=300_000):
    om = 2*math.pi*n_rpm/60; ps = 6*eta*om*(R/C_CLEARANCE)**2
    H0 = 1.0 + float(X)*np.cos(Phi) + float(Y)*np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0**2 + (sigma/C_CLEARANCE)**2)
    H = H0 + relief
    P, theta, _, _ = _ps_call(H, dp, dz, phi_bc, ps_tol, ps_max_iter)
    Pd = P * ps
    Fx = -np.trapezoid(np.trapezoid(Pd*np.cos(Phi), p1, axis=1),
                        z1, axis=0)*R*L/2
    Fy = -np.trapezoid(np.trapezoid(Pd*np.sin(Phi), p1, axis=1),
                        z1, axis=0)*R*L/2
    hd = H * C_CLEARANCE
    hm = float(np.min(hd))
    pm = float(np.max(Pd))
    cv = float(np.mean(theta < 1-1e-6))
    tc = eta*om*R/hd
    dPdp = np.gradient(Pd, dp, axis=1)
    tp = hd/2*dPdp/R
    ta = tc + tp
    fr = float(np.trapezoid(np.trapezoid(np.abs(ta), p1, axis=1),
                              z1, axis=0)*R*L/2)
    W = math.sqrt(float(Fx)**2 + float(Fy)**2)
    COF = fr / max(W, 1e-20)
    return dict(Fx=float(Fx), Fy=float(Fy), W=W, COF=COF,
                h_min=hm, p_max=pm, cav_frac=cv, friction=fr)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--stageA", required=True)
    pa.add_argument("--variants", nargs="+", default=["straight_ramped","arc_ramped"])
    pa.add_argument("--N-branch-list", type=int, nargs="+", default=[6,8,10])
    pa.add_argument("--dg-list", type=float, nargs="+", default=[10,15,20,25,50])
    pa.add_argument("--belt-list", type=float, nargs="+", default=[0.15])
    pa.add_argument("--taper-list", type=float, nargs="+", default=[1.0,0.6])
    pa.add_argument("--curvature-k-list", type=float, nargs="+", default=[0.15])
    pa.add_argument("--beta-list", type=float, nargs="+", default=[20])
    pa.add_argument("--chirality-list", nargs="+", default=["pump_to_belt"])
    pa.add_argument("--coverage-modes", nargs="+", default=["protect_loaded_union","full_360"])
    pa.add_argument("--feed-mode", default="groove")
    pa.add_argument("--eps-list", type=float, nargs="+", default=[0.5,0.8])
    pa.add_argument("--attitude-scan-deg", type=float, nargs="+", default=[-8,0,8])
    pa.add_argument("--grid", default="400x120")
    pa.add_argument("--quiet", action="store_true",
                    help="suppress HS warmup warnings from stdout")
    pa.add_argument("--out", required=True)
    args = pa.parse_args()

    Np, Nz = (int(x) for x in args.grid.split("x"))
    p1, z1, Phi, Zm, dp, dz = make_grid(Np, Nz)
    os.makedirs(args.out, exist_ok=True)

    stA = resolve_stage_dir(args.stageA)
    with open(os.path.join(stA, "working_geometry_manifest.json")) as f:
        mA = json.load(f)
    anchors = {}
    for lc, v in mA.get("loadcases", {}).items():
        anchors[float(v["eps_source"])] = dict(
            W=v["applied_load_N"],
            phi_loaded=v.get("phi_loaded_deg", 140))

    global _quiet_mode, _ww_count
    _quiet_mode = args.quiet
    _ww_count = 0

    print(f"Stage G4.0: fixed-ε probe")
    print(f"Grid: {Np}x{Nz}, feed={args.feed_mode}")
    print(f"Variants: {args.variants}, N: {args.N_branch_list}")
    print(f"d_g: {args.dg_list} μm, beta: {args.beta_list}°")
    print(f"chirality: {args.chirality_list}")
    print(f"attitude scan: {args.attitude_scan_deg}°")

    smooth_cache: Dict[Tuple, Dict] = {}
    rows: List[Dict] = []
    t0g = time.time()

    for eps in args.eps_list:
        anc = anchors.get(eps, {})
        Wa = np.array(anc.get("W", [0, -100]), dtype=float)
        Wn = float(np.linalg.norm(Wa))
        Wd = Wa / max(Wn, 1e-20)
        base_att = math.atan2(-Wd[1], -Wd[0])

        for att_offset in args.attitude_scan_deg:
            att = base_att + math.radians(att_offset)
            X0 = eps * math.cos(att)
            Y0 = eps * math.sin(att)

            for cov in args.coverage_modes:
                sk = (eps, att_offset, cov, args.feed_mode, args.grid)
                if sk not in smooth_cache:
                    t0 = time.time()
                    ds = eval_fixed(X0, Y0, Phi, Zm, p1, z1, dp, dz,
                                     np.zeros_like(Phi), args.feed_mode)
                    smooth_cache[sk] = ds
                    dt = time.time() - t0
                    print(f"  [smooth] eps={eps} att={att_offset:+.0f}° "
                          f"cov={cov} COF={ds['COF']:.6f} W={ds['W']:.1f}N "
                          f"h={ds['h_min']*1e6:.1f}μm {dt:.1f}s")
                else:
                    ds = smooth_cache[sk]

                for var in args.variants:
                    for Nb in args.N_branch_list:
                        for dg in args.dg_list:
                            taper_list = args.taper_list
                            curv_list = args.curvature_k_list if var == "arc_ramped" else [0.0]
                            beta_list = args.beta_list if "herringbone" in var else [0.0]
                            chir_list = args.chirality_list if "herringbone" in var else ["pump_to_belt"]
                            for taper in taper_list:
                                for curv in curv_list:
                                    for beta in beta_list:
                                        for chir in chir_list:
                                            for belt in args.belt_list:
                                                relief = build_relief(
                                                    Phi, Zm, variant=var,
                                                    depth_nondim=dg*1e-6/C_CLEARANCE,
                                                    N_branch_per_side=Nb,
                                                    w_branch_nondim=w_g/R,
                                                    belt_half_nondim=belt,
                                                    beta_deg=beta,
                                                    curvature_k=curv,
                                                    chirality=chir,
                                                    ramp_frac=0.15,
                                                    taper_ratio=taper,
                                                    coverage_mode=cov,
                                                    protected_lo_deg=PROTECTED_LO_DEG,
                                                    protected_hi_deg=PROTECTED_HI_DEG)
                                                t0 = time.time()
                                                dv = eval_fixed(
                                                    X0, Y0, Phi, Zm, p1, z1,
                                                    dp, dz, relief, args.feed_mode)
                                                dt = time.time() - t0

                                                dCOF = (dv["COF"]-ds["COF"])/max(ds["COF"],1e-20)*100
                                                Ws = np.array([ds["Fx"], ds["Fy"]])
                                                Wt = np.array([dv["Fx"], dv["Fy"]])
                                                Ws_n = max(float(np.linalg.norm(Ws)), 1e-20)
                                                dot_proj = float(np.dot(Wt, Ws)) / Ws_n
                                                gain_par = (dot_proj / Ws_n - 1) * 100
                                                cross = abs(float(Wt[0]*Ws[1] - Wt[1]*Ws[0]))
                                                dot_abs = abs(float(np.dot(Wt, Ws)))
                                                side_ratio = cross / max(dot_abs, 1e-20)
                                                dp_pct = (dv["p_max"]-ds["p_max"])/max(ds["p_max"],1e-20)*100
                                                dg_hm = (dg*1e-6)/max(dv["h_min"],1e-12)
                                                dh_pct = (dv["h_min"]-ds["h_min"])/max(ds["h_min"],1e-20)*100

                                                row = dict(
                                                    variant=var, N_branch=Nb, d_g_um=dg,
                                                    belt=belt, taper_ratio=taper,
                                                    curvature_k=curv, beta_deg=beta,
                                                    chirality=chir, coverage_mode=cov,
                                                    eps_ref=eps, attitude_offset_deg=att_offset,
                                                    dCOF_fixed_pct=dCOF,
                                                    load_gain_parallel_pct=gain_par,
                                                    side_force_ratio=side_ratio,
                                                    dp_max_pct=dp_pct,
                                                    dh_pct=dh_pct,
                                                    dg_over_hmin=dg_hm,
                                                    h_min_um=dv["h_min"]*1e6,
                                                    W_tex=dv["W"],
                                                    W_smooth=ds["W"],
                                                    COF_tex=dv["COF"],
                                                    COF_smooth=ds["COF"],
                                                    elapsed_sec=dt)
                                                rows.append(row)

                                                chir_short = "edge" if chir == "pump_to_edge" else "belt"
                                                tag = f"{var[:5]}_N{Nb}_dg{dg}_t{taper}_b{beta}_ch_{chir_short}"
                                                print(f"    {tag:45s} cov={cov[:8]:8s} "
                                                      f"att={att_offset:+.0f}° "
                                                      f"ΔCOF={dCOF:+.1f}% "
                                                      f"gain_W={gain_par:+.1f}% "
                                                      f"side={side_ratio:.3f} "
                                                      f"dg/h={dg_hm:.2f} "
                                                      f"{dt:.1f}s")

    total = time.time() - t0g

    # Filter candidates for G4.1
    cands = [r for r in rows
             if (r["dCOF_fixed_pct"] <= -3 or r["load_gain_parallel_pct"] >= 5)
             and r["side_force_ratio"] <= 0.25
             and r["dp_max_pct"] <= 75
             and r["eps_ref"] == 0.5
             and r["dg_over_hmin"] <= 3.0]
    cands.sort(key=lambda r: r["dCOF_fixed_pct"])
    top6 = cands[:6]

    print(f"\n{'='*60}")
    print(f"Fixed-ε probe complete: {len(rows)} points in {total:.0f}s")
    n_promising = len(cands)
    print(f"Candidates for G4.1: {n_promising} passed filter, top-6:")
    for i, c in enumerate(top6):
        print(f"  [{i+1}] {c['variant']} N={c['N_branch']} dg={c['d_g_um']} "
              f"t={c['taper_ratio']} k={c['curvature_k']} "
              f"cov={c['coverage_mode'][:8]} att={c['attitude_offset_deg']:+.0f}° "
              f"ΔCOF={c['dCOF_fixed_pct']:+.1f}% "
              f"gain_W={c['load_gain_parallel_pct']:+.1f}%")
    recommend = "GO" if n_promising >= 3 else "NO_GO"
    print(f"\nG4.1 recommendation: {recommend}")

    # CSV
    if rows:
        cp = os.path.join(args.out, "fixed_eps_probe_results.csv")
        flds = sorted(rows[0].keys())
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    with open(os.path.join(args.out, "selected_candidates.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(schema_version=SCHEMA, n_filtered=n_promising,
                        candidates=top6), f, indent=2, ensure_ascii=False)

    manifest = dict(
        schema_version=SCHEMA, stage="G4_fixed_eps_probe",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        grid=dict(N_phi=Np, N_Z=Nz),
        n_points=len(rows), n_smooth_cached=len(smooth_cache),
        n_candidates=n_promising, recommendation=recommend,
        total_time_sec=total)
    with open(os.path.join(args.out, "fixed_eps_probe_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary MD
    with open(os.path.join(args.out, "fixed_eps_probe_summary.md"), "w",
              encoding="utf-8") as f:
        f.write("# Stage G4.0 — fixed-ε probe\n\n")
        f.write(f"- {len(rows)} points, {total:.0f}s\n")
        f.write(f"- candidates for G4.1: {n_promising}\n")
        f.write(f"- recommendation: **{recommend}**\n\n")
        if top6:
            f.write("## Top-6 candidates\n\n")
            f.write("| # | variant | N | dg | taper | curv | cov | att | ΔCOF | gain_W | side |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for i, c in enumerate(top6):
                f.write(f"| {i+1} | {c['variant']} | {c['N_branch']} | "
                        f"{c['d_g_um']} | {c['taper_ratio']} | "
                        f"{c['curvature_k']} | {c['coverage_mode'][:8]} | "
                        f"{c['attitude_offset_deg']:+.0f}° | "
                        f"{c['dCOF_fixed_pct']:+.1f}% | "
                        f"{c['load_gain_parallel_pct']:+.1f}% | "
                        f"{c['side_force_ratio']:.3f} |\n")

    print(f"\nArtifacts: {args.out}/")

if __name__ == "__main__":
    main()
