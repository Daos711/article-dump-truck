#!/usr/bin/env python3
"""Stage H fixed-ε — supply pressure probe without equilibrium search.

One PS call per point. p_supply continuation warm-start within each
(geometry, eps, feed_variant) chain.
"""
from __future__ import annotations
import argparse, csv, datetime, json, math, os, sys, time, warnings
from typing import Any, Dict, List, Tuple
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu as _ps_fn
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu as _ps_fn

from models.feed_geometry import (
    build_feed_mask_variant, p_supply_to_g_bc,
)
from cases.gu_loaded_side_v4.geometry_builders import build_relief
from cases.gu_loaded_side_v4.schema import (
    SCHEMA, resolve_stage_dir, PROTECTED_LO_DEG, PROTECTED_HI_DEG,
)
from cases.gu_loaded_side_v4.common import (
    D, R, L, c as C_CLEARANCE, n_rpm, eta, sigma, w_g,
)

OMEGA = 2 * math.pi * n_rpm / 60.0
P_SCALE = 6.0 * eta * OMEGA * (R / C_CLEARANCE) ** 2


def make_grid(Np, Nz):
    phi = np.linspace(0, 2*math.pi, Np, endpoint=False)
    Z = np.linspace(-1, 1, Nz)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1]-phi[0], Z[1]-Z[0]


def pack_g_init(P, theta, eps=1e-14):
    g = np.where(P > eps, P, theta - 1.0)
    return np.ascontiguousarray(g, dtype=np.float64)


def _ps_call(H, dp, dz, phi_bc, g_init=None,
             dirichlet_mask=None, g_bc=None):
    kw = dict(tol=1e-5, max_iter=300_000, phi_bc=phi_bc)
    if g_init is not None:
        kw["g_init"] = g_init
    if dirichlet_mask is not None and g_bc is not None:
        kw["dirichlet_mask"] = dirichlet_mask
        kw["g_bc"] = g_bc
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return _ps_fn(H, dp, dz, R, L, **kw)


def eval_fixed(X, Y, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
               g_init=None, dirichlet_mask=None, g_bc=None):
    H0 = 1.0 + float(X)*np.cos(Phi) + float(Y)*np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0**2 + (sigma/C_CLEARANCE)**2)
    H = H0 + relief
    P, theta, ps_res, ps_iters = _ps_call(
        H, dp, dz, phi_bc, g_init, dirichlet_mask, g_bc)
    Pd = P * P_SCALE
    Fx = -np.trapezoid(np.trapezoid(Pd*np.cos(Phi), p1, axis=1),
                        z1, axis=0)*R*L/2
    Fy = -np.trapezoid(np.trapezoid(Pd*np.sin(Phi), p1, axis=1),
                        z1, axis=0)*R*L/2
    hd = H * C_CLEARANCE
    hm = float(np.min(hd))
    pm = float(np.max(Pd))
    cv = float(np.mean(theta < 1-1e-6))
    tc = eta*OMEGA*R/hd
    dPdp = np.gradient(Pd, dp, axis=1)
    tp = hd/2*dPdp/R
    ta = tc + tp
    fr = float(np.trapezoid(np.trapezoid(np.abs(ta), p1, axis=1),
                              z1, axis=0)*R*L/2)
    W = math.sqrt(float(Fx)**2 + float(Fy)**2)
    COF = fr / max(W, 1e-20)
    ang = math.degrees(math.atan2(float(Fy), float(Fx)))
    return dict(
        Fx=float(Fx), Fy=float(Fy), W=W, COF=COF,
        h_min=hm, p_max=pm, cav_frac=cv, friction=fr,
        force_angle_deg=ang,
        ps_iters=int(ps_iters) if ps_iters is not None else 0,
        ps_residual=float(ps_res) if ps_res is not None else 0.0,
        ps_converged=bool(ps_res is not None and float(ps_res) < 1e-5),
    ), P, theta


def main():
    pa = argparse.ArgumentParser(description="Stage H: fixed-ε supply pressure probe")
    pa.add_argument("--stageA", required=True)
    pa.add_argument("--target-beta", type=float, default=45)
    pa.add_argument("--target-dg", type=float, default=15)
    pa.add_argument("--N-branch", type=int, default=10)
    pa.add_argument("--taper", type=float, default=0.6)
    pa.add_argument("--chirality", default="pump_to_edge")
    pa.add_argument("--coverage-mode", default="protect_loaded_union")
    pa.add_argument("--phi-bc", default="periodic")
    pa.add_argument("--p-supply-bar", type=float, nargs="+", default=[0, 0.5, 2, 5])
    pa.add_argument("--feed-variants", nargs="+", default=["belt_wide"])
    pa.add_argument("--phi-feed-half-deg", type=float, default=5)
    pa.add_argument("--z-belt-half", type=float, default=0.15)
    pa.add_argument("--phi-feed-deg", type=float, default=None)
    pa.add_argument("--eps-list", type=float, nargs="+", default=[0.5])
    pa.add_argument("--grid", default="400x120")
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

    tex_relief = build_relief(
        Phi, Zm, variant="half_herringbone_ramped",
        depth_nondim=args.target_dg*1e-6/C_CLEARANCE,
        N_branch_per_side=args.N_branch,
        w_branch_nondim=w_g/R, belt_half_nondim=0.15,
        beta_deg=args.target_beta, chirality=args.chirality,
        ramp_frac=0.15, taper_ratio=args.taper,
        coverage_mode=args.coverage_mode,
        protected_lo_deg=PROTECTED_LO_DEG,
        protected_hi_deg=PROTECTED_HI_DEG)
    smooth_relief = np.zeros_like(Phi)

    p_list = sorted(args.p_supply_bar)

    print(f"Stage H fixed-ε: supply pressure probe")
    print(f"Grid: {Np}x{Nz}, phi_bc={args.phi_bc}")
    print(f"p_supply: {p_list} bar")
    print(f"Feed variants: {args.feed_variants}")
    print(f"eps: {args.eps_list}")

    rows: List[Dict] = []
    t_global = time.time()

    for eps in args.eps_list:
        anc = anchors.get(eps, {})
        Wa = np.array(anc.get("W", [0, -100]), dtype=float)
        Wn = float(np.linalg.norm(Wa))
        Wd = Wa / max(Wn, 1e-20)
        base_att = math.atan2(-Wd[1], -Wd[0])
        X0 = eps * math.cos(base_att)
        Y0 = eps * math.sin(base_att)
        phi_loaded = float(anc.get("phi_loaded", 140))

        for fv in args.feed_variants:
            for geo_tag, relief in [("smooth", smooth_relief),
                                     ("textured", tex_relief)]:
                g_prev = None
                for p_bar in p_list:
                    p_Pa = float(p_bar) * 1e5
                    if p_bar > 0:
                        g_bc_val = p_supply_to_g_bc(p_Pa, eta, OMEGA, R, C_CLEARANCE)
                        mask, meta = build_feed_mask_variant(
                            Phi, Zm, fv,
                            phi_loaded_deg=phi_loaded,
                            phi_feed_deg=args.phi_feed_deg,
                            phi_feed_half_deg=args.phi_feed_half_deg,
                            z_belt_half=args.z_belt_half)
                    else:
                        g_bc_val = 0.0
                        mask = None
                        meta = dict(variant=fv, n_cells=0, frac=0,
                                     phi_center_deg=None)

                    t0 = time.time()
                    dv, P, theta = eval_fixed(
                        X0, Y0, Phi, Zm, p1, z1, dp, dz,
                        relief, args.phi_bc,
                        g_init=g_prev,
                        dirichlet_mask=mask,
                        g_bc=g_bc_val if p_bar > 0 else None)
                    dt = time.time() - t0
                    g_prev = pack_g_init(P, theta)

                    row = dict(
                        geometry=geo_tag, feed_variant=fv,
                        grid=args.grid, eps=eps,
                        phi_bc=args.phi_bc,
                        p_supply_bar=p_bar, g_bc=g_bc_val,
                        mask_cells=meta.get("n_cells", 0),
                        mask_frac=meta.get("frac", 0),
                        phi_center_deg=meta.get("phi_center_deg"),
                        Fx=dv["Fx"], Fy=dv["Fy"], W=dv["W"],
                        COF=dv["COF"],
                        hmin_um=dv["h_min"]*1e6,
                        pmax_MPa=dv["p_max"]/1e6,
                        cav_frac=dv["cav_frac"],
                        force_angle_deg=dv["force_angle_deg"],
                        ps_iters=dv["ps_iters"],
                        ps_residual=dv["ps_residual"],
                        ps_converged=dv["ps_converged"],
                        elapsed_sec=dt)
                    rows.append(row)

                    print(f"  {geo_tag:>10s} {fv:>15s} eps={eps} "
                          f"p={p_bar}bar COF={dv['COF']:.6f} "
                          f"W={dv['W']:.1f}N h={dv['h_min']*1e6:.1f}μm "
                          f"ps_it={dv['ps_iters']} "
                          f"ps_res={dv['ps_residual']:.1e} "
                          f"{'✓' if dv['ps_converged'] else '✗'} "
                          f"{dt:.1f}s")

    total = time.time() - t_global

    # Pairwise: textured vs smooth at same (eps, feed_variant, p_supply)
    print(f"\n{'='*60}")
    print(f"Pairwise: textured vs smooth")
    for fv in args.feed_variants:
        for eps in args.eps_list:
            for p_bar in p_list:
                sm = next((r for r in rows if r["geometry"]=="smooth"
                            and r["feed_variant"]==fv and r["eps"]==eps
                            and r["p_supply_bar"]==p_bar), None)
                tx = next((r for r in rows if r["geometry"]=="textured"
                            and r["feed_variant"]==fv and r["eps"]==eps
                            and r["p_supply_bar"]==p_bar), None)
                if sm and tx:
                    dCOF = (tx["COF"]-sm["COF"])/max(sm["COF"],1e-20)*100
                    gW = (tx["W"]-sm["W"])/max(sm["W"],1e-20)*100
                    Ws = np.array([sm["Fx"],sm["Fy"]])
                    Wt = np.array([tx["Fx"],tx["Fy"]])
                    Ws_n = max(float(np.linalg.norm(Ws)),1e-20)
                    cross = abs(float(Wt[0]*Ws[1]-Wt[1]*Ws[0]))
                    dot_a = abs(float(np.dot(Wt,Ws)))
                    side = cross/max(dot_a,1e-20)
                    print(f"  {fv:>15s} eps={eps} p={p_bar}bar: "
                          f"ΔCOF={dCOF:+.1f}% gain_W={gW:+.1f}% "
                          f"side={side:.3f}")

    # CSV
    if rows:
        cp = os.path.join(args.out, "stageH_fixed_eps_raw.csv")
        flds = sorted(rows[0].keys())
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    manifest = dict(
        schema_version=SCHEMA, stage="H_fixed_eps_supply",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        phi_bc=args.phi_bc, p_supply_bar_list=p_list,
        feed_variants=args.feed_variants, eps_list=args.eps_list,
        grid=dict(N_phi=Np, N_Z=Nz),
        total_time_sec=total, n_points=len(rows))
    with open(os.path.join(args.out, "stageH_fixed_eps_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total:.0f}s ({len(rows)} points)")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
