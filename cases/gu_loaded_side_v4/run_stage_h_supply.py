#!/usr/bin/env python3
"""Stage H — explicit supply pressure in central belt.

p_supply sweep × {smooth, textured} equilibrium.
Uses dirichlet_mask + g_bc from solver API.
p_supply=0 → None,None (no mask, baseline equivalence).
"""
from __future__ import annotations
import argparse, csv, datetime, json, math, os, sys, time
from typing import Any, Dict, List
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np

from models.feed_geometry import build_feed_mask, p_supply_to_g_bc
from cases.gu_loaded_side_v4.geometry_builders import build_relief
from cases.gu_loaded_side_v4.schema import SCHEMA, resolve_stage_dir
from cases.gu_loaded_side_v4.common import (
    D, R, L, c as C_CLEARANCE, n_rpm, eta, sigma, w_g,
)
from cases.gu_loaded_side_v4.run_herringbone_eq_continuation import (
    make_grid, eval_point, solve_equilibrium_nr, pack_g_init,
    PROTECTED_LO_DEG, PROTECTED_HI_DEG,
)

OMEGA = 2 * math.pi * n_rpm / 60.0


def main():
    pa = argparse.ArgumentParser(description="Stage H: supply pressure sweep")
    pa.add_argument("--stageA", required=True)
    pa.add_argument("--target-beta", type=float, default=45)
    pa.add_argument("--target-dg", type=float, default=15)
    pa.add_argument("--N-branch", type=int, default=10)
    pa.add_argument("--taper", type=float, default=0.6)
    pa.add_argument("--chirality", default="pump_to_edge")
    pa.add_argument("--coverage-mode", default="protect_loaded_union")
    pa.add_argument("--phi-bc", default="periodic",
                    choices=["periodic", "groove"])
    pa.add_argument("--p-supply-bar", type=float, nargs="+",
                    default=[0, 0.5, 2, 5])
    pa.add_argument("--phi-feed-deg", type=float, default=300)
    pa.add_argument("--phi-feed-half-deg", type=float, default=5)
    pa.add_argument("--z-belt-half", type=float, default=0.15)
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

    # Build feed mask (once — same for all p_supply > 0)
    feed_mask = build_feed_mask(
        Phi, Zm,
        phi_feed_deg=args.phi_feed_deg,
        phi_feed_half_deg=args.phi_feed_half_deg,
        z_belt_half=args.z_belt_half)
    n_masked = int(np.sum(feed_mask))
    print(f"Stage H: supply pressure sweep")
    print(f"Grid: {Np}x{Nz}, phi_bc={args.phi_bc}")
    print(f"Feed window: phi={args.phi_feed_deg}°±{args.phi_feed_half_deg}°, "
          f"|Z|<={args.z_belt_half}")
    print(f"Feed mask: {n_masked} cells ({n_masked/(Np*Nz)*100:.1f}%)")
    print(f"p_supply: {args.p_supply_bar} bar")
    print(f"Texture: beta={args.target_beta}° dg={args.target_dg}μm "
          f"N={args.N_branch} taper={args.taper}")

    # Build textured relief
    tex_relief = build_relief(
        Phi, Zm, variant="half_herringbone_ramped",
        depth_nondim=args.target_dg * 1e-6 / C_CLEARANCE,
        N_branch_per_side=args.N_branch,
        w_branch_nondim=w_g / R,
        belt_half_nondim=0.15,
        beta_deg=args.target_beta,
        chirality=args.chirality,
        ramp_frac=0.15,
        taper_ratio=args.taper,
        coverage_mode=args.coverage_mode,
        protected_lo_deg=PROTECTED_LO_DEG,
        protected_hi_deg=PROTECTED_HI_DEG)
    smooth_relief = np.zeros_like(Phi)

    rows: List[Dict] = []
    t_global = time.time()

    for lc_name in args.loadcases:
        lc = mA["loadcases"].get(lc_name)
        if not lc:
            continue
        Wa = np.array(lc["applied_load_N"], dtype=float)
        Wn = float(np.linalg.norm(Wa))
        Wd = Wa / max(Wn, 1e-20)
        eps_ref = float(lc["eps_source"])
        X_seed = -eps_ref * float(Wd[0])
        Y_seed = -eps_ref * float(Wd[1])

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wn:.1f}N  eps_ref={eps_ref}")

        for p_bar in args.p_supply_bar:
            p_Pa = float(p_bar) * 1e5
            if p_bar > 0:
                g_bc_val = p_supply_to_g_bc(p_Pa, eta, OMEGA, R, C_CLEARANCE)
                d_mask = feed_mask
                d_gbc = g_bc_val
            else:
                d_mask = None
                d_gbc = None
                g_bc_val = 0.0

            print(f"\n  p_supply = {p_bar} bar "
                  f"(g_bc = {g_bc_val:.6f})")

            for geo_tag, relief in [("smooth", smooth_relief),
                                     ("textured", tex_relief)]:
                t0 = time.time()
                d, g_out = solve_equilibrium_nr(
                    Wa, Phi, Zm, p1, z1, dp, dz,
                    relief, args.phi_bc, X_seed, Y_seed,
                    g_init_0=None,
                    dirichlet_mask=d_mask, g_bc=d_gbc)
                dt = time.time() - t0

                dg_hm = (args.target_dg * 1e-6 / max(d["h_min"], 1e-12)
                          if geo_tag == "textured" else 0.0)

                row = dict(
                    loadcase=lc_name,
                    geometry=geo_tag,
                    phi_bc=args.phi_bc,
                    p_supply_bar=p_bar,
                    g_bc=g_bc_val,
                    phi_feed_deg=args.phi_feed_deg,
                    phi_feed_half_deg=args.phi_feed_half_deg,
                    z_belt_half=args.z_belt_half,
                    eps=d["eps"],
                    X=d["X"], Y=d["Y"],
                    h_min_um=d["h_min"] * 1e6,
                    p_max_MPa=d["p_max"] / 1e6,
                    COF=d["COF"],
                    cav_frac=d["cav_frac"],
                    dg_over_hmin=dg_hm,
                    status=d["status"],
                    nr_rel_residual=d["rel_residual"],
                    elapsed_sec=dt)
                rows.append(row)

                print(f"    {geo_tag:>10s}: eps={d['eps']:.4f} "
                      f"COF={d['COF']:.6f} h={d['h_min']*1e6:.1f}μm "
                      f"res={d['rel_residual']:.1e} "
                      f"[{d['status']}] {dt:.0f}s")

    total = time.time() - t_global

    # Pairwise summary
    print(f"\n{'='*60}")
    print(f"Pairwise: textured vs smooth")
    pairs = []
    for p_bar in args.p_supply_bar:
        for lc_name in args.loadcases:
            sm = next((r for r in rows if r["geometry"] == "smooth"
                        and r["p_supply_bar"] == p_bar
                        and r["loadcase"] == lc_name), None)
            tx = next((r for r in rows if r["geometry"] == "textured"
                        and r["p_supply_bar"] == p_bar
                        and r["loadcase"] == lc_name), None)
            if sm and tx:
                comparable = (sm["status"] != "failed"
                               and tx["status"] != "failed")
                dCOF = ((tx["COF"] - sm["COF"]) / max(sm["COF"], 1e-20) * 100
                        if comparable else None)
                dh = ((tx["h_min_um"] - sm["h_min_um"])
                      / max(sm["h_min_um"], 1e-20) * 100
                      if comparable else None)
                pair = dict(
                    loadcase=lc_name, p_supply_bar=p_bar,
                    comparable=comparable,
                    smooth_status=sm["status"],
                    textured_status=tx["status"],
                    smooth_COF=sm["COF"], textured_COF=tx["COF"],
                    dCOF_eq_pct=dCOF,
                    smooth_hmin=sm["h_min_um"], textured_hmin=tx["h_min_um"],
                    dhmin_eq_pct=dh,
                    smooth_eps=sm["eps"], textured_eps=tx["eps"])
                pairs.append(pair)
                tag = "✓" if comparable else "✗"
                print(f"  [{tag}] {lc_name} p={p_bar}bar: "
                      f"ΔCOF={'n/a' if dCOF is None else f'{dCOF:+.1f}%':>7s} "
                      f"Δh={'n/a' if dh is None else f'{dh:+.1f}%':>7s} "
                      f"sm={sm['status'][:4]} tx={tx['status'][:4]}")

    # CSV
    if rows:
        cp = os.path.join(args.out, "stage_h_results.csv")
        flds = sorted(rows[0].keys())
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    if pairs:
        pp = os.path.join(args.out, "stage_h_pairwise.csv")
        pf = sorted(pairs[0].keys())
        with open(pp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pf)
            w.writeheader()
            for p in pairs:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, (int, float))
                                and not isinstance(v, bool) else v)
                            for k, v in p.items()})

    manifest = dict(
        schema_version=SCHEMA, stage="H_supply_pressure",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        phi_bc=args.phi_bc,
        p_supply_bar_list=args.p_supply_bar,
        feed_window=dict(phi_feed_deg=args.phi_feed_deg,
                          phi_feed_half_deg=args.phi_feed_half_deg,
                          z_belt_half=args.z_belt_half),
        n_masked_cells=n_masked,
        grid=dict(N_phi=Np, N_Z=Nz),
        texture=dict(beta=args.target_beta, dg=args.target_dg,
                      N=args.N_branch, taper=args.taper,
                      chirality=args.chirality),
        total_time_sec=total)
    with open(os.path.join(args.out, "stage_h_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Report
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Stage H — supply pressure sweep\n\n")
        f.write(f"- phi_bc: `{args.phi_bc}`\n")
        f.write(f"- feed window: φ={args.phi_feed_deg}°±"
                f"{args.phi_feed_half_deg}°, |Z|≤{args.z_belt_half}\n")
        f.write(f"- p_supply: {args.p_supply_bar} bar\n\n")
        f.write("## Convergence\n\n")
        f.write("| p_bar | geometry | status | COF | h_min μm | eps |\n")
        f.write("|-------|----------|--------|-----|----------|-----|\n")
        for r in rows:
            f.write(f"| {r['p_supply_bar']} | {r['geometry']} | "
                    f"{r['status']} | {r['COF']:.6f} | "
                    f"{r['h_min_um']:.1f} | {r['eps']:.4f} |\n")
        f.write("\n## Pairwise\n\n")
        f.write("| p_bar | comparable | ΔCOF% | Δh% |\n")
        f.write("|-------|-----------|-------|-----|\n")
        for p in pairs:
            dc = f"{p['dCOF_eq_pct']:+.1f}" if p["dCOF_eq_pct"] is not None else "n/a"
            dh = f"{p['dhmin_eq_pct']:+.1f}" if p["dhmin_eq_pct"] is not None else "n/a"
            f.write(f"| {p['p_supply_bar']} | {p['comparable']} | "
                    f"{dc} | {dh} |\n")

    print(f"\nTotal: {total:.0f}s")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
