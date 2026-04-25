#!/usr/bin/env python3
"""Stage G3.1 — coarse geometry rescue screen with smooth cache."""
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

from cases.gu_loaded_side_v4.geometry_builders import (
    build_relief, get_branch_centers, get_removed_branches,
)
from cases.gu_loaded_side_v4.schema import (
    SCHEMA, resolve_stage_dir, classify_status, TOL_HARD,
    PROTECTED_LO_DEG, PROTECTED_HI_DEG,
)
from cases.gu_loaded_side_v4.common import (
    D, R, L, c, n_rpm, eta, sigma, w_g,
    MAX_ITER_NR, STEP_CAP, EPS_MAX, HS_WARMUP_ITER, HS_WARMUP_TOL,
)

_ww: List[Dict] = []

def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2*math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1]-phi[0], Z[1]-Z[0]

def _ps(H, dp, dz, phi_bc, tag=""):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = _ps_fn(H, dp, dz, R, L, tol=1e-6, max_iter=10_000_000,
                   hs_warmup_iter=HS_WARMUP_ITER, hs_warmup_tol=HS_WARMUP_TOL,
                   phi_bc=phi_bc)
    import re
    for w in caught:
        m = str(w.message)
        if "warmup" in m.lower():
            mx = re.search(r"res=([0-9.e+-]+)", m)
            _ww.append(dict(tag=tag, res=float(mx.group(1)) if mx else None))
    return r

def _ev(X, Y, Phi, Zm, p1, z1, dp, dz, rel, om, ps, pbc, tag=""):
    H0 = 1.0 + float(X)*np.cos(Phi) + float(Y)*np.sin(Phi)
    if sigma > 0:
        H0 = np.sqrt(H0**2 + (sigma/c)**2)
    H = H0 + rel
    P, th, _, _ = _ps(H, dp, dz, pbc, tag)
    Pd = P * ps
    Fx = -np.trapezoid(np.trapezoid(Pd*np.cos(Phi), p1, axis=1), z1, axis=0)*R*L/2
    Fy = -np.trapezoid(np.trapezoid(Pd*np.sin(Phi), p1, axis=1), z1, axis=0)*R*L/2
    hd = H * c
    hm = float(np.min(hd))
    pm = float(np.max(Pd))
    cv = float(np.mean(th < 1-1e-6))
    tc = eta*om*R/hd
    dPdp = np.gradient(Pd, dp, axis=1)
    tp = hd/2*dPdp/R
    ta = tc + tp
    fr = float(np.trapezoid(np.trapezoid(np.abs(ta), p1, axis=1), z1, axis=0)*R*L/2)
    return float(Fx), float(Fy), hm, pm, cv, fr

def _solve(Wa, Phi, Zm, p1, z1, dp, dz, rel, pbc, X0, Y0, tag=""):
    om = 2*math.pi*n_rpm/60; ps = 6*eta*om*(R/c)**2
    Wn = float(np.linalg.norm(Wa)); dXY = 1e-4
    X, Y = float(X0), float(Y0)
    Fx, Fy, hm, pm, cv, fr = _ev(X,Y,Phi,Zm,p1,z1,dp,dz,rel,om,ps,pbc,tag)
    rR = math.sqrt((Fx-Wa[0])**2+(Fy-Wa[1])**2)/max(Wn,1e-20)
    for _ in range(MAX_ITER_NR):
        if rR < TOL_HARD: break
        J = np.zeros((2,2))
        for col,(dX_,dY_) in enumerate([(dXY,0),(0,dXY)]):
            Fxp,Fyp,*_ = _ev(X+dX_,Y+dY_,Phi,Zm,p1,z1,dp,dz,rel,om,ps,pbc,tag)
            Fxn,Fyn,*_ = _ev(X-dX_,Y-dY_,Phi,Zm,p1,z1,dp,dz,rel,om,ps,pbc,tag)
            J[0,col]=(Fxp-Fxn)/(2*dXY); J[1,col]=(Fyp-Fyn)/(2*dXY)
        Rx=Fx-Wa[0]; Ry=Fy-Wa[1]
        det=J[0,0]*J[1,1]-J[0,1]*J[1,0]
        if abs(det)<1e-30: break
        dX=-(J[1,1]*Rx-J[0,1]*Ry)/det; dY=-(-J[1,0]*Rx+J[0,0]*Ry)/det
        cf=STEP_CAP/max(abs(dX),abs(dY),1e-20)
        if cf<1: dX*=cf; dY*=cf
        ok=False
        for a in [1,.5,.25,.125]:
            Xt=X+a*dX; Yt=Y+a*dY
            if math.sqrt(Xt**2+Yt**2)>=EPS_MAX: continue
            Fxt,Fyt,ht,pt,ct,ft=_ev(Xt,Yt,Phi,Zm,p1,z1,dp,dz,rel,om,ps,pbc,tag)
            rt=math.sqrt((Fxt-Wa[0])**2+(Fyt-Wa[1])**2)/max(Wn,1e-20)
            if rt<rR:
                X,Y=Xt,Yt; Fx,Fy=Fxt,Fyt; hm,pm,cv,fr=ht,pt,ct,ft; rR=rt; ok=True; break
        if not ok: break
    eps=math.sqrt(X**2+Y**2)
    COF=fr/max(Wn,1e-20)
    st=classify_status(rR, rR<=0.10)
    return dict(eps=eps,h_min=hm,p_max=pm,cav_frac=cv,friction=fr,
                COF=COF,rel_residual=rR,status=st)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--stageA", required=True)
    pa.add_argument("--variants", nargs="+", default=["straight_ramped","arc_ramped"])
    pa.add_argument("--N-branch-list", type=int, nargs="+", default=[5,6,8,10])
    pa.add_argument("--dg-list", type=float, nargs="+", default=[10,25,50])
    pa.add_argument("--belt-list", type=float, nargs="+", default=[0.15])
    pa.add_argument("--taper-list", type=float, nargs="+", default=[1.0,0.6])
    pa.add_argument("--curvature-k-list", type=float, nargs="+", default=[0.15,0.30])
    pa.add_argument("--beta-list", type=float, nargs="+", default=[20])
    pa.add_argument("--coverage-mode", default="protect_loaded_union")
    pa.add_argument("--feed-mode", default="groove")
    pa.add_argument("--eps-list", type=float, nargs="+", default=[0.5,0.8])
    pa.add_argument("--grid", default="800x240")
    pa.add_argument("--retry-failed", action="store_true")
    pa.add_argument("--out", required=True)
    args = pa.parse_args()

    Np,Nz=(int(x) for x in args.grid.split("x"))
    p1,z1,Phi,Zm,dp,dz=make_grid(Np,Nz)
    os.makedirs(args.out, exist_ok=True)

    stA=resolve_stage_dir(args.stageA)
    with open(os.path.join(stA,"working_geometry_manifest.json")) as f:
        mA=json.load(f)
    anchors={}
    for lc,v in mA.get("loadcases",{}).items():
        anchors[float(v["eps_source"])]=dict(W=v["applied_load_N"],
            phi_loaded=v.get("phi_loaded_deg",140))

    print(f"Stage G3.1: geometry rescue screen")
    print(f"Grid: {Np}x{Nz}, feed={args.feed_mode}, coverage={args.coverage_mode}")
    print(f"Variants: {args.variants}, N_branch: {args.N_branch_list}")

    _ww.clear()
    smooth_cache: Dict[Tuple, Dict] = {}
    rows: List[Dict] = []
    t0g=time.time()

    for var in args.variants:
        for Nb in args.N_branch_list:
            # Branch info
            centers_rad = get_branch_centers(Nb)
            centers_deg = [float(math.degrees(c)) for c in centers_rad]
            removed = get_removed_branches(centers_deg, PROTECTED_LO_DEG, PROTECTED_HI_DEG)
            n_removed = len(removed) if args.coverage_mode != "full_360" else 0
            active_count = Nb - n_removed

            for dg in args.dg_list:
                taper_list = args.taper_list
                curv_list = args.curvature_k_list if var == "arc_ramped" else [0.0]
                beta_list = args.beta_list if "herringbone" in var else [0.0]

                for taper in taper_list:
                    for curv in curv_list:
                        for beta in beta_list:
                            for belt in args.belt_list:
                                relief = build_relief(
                                    Phi, Zm, variant=var,
                                    depth_nondim=dg*1e-6/c,
                                    N_branch_per_side=Nb,
                                    w_branch_nondim=w_g/R,
                                    belt_half_nondim=belt,
                                    beta_deg=beta, ramp_frac=0.15,
                                    taper_ratio=taper,
                                    apex_radius_frac=0.5,
                                    curvature_k=curv,
                                    coverage_mode=args.coverage_mode,
                                    protected_lo_deg=PROTECTED_LO_DEG,
                                    protected_hi_deg=PROTECTED_HI_DEG)
                                zero_rel = np.zeros_like(Phi)

                                for eps in args.eps_list:
                                    anc = anchors.get(eps, {})
                                    Wa = np.array(anc.get("W",[0,-100]),dtype=float)
                                    Wn = float(np.linalg.norm(Wa))
                                    Wd = Wa/max(Wn,1e-20)
                                    X0 = -eps*float(Wd[0])
                                    Y0 = -eps*float(Wd[1])

                                    # Smooth cache
                                    sk = (eps, args.feed_mode, belt, args.grid)
                                    if sk not in smooth_cache:
                                        t0=time.time()
                                        ds=_solve(Wa,Phi,Zm,p1,z1,dp,dz,
                                                  zero_rel,args.feed_mode,X0,Y0,
                                                  f"smooth_eps{eps}")
                                        dt=time.time()-t0
                                        ds["config"]="smooth"
                                        ds["elapsed_sec"]=dt
                                        smooth_cache[sk]=ds
                                        print(f"  [smooth] eps={eps} COF={ds['COF']:.6f} "
                                              f"h={ds['h_min']*1e6:.1f}μm [{ds['status']}] {dt:.0f}s")
                                    else:
                                        ds = smooth_cache[sk]
                                        print(f"  [cache hit] smooth eps={eps}")

                                    # Veined
                                    tag=f"{var}_N{Nb}_dg{dg}_t{taper}_k{curv}_b{beta}_eps{eps}"
                                    t0=time.time()
                                    dv=_solve(Wa,Phi,Zm,p1,z1,dp,dz,
                                              relief,args.feed_mode,X0,Y0,tag)
                                    dt=time.time()-t0

                                    dg_hm=(dg*1e-6)/max(dv["h_min"],1e-12)
                                    dCOF=(dv["COF"]-ds["COF"])/max(ds["COF"],1e-20)*100
                                    dh=(dv["h_min"]-ds["h_min"])/max(ds["h_min"],1e-20)*100*1e6/(1e6)  # fix
                                    dh_pct=(dv["h_min"]*1e6-ds["h_min"]*1e6)/max(ds["h_min"]*1e6,1e-20)*100
                                    dp_pct=(dv["p_max"]/1e6-ds["p_max"]/1e6)/max(ds["p_max"]/1e6,1e-20)*100

                                    row=dict(
                                        variant=var,N_branch=Nb,d_g_um=dg,
                                        belt=belt,taper_ratio=taper,
                                        curvature_k=curv,beta_deg=beta,
                                        coverage_mode=args.coverage_mode,
                                        feed_mode=args.feed_mode,
                                        eps_ref=eps,
                                        n_removed_branches=n_removed,
                                        active_branches=active_count,
                                        eps_eq=dv["eps"],
                                        h_min_um=dv["h_min"]*1e6,
                                        p_max_MPa=dv["p_max"]/1e6,
                                        COF_veined=dv["COF"],
                                        COF_smooth=ds["COF"],
                                        dCOF_pct=dCOF,
                                        dh_pct=dh_pct,
                                        dp_pct=dp_pct,
                                        dg_over_hmin=dg_hm,
                                        status=dv["status"],
                                        rel_residual=dv["rel_residual"],
                                        elapsed_sec=dt)
                                    rows.append(row)

                                    st="✓" if dv["status"] in ("hard_converged","soft_converged") else "✗"
                                    print(f"  [{st}] {tag[:55]:55s} "
                                          f"ΔCOF={dCOF:+.1f}% Δh={dh_pct:+.1f}% "
                                          f"dg/h={dg_hm:.2f} [{dv['status']}] {dt:.0f}s")

    total=time.time()-t0g

    # Gate check
    accepted=[r for r in rows if r["status"] in ("hard_converged","soft_converged")]
    weak=any(r["dCOF_pct"]<=-3 and r["dh_pct"]>=-2 for r in accepted)
    strong=any(r["dCOF_pct"]<=-5 and r["dh_pct"]>=0 for r in accepted)
    verdict="STRONG_PASS" if strong else ("WEAK_PASS" if weak else "NO_GO")
    print(f"\nGate verdict: {verdict} ({len(accepted)}/{len(rows)} accepted)")

    # CSV
    if rows:
        cp=os.path.join(args.out,"geometry_rescue_results.csv")
        flds=sorted(rows[0].keys())
        with open(cp,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=flds); w.writeheader()
            for r in rows:
                w.writerow({k:(f"{v:.8e}" if isinstance(v,float) else v) for k,v in r.items()})
        # pairs csv (same as results — each row already has smooth ref)
        pp=os.path.join(args.out,"geometry_rescue_pairs.csv")
        with open(pp,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=flds); w.writeheader()
            for r in rows:
                w.writerow({k:(f"{v:.8e}" if isinstance(v,float) else v) for k,v in r.items()})

    wc=len(_ww)
    wm=max((w["res"] for w in _ww if w["res"]),default=0)
    manifest=dict(
        schema_version=SCHEMA, stage="G3_screen",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        variants=args.variants, N_branch_list=args.N_branch_list,
        dg_list=args.dg_list, belt_list=args.belt_list,
        taper_list=args.taper_list, curvature_k_list=args.curvature_k_list,
        beta_list=args.beta_list,
        coverage_mode=args.coverage_mode, feed_mode=args.feed_mode,
        eps_list=args.eps_list, grid=dict(N_phi=Np,N_Z=Nz),
        protected_zone_deg=[PROTECTED_LO_DEG,PROTECTED_HI_DEG],
        gate_verdict=verdict, n_solves=len(rows),
        n_smooth_cached=len(smooth_cache),
        total_time_sec=total,
        hs_warmup_warnings=dict(count=wc,max_residual=wm))
    with open(os.path.join(args.out,"geometry_rescue_manifest.json"),"w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False)

    print(f"\nTotal: {total:.0f}s ({len(rows)} veined + {len(smooth_cache)} smooth cached)")
    print(f"HS warmup warnings: {wc}")
    print(f"Artifacts: {args.out}/")

if __name__=="__main__":
    main()
