#!/usr/bin/env python3
"""Stage I-A — cold quasi-static diesel cycle (smooth vs textured).

For each crank angle: solve stationary equilibrium with warm-start.
No squeeze, no thermal. Shell-fixed placement.
"""
from __future__ import annotations

import argparse, csv, datetime, json, math, os, sys, time, warnings
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu as _ps_fn
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu as _ps_fn

from models.feed_geometry import build_feed_mask_variant, p_supply_to_g_bc
from cases.gu_loaded_side_v4.geometry_builders import build_relief
from config.diesel_stage1_case import (
    D, R, L, c, n_rpm, ETA_COLD,
    FEED_VARIANT, FEED_PHI_HALF_DEG, FEED_Z_BELT_HALF, P_SUPPLY_PA,
    TEX_VARIANT, TEX_CHIRALITY, TEX_N_BRANCH, TEX_BETA_DEG,
    TEX_DG_M, TEX_TAPER, TEX_BELT_HALF, TEX_RAMP_FRAC,
    GRID_COARSE, GRID_CONFIRM,
    MAX_ITER_NR, STEP_CAP, EPS_MAX, TOL_HARD, TOL_SOFT,
    CYCLE_N_COARSE, CYCLE_N_FINE,
    PLACEMENT_MODE, PHI_REF_DEG,
)
from models.diesel_stage1_cycle import (
    build_surrogate_heavyduty_v1,
    get_load_at_angle,
    get_placement_from_reference,
    quasistatic_ratio,
)
from models.continuation_runner import (
    ContinuationConfig,
    run_continuation_cycle,
)
from models.anchor_solver import (
    PS_BUDGETS,
    AnchorState,
    AnchorReport,
    DEFAULT_EXPLICIT_PHI_ANCHOR_DEG,
    pick_anchor_phi,
    solve_anchor_smooth,
    solve_anchor_textured,
)

OMEGA = 2 * math.pi * n_rpm / 60.0
P_SCALE = 6.0 * ETA_COLD * OMEGA * (R / c) ** 2
SCHEMA = "diesel_stage1_v1"

PROTECTED_LO = 105.0
PROTECTED_HI = 175.0


def make_grid(Np, Nz):
    phi = np.linspace(0, 2*math.pi, Np, endpoint=False)
    Z = np.linspace(-1, 1, Nz)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, phi[1]-phi[0], Z[1]-Z[0]


def pack_g(P, theta):
    g = np.where(P > 1e-14, P, theta - 1.0)
    return np.ascontiguousarray(g, dtype=np.float64)


def _ps(H, dp, dz, phi_bc, g_init=None, d_mask=None, g_bc=None,
        max_iter=50_000, hs_warmup_iter=20_000):
    kw = dict(tol=1e-5, max_iter=max_iter,
              hs_warmup_iter=hs_warmup_iter, hs_warmup_tol=1e-5,
              phi_bc=phi_bc)
    if g_init is not None:
        kw["g_init"] = g_init
    if d_mask is not None and g_bc is not None:
        kw["dirichlet_mask"] = d_mask
        kw["g_bc"] = g_bc
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return _ps_fn(H, dp, dz, R, L, **kw)


def eval_pt(X, Y, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
            g_init=None, d_mask=None, g_bc=None,
            ps_max_iter=50_000, hs_warmup_iter=20_000):
    H0 = 1.0 + float(X)*np.cos(Phi) + float(Y)*np.sin(Phi)
    H = H0 + relief
    P, theta, ps_res, ps_it = _ps(H, dp, dz, phi_bc, g_init, d_mask, g_bc,
                                    max_iter=ps_max_iter,
                                    hs_warmup_iter=hs_warmup_iter)
    Pd = P * P_SCALE
    Fx = -np.trapezoid(np.trapezoid(Pd*np.cos(Phi), p1, axis=1),
                        z1, axis=0)*R*L/2
    Fy = -np.trapezoid(np.trapezoid(Pd*np.sin(Phi), p1, axis=1),
                        z1, axis=0)*R*L/2
    hd = H * c
    hm = float(np.min(hd))
    pm = float(np.max(Pd))
    cv = float(np.mean(theta < 1-1e-6))
    tc = ETA_COLD*OMEGA*R/hd
    dPdp = np.gradient(Pd, dp, axis=1)
    tp = hd/2*dPdp/R
    ta = tc + tp
    fr = float(np.trapezoid(np.trapezoid(np.abs(ta), p1, axis=1),
                              z1, axis=0)*R*L/2)
    Ploss = fr * OMEGA * R
    # Side leakage (rough estimate from axial P gradient at Z boundaries)
    dPdZ = np.gradient(Pd, dz, axis=0)
    q_top = hd[-1, :]**3 / (12*ETA_COLD) * np.abs(dPdZ[-1, :]) * R
    q_bot = hd[0, :]**3 / (12*ETA_COLD) * np.abs(dPdZ[0, :]) * R
    Qout = float(np.trapezoid(q_top, p1) + np.trapezoid(q_bot, p1))
    return dict(Fx=float(Fx), Fy=float(Fy), h_min=hm, p_max=pm,
                cav_frac=cv, friction=fr, Ploss=Ploss, Qout=Qout,
                ps_iters=int(ps_it) if ps_it else 0,
                ps_res=float(ps_res) if ps_res else 0), P, theta


def solve_eq(Wa, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
             X0, Y0, g_init=None, d_mask=None, g_bc=None):
    Wn = float(np.linalg.norm(Wa))
    dXY = 1e-4
    X, Y = float(X0), float(Y0)
    g_c = g_init
    m, P, th = eval_pt(X, Y, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
                        g_c, d_mask, g_bc)
    g_c = pack_g(P, th)
    Rx = m["Fx"]-Wa[0]; Ry = m["Fy"]-Wa[1]
    rR = math.sqrt(Rx**2+Ry**2)/max(Wn,1e-20)
    Fxh, Fyh = m["Fx"], m["Fy"]
    hm, pm, cv, fr, pl, qo = m["h_min"], m["p_max"], m["cav_frac"], m["friction"], m["Ploss"], m["Qout"]
    nr_it = 0
    for _ in range(MAX_ITER_NR):
        if rR < TOL_HARD:
            break
        J = np.zeros((2,2))
        for col,(dX_,dY_) in enumerate([(dXY,0),(0,dXY)]):
            mp,_,_ = eval_pt(X+dX_,Y+dY_,Phi,Zm,p1,z1,dp,dz,relief,phi_bc,g_c,d_mask,g_bc)
            mn,_,_ = eval_pt(X-dX_,Y-dY_,Phi,Zm,p1,z1,dp,dz,relief,phi_bc,g_c,d_mask,g_bc)
            J[0,col]=(mp["Fx"]-mn["Fx"])/(2*dXY)
            J[1,col]=(mp["Fy"]-mn["Fy"])/(2*dXY)
        Rx=Fxh-Wa[0]; Ry=Fyh-Wa[1]
        det=J[0,0]*J[1,1]-J[0,1]*J[1,0]
        if abs(det)<1e-30: break
        ddX=-(J[1,1]*Rx-J[0,1]*Ry)/det
        ddY=-(-J[1,0]*Rx+J[0,0]*Ry)/det
        cf=STEP_CAP/max(abs(ddX),abs(ddY),1e-20)
        if cf<1: ddX*=cf; ddY*=cf
        ok=False
        for alpha in [1,.5,.25,.125]:
            Xt=X+alpha*ddX; Yt=Y+alpha*ddY
            if math.sqrt(Xt**2+Yt**2)>=EPS_MAX: continue
            mt,Pt,tht=eval_pt(Xt,Yt,Phi,Zm,p1,z1,dp,dz,relief,phi_bc,g_c,d_mask,g_bc)
            rt=math.sqrt((mt["Fx"]-Wa[0])**2+(mt["Fy"]-Wa[1])**2)/max(Wn,1e-20)
            if rt<rR:
                X,Y=Xt,Yt; Fxh,Fyh=mt["Fx"],mt["Fy"]
                hm,pm,cv,fr,pl,qo=mt["h_min"],mt["p_max"],mt["cav_frac"],mt["friction"],mt["Ploss"],mt["Qout"]
                rR=rt; g_c=pack_g(Pt,tht); ok=True; nr_it+=1; break
        if not ok: break
    eps=math.sqrt(X**2+Y**2)
    att=math.degrees(math.atan2(Y,X))
    st="hard_converged" if rR<=TOL_HARD else ("soft_converged" if rR<=TOL_SOFT else "failed")
    return dict(X=X,Y=Y,eps=eps,attitude_deg=att,h_min=hm,p_max=pm,
                cav_frac=cv,friction=fr,Ploss=pl,Qout=qo,
                rel_residual=rR,status=st,nr_iters=nr_it), g_c


def _run_independent_nr(phi_arr, cycle, Phi, Zm, p1, z1, dp, dz, phi_bc,
                         smooth_relief, tex_relief, feed_mask, g_bc_val, p_Pa):
    all_rows: List[Dict] = []
    for geo_tag, relief in [("smooth", smooth_relief), ("textured", tex_relief)]:
        X_prev, Y_prev = 0.0, -0.4
        g_prev = None
        W_prev = 0.0
        print(f"\n{'='*60}")
        print(f"  {geo_tag.upper()}")

        for i, phi_deg in enumerate(phi_arr):
            Wx, Wy = get_load_at_angle(cycle, phi_deg)
            Wa = np.array([Wx, Wy], dtype=float)
            Wn = float(np.linalg.norm(Wa))

            load_jump = abs(Wn - W_prev) / max(W_prev, 100.0)
            if load_jump > 0.5 or i == 0:
                eps_seed = min(0.85, 0.4 * (Wn / 1000.0) ** 0.25)
                X_prev, Y_prev = 0.0, -eps_seed
                g_prev = None

            t0 = time.time()
            d, g_out = solve_eq(
                Wa, Phi, Zm, p1, z1, dp, dz, relief, phi_bc,
                X_prev, Y_prev, g_prev, feed_mask,
                g_bc_val if p_Pa > 0 else None)
            dt = time.time() - t0

            if d["status"] != "failed":
                X_prev, Y_prev = d["X"], d["Y"]
                g_prev = g_out
            W_prev = Wn

            row = dict(
                geometry=geo_tag,
                phi_crank_deg=float(phi_deg),
                Wx_N=Wx, Wy_N=Wy, W_N=Wn,
                eps=d["eps"], attitude_deg=d["attitude_deg"],
                h_min_um=d["h_min"]*1e6,
                p_max_MPa=d["p_max"]/1e6,
                Ploss_W=d["Ploss"],
                Qout_m3s=d["Qout"],
                friction=d["friction"],
                cav_frac=d["cav_frac"],
                status=d["status"],
                rel_residual=d["rel_residual"],
                nr_iters=d["nr_iters"],
                elapsed_sec=dt)
            all_rows.append(row)

            st = "✓" if d["status"] in ("hard_converged","soft_converged") else "✗"
            print(f"  [{st}] φ={phi_deg:6.1f}° W={Wn:7.0f}N "
                  f"ε={d['eps']:.4f} h={d['h_min']*1e6:.1f}μm "
                  f"P={d['Ploss']:.1f}W "
                  f"res={d['rel_residual']:.1e} [{d['status'][:4]}] {dt:.0f}s")
    return all_rows


def _make_eval_factory(Phi, Zm, p1, z1, dp, dz, phi_bc,
                         relief, d_mask, g_bc):
    """Mode-aware eval factory: ``factory(mode_str)`` -> eval_fn(X,Y,g)."""
    def factory(mode="accepted_node"):
        b = PS_BUDGETS.get(mode, PS_BUDGETS["accepted_node"])
        def _eval(X, Y, g_init, _b=b):
            return eval_pt(X, Y, Phi, Zm, p1, z1, dp, dz,
                           relief, phi_bc, g_init, d_mask, g_bc,
                           ps_max_iter=_b["ps_max_iter"],
                           hs_warmup_iter=_b["hs_warmup_iter"])
        return _eval
    return factory


def _make_textured_geometry_factory(Phi, Zm, p1, z1, dp, dz, phi_bc,
                                      base_tex_relief, d_mask, g_bc):
    """Returns ``geometry_factory(alpha_tex, mode)`` -> eval_fn.

    Texture amplitude is scaled by alpha in [0, 1]; only the relief field
    changes (Section 4.2: keep all other parameters fixed).
    """
    def make(alpha_tex: float, mode: str = "anchor_stage_first"):
        rel = float(alpha_tex) * base_tex_relief
        b = PS_BUDGETS.get(mode, PS_BUDGETS["anchor_stage_first"])
        def _eval(X, Y, g_init, _b=b, _rel=rel):
            return eval_pt(X, Y, Phi, Zm, p1, z1, dp, dz,
                           _rel, phi_bc, g_init, d_mask, g_bc,
                           ps_max_iter=_b["ps_max_iter"],
                           hs_warmup_iter=_b["hs_warmup_iter"])
        return _eval
    return make


def _log_anchor_report(rep: AnchorReport, geo_tag: str):
    print(f"  [anchor:{geo_tag}] mode={rep.mode} "
          f"phi_anchor={rep.phi_anchor_deg:.1f}° "
          f"reason: {rep.selection_reason}")
    if rep.candidates:
        cs = ", ".join(f"{c:.0f}" for c in rep.candidates)
        print(f"  [anchor:{geo_tag}] candidates: [{cs}]")
    for e in rep.smooth_lambda_log:
        print(f"  [anchor:{geo_tag}:smooth] lam={e['lambda_']:.2f} "
              f"sched={e['schedule']} st={e['status'][:4]} "
              f"nr={e['nr_iters']} eps={e['eps']:.3f} "
              f"res={e['rel_residual']:.1e} t={e['wall_sec']:.1f}s")
    for e in rep.textured_log:
        path = e.get("path", "?")
        if "alpha_tex" in e and path == "geometry_continuation":
            tag = f"alpha={e['alpha_tex']:.2f}"
        else:
            tag = path
        print(f"  [anchor:{geo_tag}:textured] {tag} "
              f"st={e['status'][:4]} nr={e['nr_iters']} "
              f"eps={e['eps']:.3f} res={e['rel_residual']:.1e} "
              f"t={e['wall_sec']:.1f}s")


def _node_to_row(nd, geo_tag, load_fn, dt_total, n_nodes):
    Wx, Wy = load_fn(nd.phi_deg)
    Wn = math.sqrt(Wx**2 + Wy**2)
    return dict(
        geometry=geo_tag,
        phi_crank_deg=nd.phi_deg,
        Wx_N=Wx, Wy_N=Wy, W_N=Wn,
        eps=nd.eps, attitude_deg=nd.attitude_deg,
        h_min_um=nd.h_min*1e6,
        p_max_MPa=nd.p_max/1e6,
        Ploss_W=nd.Ploss,
        Qout_m3s=nd.Qout,
        friction=nd.friction,
        cav_frac=nd.cav_frac,
        status=nd.status,
        rel_residual=nd.rel_residual,
        nr_iters=nd.nr_iters,
        elapsed_sec=dt_total / max(n_nodes, 1))


def _run_continuation(phi_arr, cycle, Phi, Zm, p1, z1, dp, dz, phi_bc,
                       smooth_relief, tex_relief, feed_mask, g_bc_val, p_Pa,
                       load_fn,
                       *,
                       anchor_mode: str = "explicit",
                       phi_anchor_deg: Optional[float] = None,
                       legacy_matched_phis: Optional[List[float]] = None,
                       out_dir: Optional[str] = None):
    """Anchor-first continuation runner.

    Stage I-A patch policy (see ANCHOR_PATCH_NOTE.md):
      1. Pick anchor angle in a medium-load, well-conditioned sector
         (NEVER global min/max load).
      2. Solve smooth anchor via load homotopy in lambda at fixed phi_a.
      3. Solve textured anchor seeded from accepted smooth state; rescue
         via geometry continuation in alpha_tex if direct solve fails.
      4. Only after accepted anchor — march the regular continuation_phi
         cycle from the anchor outward.
    """
    all_rows: List[Dict] = []
    phi_list = phi_arr.tolist()

    d_mask = feed_mask
    g_bc = g_bc_val if p_Pa > 0 else None

    smooth_factory = _make_eval_factory(
        Phi, Zm, p1, z1, dp, dz, phi_bc, smooth_relief, d_mask, g_bc)
    tex_factory = _make_eval_factory(
        Phi, Zm, p1, z1, dp, dz, phi_bc, tex_relief, d_mask, g_bc)
    tex_geom_factory = _make_textured_geometry_factory(
        Phi, Zm, p1, z1, dp, dz, phi_bc, tex_relief, d_mask, g_bc)

    # ── 1. Anchor selection ───────────────────────────────────────
    explicit_phi = (phi_anchor_deg if phi_anchor_deg is not None
                    else DEFAULT_EXPLICIT_PHI_ANCHOR_DEG)
    phi_anchor, reason, cand_list = pick_anchor_phi(
        phi_list, load_fn,
        mode=anchor_mode,
        explicit_phi_deg=explicit_phi,
        legacy_matched_phis=legacy_matched_phis,
        scout_eval_fn=(smooth_factory("scout") if anchor_mode == "scout_best"
                        else None),
    )
    Wx_a, Wy_a = load_fn(phi_anchor)
    Wa_anchor = np.array([Wx_a, Wy_a], dtype=float)
    Wn_a = float(np.linalg.norm(Wa_anchor))
    print(f"\n{'='*60}")
    print(f"  ANCHOR POLICY: {anchor_mode} "
          f"-> phi_anchor={phi_anchor:.1f}° |W|={Wn_a:.0f}N")
    print(f"  reason: {reason}")
    if cand_list:
        cs = ", ".join(f"{c:.0f}" for c in cand_list)
        print(f"  candidates: [{cs}]")

    # ── 2. Smooth anchor (load homotopy at fixed phi_a) ──────────
    print(f"\n  SMOOTH ANCHOR @ phi={phi_anchor:.1f}°")
    t0 = time.time()
    # Anchor uses its own (looser) soft_tol = 5e-2 by default: anchor is
    # a landing entry point, not a production residual. Step_cap is also
    # the anchor-side default (0.20), not the in-branch STEP_CAP (0.05).
    smooth_anchor, smooth_log = solve_anchor_smooth(
        phi_anchor, Wa_anchor, smooth_factory,
        eps_max=EPS_MAX,
        tol=TOL_HARD,
    )
    smooth_anchor_dt = time.time() - t0

    rep_sm = AnchorReport(
        mode=anchor_mode, phi_anchor_deg=phi_anchor,
        candidates=cand_list, selection_reason=reason,
        smooth_lambda_log=smooth_log,
        smooth_state=smooth_anchor,
        smooth_ok=smooth_anchor is not None,
        wall_total_sec=smooth_anchor_dt,
    )
    _log_anchor_report(rep_sm, "smooth")
    if smooth_anchor is None:
        print(f"  [anchor:smooth] FAILED after {smooth_anchor_dt:.1f}s "
              f"— aborting continuation")
        if out_dir is not None:
            _dump_anchor_log(out_dir, rep_sm, None)
        return all_rows
    print(f"  [anchor:smooth] ✓ accepted in {smooth_anchor_dt:.1f}s "
          f"eps={smooth_anchor.eps:.4f} h={smooth_anchor.h_min*1e6:.1f}μm "
          f"res={smooth_anchor.rel_residual:.1e}")

    # ── 3. Textured anchor (smooth-seeded, optional geom continuation) ──
    print(f"\n  TEXTURED ANCHOR @ phi={phi_anchor:.1f}°")
    t0 = time.time()
    tex_anchor, tex_log = solve_anchor_textured(
        phi_anchor, Wa_anchor, tex_factory, smooth_anchor,
        geometry_factory=tex_geom_factory,
        eps_max=EPS_MAX,
        tol=TOL_HARD,
    )
    tex_anchor_dt = time.time() - t0
    rep_tx = AnchorReport(
        mode=anchor_mode, phi_anchor_deg=phi_anchor,
        candidates=cand_list, selection_reason=reason,
        textured_log=tex_log,
        textured_state=tex_anchor,
        textured_ok=tex_anchor is not None,
        wall_total_sec=tex_anchor_dt,
    )
    _log_anchor_report(rep_tx, "textured")
    if tex_anchor is None:
        print(f"  [anchor:textured] FAILED after {tex_anchor_dt:.1f}s "
              f"— textured branch will be skipped")
    else:
        print(f"  [anchor:textured] ✓ accepted in {tex_anchor_dt:.1f}s "
              f"eps={tex_anchor.eps:.4f} h={tex_anchor.h_min*1e6:.1f}μm "
              f"res={tex_anchor.rel_residual:.1e}")

    if out_dir is not None:
        _dump_anchor_log(out_dir, rep_sm, rep_tx)

    # ── 4. Cycle march from anchor (existing continuation_phi logic) ──
    cfg = ContinuationConfig(
        corrector_max_iter=10,
        corrector_tol=TOL_HARD,
        corrector_soft_tol=TOL_SOFT,
        step_cap=STEP_CAP,
        eps_max=EPS_MAX,
        max_subdiv_depth=4,
        min_dphi_deg=1.5)

    geo_tasks = [("smooth", smooth_factory, smooth_anchor)]
    if tex_anchor is not None:
        geo_tasks.append(("textured", tex_factory, tex_anchor))

    for geo_tag, factory, anchor_state in geo_tasks:
        print(f"\n{'='*60}")
        print(f"  {geo_tag.upper()} (continuation_phi from anchor "
              f"@ {anchor_state.phi_deg:.1f}°)", flush=True)

        # Streaming progress callback: print as soon as each node is done,
        # so the user can see actual progress instead of one big silent
        # block until the whole cycle finishes.
        t_start = [time.time()]
        def _on_node(nd, idx, n_total, _ts=t_start, _gt=geo_tag):
            now = time.time()
            dt_node = now - _ts[0]
            _ts[0] = now
            st = "✓" if nd.status in ("hard_converged","soft_converged") else "✗"
            Wx_n, Wy_n = load_fn(nd.phi_deg)
            Wn_n = math.sqrt(Wx_n**2 + Wy_n**2)
            print(f"  [{st}] {idx+1:3d}/{n_total} φ={nd.phi_deg:6.1f}° "
                  f"W={Wn_n:7.0f}N ε={nd.eps:.4f} "
                  f"h={nd.h_min*1e6:.1f}μm P={nd.Ploss:.1f}W "
                  f"res={nd.rel_residual:.1e} [{nd.status[:4]}] "
                  f"pred={nd.predictor_type} d={nd.subdiv_depth} "
                  f"{dt_node:.0f}s",
                  flush=True)

        t0 = time.time()
        nodes = run_continuation_cycle(
            phi_list, load_fn, factory, cfg,
            phi_anchor_deg=anchor_state.phi_deg,
            anchor_state=anchor_state,
            node_callback=_on_node)
        dt_total = time.time() - t0

        for nd in nodes:
            row = _node_to_row(nd, geo_tag, load_fn, dt_total, len(nodes))
            all_rows.append(row)

        n_acc = sum(1 for nd in nodes if nd.status != "failed")
        print(f"  {geo_tag}: {n_acc}/{len(nodes)} converged "
              f"({n_acc/max(len(nodes),1)*100:.0f}%) in {dt_total:.0f}s",
              flush=True)

    return all_rows


def _dump_anchor_log(out_dir: str,
                       rep_sm: Optional[AnchorReport],
                       rep_tx: Optional[AnchorReport]):
    """Write anchor.json next to the cycle outputs."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    payload = dict()
    if rep_sm is not None:
        payload["smooth"] = dict(
            mode=rep_sm.mode,
            phi_anchor_deg=rep_sm.phi_anchor_deg,
            candidates=rep_sm.candidates,
            selection_reason=rep_sm.selection_reason,
            wall_total_sec=rep_sm.wall_total_sec,
            ok=rep_sm.smooth_ok,
            lambda_log=rep_sm.smooth_lambda_log,
            ps_budgets={k: PS_BUDGETS[k] for k in
                         ("scout", "trial", "anchor_stage_first",
                          "anchor_stage_later", "accepted_node",
                          "midpoint_rescue")},
        )
        if rep_sm.smooth_state is not None:
            s = rep_sm.smooth_state
            payload["smooth"]["state"] = dict(
                phi_deg=s.phi_deg, X=s.X, Y=s.Y, eps=s.eps,
                attitude_deg=s.attitude_deg, h_min=s.h_min, p_max=s.p_max,
                Ploss=s.Ploss, friction=s.friction,
                rel_residual=s.rel_residual, status=s.status)
    if rep_tx is not None:
        payload["textured"] = dict(
            wall_total_sec=rep_tx.wall_total_sec,
            ok=rep_tx.textured_ok,
            log=rep_tx.textured_log,
        )
        if rep_tx.textured_state is not None:
            s = rep_tx.textured_state
            payload["textured"]["state"] = dict(
                phi_deg=s.phi_deg, X=s.X, Y=s.Y, eps=s.eps,
                attitude_deg=s.attitude_deg, h_min=s.h_min, p_max=s.p_max,
                Ploss=s.Ploss, friction=s.friction,
                rel_residual=s.rel_residual, status=s.status)
    with open(os.path.join(out_dir, "anchor.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    pa = argparse.ArgumentParser(description="Stage I-A: cold quasi-static diesel cycle")
    pa.add_argument("--n-points", type=int, default=CYCLE_N_COARSE)
    pa.add_argument("--grid", default="400x120")
    pa.add_argument("--phi-bc", default="periodic")
    pa.add_argument("--p-supply-bar", type=float, default=P_SUPPLY_PA/1e5)
    pa.add_argument("--out", required=True)
    pa.add_argument("--mode", default="independent_nr",
                    choices=["independent_nr", "continuation_phi"])
    pa.add_argument("--anchor-mode", default="explicit",
                    choices=["explicit", "from_legacy_matched_sector",
                             "scout_best"],
                    help="Stage I-A anchor selection policy. Default "
                         "'explicit' uses --phi-anchor-deg.")
    pa.add_argument("--phi-anchor-deg", type=float,
                    default=DEFAULT_EXPLICIT_PHI_ANCHOR_DEG,
                    help="Anchor crank angle for --anchor-mode=explicit. "
                         "Default 500° (medium-load sector for the "
                         "current surrogate_heavyduty_v1 case).")
    pa.add_argument("--legacy-history-csv",
                    help="Optional CSV from a previous run for "
                         "--anchor-mode=from_legacy_matched_sector.")
    args = pa.parse_args()

    Np,Nz=(int(x) for x in args.grid.split("x"))
    p1,z1,Phi,Zm,dp,dz=make_grid(Np,Nz)
    os.makedirs(args.out, exist_ok=True)

    cycle = build_surrogate_heavyduty_v1(n_points=args.n_points)
    placement = get_placement_from_reference(cycle, PHI_REF_DEG)
    phi_loaded = placement["phi_loaded_deg"]
    phi_unloaded = placement["phi_unloaded_deg"]

    # Feed mask
    p_Pa = float(args.p_supply_bar) * 1e5
    if p_Pa > 0:
        g_bc_val = p_supply_to_g_bc(p_Pa, ETA_COLD, OMEGA, R, c)
        feed_mask, feed_meta = build_feed_mask_variant(
            Phi, Zm, FEED_VARIANT,
            phi_loaded_deg=phi_loaded,
            phi_feed_half_deg=FEED_PHI_HALF_DEG,
            z_belt_half=FEED_Z_BELT_HALF)
    else:
        g_bc_val = None
        feed_mask = None
        feed_meta = {}

    # Texture relief
    tex_relief = build_relief(
        Phi, Zm, variant=TEX_VARIANT,
        depth_nondim=TEX_DG_M/c,
        N_branch_per_side=TEX_N_BRANCH,
        w_branch_nondim=0.004/R,
        belt_half_nondim=TEX_BELT_HALF,
        beta_deg=TEX_BETA_DEG,
        chirality=TEX_CHIRALITY,
        ramp_frac=TEX_RAMP_FRAC,
        taper_ratio=TEX_TAPER,
        coverage_mode="protect_loaded_union",
        protected_lo_deg=PROTECTED_LO,
        protected_hi_deg=PROTECTED_HI)
    smooth_relief = np.zeros_like(Phi)

    phi_arr = np.array(cycle["phi_crank_deg"])

    print(f"Stage I-A: cold quasi-static diesel cycle")
    print(f"Grid: {Np}x{Nz}, {args.n_points} crank angles, mode={args.mode}")
    print(f"Placement: phi_loaded={phi_loaded:.1f}° phi_unloaded={phi_unloaded:.1f}°")
    print(f"p_supply: {args.p_supply_bar} bar")

    all_rows: List[Dict] = []
    t_global = time.time()

    def _load_fn(phi_deg):
        return get_load_at_angle(cycle, phi_deg)

    if args.mode == "continuation_phi":
        legacy_phis = None
        if args.legacy_history_csv and os.path.exists(args.legacy_history_csv):
            try:
                with open(args.legacy_history_csv, "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    rows = [r for r in rdr]
                sm_ok = {float(r["phi_crank_deg"]) for r in rows
                          if r.get("geometry") == "smooth"
                          and r.get("status", "failed") != "failed"}
                tx_ok = {float(r["phi_crank_deg"]) for r in rows
                          if r.get("geometry") == "textured"
                          and r.get("status", "failed") != "failed"}
                legacy_phis = sorted(sm_ok & tx_ok)
                print(f"  [legacy] matched-sector phis: {len(legacy_phis)} "
                      f"angles where BOTH smooth+textured were accepted")
            except Exception as e:
                print(f"  [legacy] failed to parse CSV: {e}")
        all_rows = _run_continuation(
            phi_arr, cycle, Phi, Zm, p1, z1, dp, dz, args.phi_bc,
            smooth_relief, tex_relief, feed_mask, g_bc_val, p_Pa,
            _load_fn,
            anchor_mode=args.anchor_mode,
            phi_anchor_deg=args.phi_anchor_deg,
            legacy_matched_phis=legacy_phis,
            out_dir=args.out)
    else:
        all_rows = _run_independent_nr(
            phi_arr, cycle, Phi, Zm, p1, z1, dp, dz, args.phi_bc,
            smooth_relief, tex_relief, feed_mask, g_bc_val, p_Pa)

    total = time.time() - t_global

    # Quasi-static diagnostic
    for geo_tag in ["smooth", "textured"]:
        geo_rows = [r for r in all_rows if r["geometry"] == geo_tag]
        eps_arr = [r["eps"] for r in geo_rows]
        phi_arr_geo = [r["phi_crank_deg"] for r in geo_rows]
        R_sq = quasistatic_ratio(eps_arr, phi_arr_geo, OMEGA, R, c)
        max_Rsq = max(R_sq) if R_sq else 0
        print(f"\n  {geo_tag} quasi-static ratio max: {max_Rsq:.4f} "
              f"({'OK' if max_Rsq < 0.5 else 'WARN'})")

    # Summary metrics — full + matched
    print(f"\n{'='*60}")
    print(f"Summary (full — subsets may differ):")
    dphi_cycle = 720.0 / args.n_points
    dt_step = dphi_cycle / (360.0 * n_rpm / 60.0)
    for geo_tag in ["smooth", "textured"]:
        geo_rows = [r for r in all_rows if r["geometry"] == geo_tag
                     and r["status"] != "failed"]
        if not geo_rows:
            print(f"  {geo_tag}: all failed")
            continue
        Ploss_arr = [r["Ploss_W"] for r in geo_rows]
        hmin_arr = [r["h_min_um"] for r in geo_rows]
        pmax_arr = [r["p_max_MPa"] for r in geo_rows]
        eps_arr = [r["eps"] for r in geo_rows]
        Ef = sum(p * dt_step for p in Ploss_arr)
        Pmean = sum(Ploss_arr) / len(Ploss_arr)
        n_total = sum(1 for r in all_rows if r["geometry"] == geo_tag)
        print(f"  {geo_tag}: Ef={Ef:.3f}J Pmean={Pmean:.1f}W "
              f"hmin_min={min(hmin_arr):.1f}μm pmax_max={max(pmax_arr):.1f}MPa "
              f"eps_max={max(eps_arr):.4f} acc={len(geo_rows)}/{n_total}")

    # Matched comparison — only angles where BOTH converged
    sm_by_phi = {r["phi_crank_deg"]: r for r in all_rows
                  if r["geometry"] == "smooth" and r["status"] != "failed"}
    tx_by_phi = {r["phi_crank_deg"]: r for r in all_rows
                  if r["geometry"] == "textured" and r["status"] != "failed"}
    matched_phis = sorted(set(sm_by_phi.keys()) & set(tx_by_phi.keys()))
    frac_matched = len(matched_phis) / max(args.n_points, 1)
    print(f"\nMatched comparison ({len(matched_phis)}/{args.n_points} = "
          f"{frac_matched*100:.0f}% of cycle):")
    if matched_phis:
        sm_Ploss = [sm_by_phi[p]["Ploss_W"] for p in matched_phis]
        tx_Ploss = [tx_by_phi[p]["Ploss_W"] for p in matched_phis]
        Ef_sm = sum(p * dt_step for p in sm_Ploss)
        Ef_tx = sum(p * dt_step for p in tx_Ploss)
        Pm_sm = sum(sm_Ploss) / len(sm_Ploss)
        Pm_tx = sum(tx_Ploss) / len(tx_Ploss)
        dPm = (Pm_tx - Pm_sm) / max(Pm_sm, 1e-20) * 100
        print(f"  smooth_matched: Pmean={Pm_sm:.1f}W Ef={Ef_sm:.3f}J")
        print(f"  textured_matched: Pmean={Pm_tx:.1f}W Ef={Ef_tx:.3f}J")
        print(f"  ΔPmean = {dPm:+.1f}%")
    else:
        print(f"  no matched angles — cannot compare")

    # CSV
    cp = os.path.join(args.out, "cycle_history.csv")
    if all_rows:
        flds = sorted(all_rows[0].keys())
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    # Input case JSON
    case = dict(
        bearing=dict(D_m=D, L_m=L, c_m=c, rpm=n_rpm),
        oil=dict(name="mineral_105C", Tin_C=105, eta_Pa_s=ETA_COLD),
        feed=dict(variant=FEED_VARIANT, p_supply_Pa=p_Pa),
        texture=dict(variant=TEX_VARIANT, chirality=TEX_CHIRALITY,
                      N=TEX_N_BRANCH, beta_deg=TEX_BETA_DEG,
                      d_g_m=TEX_DG_M, taper_ratio=TEX_TAPER),
        cycle=dict(name=cycle["name"], n_points=args.n_points),
        placement=dict(mode=PLACEMENT_MODE, phi_ref_deg=PHI_REF_DEG,
                        phi_loaded_deg=phi_loaded,
                        phi_unloaded_deg=phi_unloaded),
        thermal=dict(mode="none"),
    )
    with open(os.path.join(args.out, "input_case.json"), "w") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)

    manifest = dict(
        schema_version=SCHEMA,
        stage="I-A_cold_cycle",
        mode=args.mode,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        grid=dict(N_phi=Np, N_Z=Nz),
        n_crank_angles=args.n_points,
        total_time_sec=total)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total:.0f}s")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
