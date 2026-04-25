#!/usr/bin/env python3
"""Stage Diesel Transient THD-0 — global_relax thermal coupling for the
BelAZ-class transient diesel runner.

Thin CLI wrapper around ``models.diesel_transient.run_transient``. Solves
the standard transient bearing problem (Reynolds + squeeze + shaft EOM)
but with a per-step thermal state advanced via ``global_relax_step``
(time-relaxation toward the static energy target).

Usage::

    python scripts/run_diesel_thd_transient.py \\
        --mode global_relax --gamma 0.7 --tau-th 0.5 \\
        --n-grid 120 --n-cycles 2 \\
        --d-phi-base 10 --d-phi-peak 2

Sweep mode (one run per gamma)::

    python scripts/run_diesel_thd_transient.py \\
        --mode global_relax --gamma-sweep 0.6,0.7,0.8 ...

Each run lands in
``results/diesel_thd_transient/<timestamp>_BelAZ_transient_<mode>_gamma<g>_tau<t>/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.diesel_transient import (
    CONFIGS, CONFIG_KEYS, run_transient,
)
from models.thermal_coupling import ThermalConfig
from models.diesel_quasistatic import SolverRetryConfig
from config import diesel_params as params


# ─── helpers ──────────────────────────────────────────────────────

def _parse_csv(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None:
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return parts or None


def _select_configs(keys: Optional[List[str]]):
    if keys is None:
        return list(CONFIGS)
    out = []
    for k in keys:
        if k not in CONFIG_KEYS:
            raise SystemExit(
                f"Unknown config key {k!r}; valid keys are: "
                f"{sorted(CONFIG_KEYS)}")
        out.append(CONFIGS[CONFIG_KEYS[k]])
    return out


def _make_run_dir(out_base: str, mode: str, gamma: float, tau_th: float,
                    *, extra_tag: str = "") -> str:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    g_str = f"gamma{gamma:.2f}".replace(".", "p")
    t_str = f"tau{tau_th:.2f}".replace(".", "p")
    name = f"{ts}_BelAZ_transient_{mode}_{g_str}_{t_str}"
    if extra_tag:
        name += f"_{extra_tag}"
    d = os.path.join(out_base, name)
    os.makedirs(d, exist_ok=True)
    return d


def _last_cycle_slice(results) -> slice:
    s = int(results["last_start"])
    nc = int(results["n_steps_per_cycle"])
    return slice(s, s + nc)


def _phi_last_mod(results) -> np.ndarray:
    phi = np.asarray(results["phi_last"]) % 720.0
    return phi


def _plot_last_cycle(results, key, ylabel, fname, title, run_dir,
                       *, scale: float = 1.0):
    sl = _last_cycle_slice(results)
    phi = _phi_last_mod(results)
    arr = np.asarray(results[key])[:, sl] * scale
    cfgs = results["configs"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for ic, cfg in enumerate(cfgs):
        ax.plot(phi, arr[ic],
                color=cfg.get("color", None),
                linestyle=cfg.get("ls", "-"),
                linewidth=1.4, label=cfg["label"])
    ax.set_xlabel("Crank angle phi (deg)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(phi.min(), phi.max())
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, fname), dpi=150)
    plt.close(fig)


def _plot_orbit(results, run_dir):
    sl = _last_cycle_slice(results)
    cfgs = results["configs"]
    eps_x = np.asarray(results["eps_x"])[:, sl]
    eps_y = np.asarray(results["eps_y"])[:, sl]
    fig, ax = plt.subplots(figsize=(7, 7))
    for ic, cfg in enumerate(cfgs):
        ax.plot(eps_x[ic], eps_y[ic],
                color=cfg.get("color", None),
                linestyle=cfg.get("ls", "-"),
                linewidth=1.2, label=cfg["label"])
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("eps_x", fontsize=11)
    ax.set_ylabel("eps_y", fontsize=11)
    ax.set_title("Shaft orbit — last cycle", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "orbit.png"), dpi=150)
    plt.close(fig)


def _plot_valid_status(results, run_dir):
    """One row per config; categorical y for solver_success/valid_no_clamp/contact_clamp/etc."""
    sl = _last_cycle_slice(results)
    phi = _phi_last_mod(results)
    cfgs = results["configs"]
    n_cfg = len(cfgs)
    valid_dyn = np.asarray(results["valid_dynamic"])[:, sl]
    contact = np.asarray(results["contact_clamp"])[:, sl]
    solver_ok = np.asarray(results["solver_success"])[:, sl]
    retry = np.asarray(results["retry_used"])[:, sl]
    fig, ax = plt.subplots(figsize=(11, 1.0 * n_cfg + 2))
    for ic, cfg in enumerate(cfgs):
        ys = ic
        # Category encoding: 0 = bad (failed solver), 1 = clamp,
        # 2 = retry-recovered, 3 = clean valid.
        cat = np.zeros_like(phi, dtype=float)
        cat[solver_ok[ic]] = 1.0
        cat[solver_ok[ic] & contact[ic]] = 1.5
        cat[solver_ok[ic] & ~contact[ic]] = 3.0
        cat[solver_ok[ic] & retry[ic]] = 2.0
        ax.scatter(phi, ys + cat * 0.0 + 0.0, c=cat, cmap="RdYlGn",
                   s=18, vmin=0, vmax=3, marker="s")
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels([c["label"] for c in cfgs], fontsize=8)
    ax.set_xlabel("Crank angle phi (deg)", fontsize=11)
    ax.set_title("Per-step validity (red=fail, orange=clamp, "
                 "yellow=retry, green=clean valid_no_clamp)",
                 fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "valid_status_vs_phi.png"), dpi=150)
    plt.close(fig)


def _plot_retry_status(results, run_dir):
    cfgs = results["configs"]
    n_cfg = len(cfgs)
    rec = np.asarray(results.get("retry_recovered_count",
                                    np.zeros(n_cfg)))
    exh = np.asarray(results.get("retry_exhausted_count",
                                    np.zeros(n_cfg)))
    fail = np.asarray(results.get("solver_failed_count",
                                     np.zeros(n_cfg)))
    omega_hits = results.get("omega_hits_per_config",
                                [{}] * n_cfg)
    fig, ax = plt.subplots(figsize=(10, 0.6 * n_cfg + 2))
    ys = np.arange(n_cfg)
    ax.barh(ys - 0.20, rec, height=0.18, color="#117733",
            label="retry recovered")
    ax.barh(ys + 0.00, exh, height=0.18, color="#cc6677",
            label="retry exhausted")
    ax.barh(ys + 0.20, fail, height=0.18, color="#aa4499",
            label="solver_failed (total)")
    for i, hits in enumerate(omega_hits):
        if not hits:
            continue
        s = ", ".join(f"{tag}: {n}" for tag, n in sorted(hits.items()))
        ax.annotate(s, (max(0, max(rec[i], exh[i], fail[i]) + 0.1),
                              i + 0.4),
                    fontsize=8, color="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels([c["label"] for c in cfgs], fontsize=8)
    ax.set_xlabel("Step count", fontsize=11)
    ax.set_title("Retry policy outcomes per config", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "retry_status.png"), dpi=150)
    plt.close(fig)


def _stats_line(arr_full, mask, *, scale: float = 1.0,
                  prefix: str, unit: str):
    """min/mean/max of arr_full[mask] (all per-config), formatted."""
    v = arr_full[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (f"  {prefix} valid_dynamic ({unit}): n=0 / no finite values")
    return (
        f"  {prefix} valid_dynamic ({unit}): "
        f"min={v.min()*scale:.4g}  mean={v.mean()*scale:.4g}  "
        f"max={v.max()*scale:.4g}"
    )


# ─── output: data + summary ───────────────────────────────────────

def _save_data(run_dir, results, thermal, retry_cfg):
    np.savez(
        os.path.join(run_dir, "data.npz"),
        phi_crank_deg=results["phi_crank_deg"],
        phi_last=results["phi_last"],
        last_start=int(results["last_start"]),
        n_steps_per_cycle=int(results["n_steps_per_cycle"]),
        eps_x=results["eps_x"],
        eps_y=results["eps_y"],
        hmin=results["hmin"],
        pmax=results["pmax"],
        f=results["f"],
        F_tr=results["F_tr"],
        N_loss=results["N_loss"],
        Fx_hyd=results["Fx_hyd"],
        Fy_hyd=results["Fy_hyd"],
        Fy_ext_last=results["Fy_ext_last"],
        F_max=float(results["F_max"]),
        T_eff_used=results["T_eff_used"],
        T_eff=results["T_eff"],
        T_target=results["T_target"],
        eta_eff=results["eta_eff"],
        eta_eff_next=results["eta_eff_next"],
        P_loss=results["P_loss"],
        Q=results["Q"],
        mdot=results["mdot"],
        mdot_floor_hit=results["mdot_floor_hit"],
        solver_success=results["solver_success"],
        valid_dynamic=results["valid_dynamic"],
        valid_no_clamp=results["valid_no_clamp"],
        contact_clamp=results["contact_clamp"],
        retry_used=results["retry_used"],
        retry_omega_used=results["retry_omega_used"],
        contact_clamp_count=results["contact_clamp_count"],
        solver_failed_count=results["solver_failed_count"],
        retry_recovered_count=results["retry_recovered_count"],
        retry_exhausted_count=results["retry_exhausted_count"],
        thermal_cycle_delta=results["thermal_cycle_delta"],
        thermal_periodic_converged=results["thermal_periodic_converged"],
        thermal_mode=thermal.mode,
        gamma=thermal.gamma_mix,
        tau_th_s=thermal.tau_th_s,
        T_in_C=thermal.T_in_C,
        cp_J_kgK=thermal.cp_J_kgK,
        labels=[c["label"] for c in results["configs"]],
    )


def _write_summary(run_dir, results, thermal, retry_cfg, *,
                    grid: int, n_cycles: int,
                    d_phi_base: float, d_phi_peak: float,
                    runtime_s: float, cli_args: str,
                    F_max_source: str = ""):
    sl = _last_cycle_slice(results)
    n_steps_last = int(results["n_steps_per_cycle"])
    cfgs = results["configs"]
    lines = [
        "Stage Diesel Transient THD-0 BelAZ run",
        f"  thermal mode    : {thermal.mode}",
        f"  gamma           : {thermal.gamma_mix:.4f}",
        f"  tau_th_s        : {thermal.tau_th_s:.4f}",
        f"  T_in_C          : {thermal.T_in_C:.2f}",
        f"  cp_J_kgK        : {thermal.cp_J_kgK:.1f}",
        f"  mdot_floor_kg_s : {thermal.mdot_floor_kg_s:.2e} "
        "(mdot policy: max(rho*|Q|, mdot_floor))",
        f"  grid            : {grid} x {grid} (N_phi x N_Z)",
        f"  n_cycles        : {n_cycles}",
        f"  d_phi_base/peak : {d_phi_base}° / {d_phi_peak}°",
        f"  F_max           : {float(results['F_max'])/1e3:.1f} kN",
        f"  F_max_source    : {F_max_source}" if F_max_source else "",
        f"  runtime_s       : {runtime_s:.1f}",
        f"  retry policy    : "
        f"{'enabled' if retry_cfg.enabled else 'disabled'} "
        f"omegas={list(retry_cfg.omega_values)} "
        f"max_iter={retry_cfg.max_iter_retry} "
        f"cold_start={retry_cfg.cold_start} "
        f"textured_only={retry_cfg.textured_only}",
        f"  cli             : {cli_args}",
        "",
    ]
    lines = [ln for ln in lines if ln != ""] + [""]

    for ic, cfg in enumerate(cfgs):
        valid_dyn = np.asarray(results["valid_dynamic"][ic, sl], dtype=bool)
        valid_noc = np.asarray(results["valid_no_clamp"][ic, sl], dtype=bool)
        contact = np.asarray(results["contact_clamp"][ic, sl], dtype=bool)
        solver_ok = np.asarray(results["solver_success"][ic, sl], dtype=bool)
        floor = np.asarray(results["mdot_floor_hit"][ic, sl], dtype=bool)
        retry_used = np.asarray(results["retry_used"][ic, sl], dtype=bool)

        def _stat(key, *, scale=1.0, label=None, unit=""):
            arr = np.asarray(results[key][ic, sl])
            return _stats_line(arr, valid_dyn, scale=scale,
                                  prefix=(label or key), unit=unit)

        lines.append(f"[{cfg['label']}]")
        lines.append(_stat("T_eff", label="T_eff",  unit="C"))
        lines.append(_stat("T_target", label="T_target", unit="C"))
        lines.append(_stat("eta_eff", label="eta_eff", unit="Pa*s"))
        lines.append(_stat("P_loss",  label="P_loss",  unit="W"))
        lines.append(_stat("Q",       label="Q",       unit="m^3/s"))
        lines.append(_stat("mdot",    label="mdot",    unit="kg/s"))
        lines.append(_stat("hmin",    label="h_min",   unit="um", scale=1e6))
        lines.append(_stat("pmax",    label="p_max",   unit="MPa", scale=1e-6))
        delta_t = float(results["thermal_cycle_delta"][ic])
        per_conv = bool(results["thermal_periodic_converged"][ic])
        omega_hits = (results.get("omega_hits_per_config",
                                     [{}] * (ic + 1))[ic])
        omega_str = (", ".join(f"{t}={n}" for t, n
                                  in sorted(omega_hits.items()))
                       or "(none)")
        lines.extend([
            f"  solver_success     : {int(solver_ok.sum())}/{n_steps_last}",
            f"  valid_dynamic      : {int(valid_dyn.sum())}/{n_steps_last}",
            f"  valid_no_clamp     : {int(valid_noc.sum())}/{n_steps_last}",
            f"  contact_clamp      : {int(contact.sum())}/{n_steps_last}",
            f"  mdot_floor_hit     : {int(floor.sum())}/{n_steps_last}",
            f"  retry_recovered    : {int(retry_used.sum())}/{n_steps_last}",
            f"  retry_exhausted    : "
            f"{int(results['retry_exhausted_count'][ic])}",
            f"  retry omega hits   : {omega_str}",
            f"  thermal_cycle_delta: "
            f"{delta_t if np.isfinite(delta_t) else float('nan'):.3f} C",
            f"  thermal_periodic_converged: {per_conv}",
            "",
        ])

    # Paired comparison block.
    paired = results.get("paired_comparison") or []
    lines.append("=" * 60)
    lines.append(
        "Paired smooth-vs-textured comparison "
        "(common_valid_no_clamp mask only — same time steps both sides)"
    )
    lines.append("=" * 60)
    if not paired:
        lines.append("  (no smooth/textured pair found in this run)")
    for rec in paired:
        lines.extend([
            f"[{rec['oil_name']}]",
            f"  smooth   : {rec['smooth_label']}",
            f"  textured : {rec['textured_label']}",
            f"  common_valid_dynamic   : "
            f"{rec['common_valid_count']}/{n_steps_last}",
            f"  common_valid_no_clamp  : "
            f"{rec['common_no_clamp_count']}/{n_steps_last}",
        ])
        if rec["common_no_clamp_count"] == 0:
            lines.append("  no overlap — paired stats unavailable")
            lines.append("")
            continue
        lines.extend([
            f"  mean(T_eff)   smooth / textured : "
            f"{rec['mean_T_smooth']:.2f} / "
            f"{rec['mean_T_textured']:.2f}  (delta = "
            f"{rec['mean_dT_eff']:+.2f} C)",
            f"  mean(P_loss)  smooth / textured : "
            f"{rec['mean_P_loss_smooth']:.1f} / "
            f"{rec['mean_P_loss_textured']:.1f}  (delta = "
            f"{rec['mean_dP_loss']:+.1f} W)",
            f"  mean(d eta_eff)                 : "
            f"{rec['mean_deta_eff']:+.4e} Pa*s",
            f"  mean(d h_min)                   : "
            f"{rec['mean_dh_min']*1e6:+.3f} um",
            f"  d p_max range                   : "
            f"[{rec['min_dp_max']/1e6:+.2f}, "
            f"{rec['max_dp_max']/1e6:+.2f}] MPa",
        ])
        lines.append("")

    with open(os.path.join(run_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_one(thermal: ThermalConfig, retry_cfg: SolverRetryConfig,
              args, configs, out_base) -> str:
    F_max_used = float(getattr(args, "F_max_resolved", 0.0)) or None
    F_max_source = getattr(args, "F_max_source", "")
    extra_tag = ""
    if getattr(args, "F_max_debug", False):
        extra_tag = "Fmaxdebug"
    if not retry_cfg.enabled:
        extra_tag = (extra_tag + "_no_retry").strip("_")
    run_dir = _make_run_dir(out_base, thermal.mode,
                              thermal.gamma_mix, thermal.tau_th_s,
                              extra_tag=extra_tag)
    print("=" * 60)
    print(f"  Stage Diesel Transient THD-0 -> {run_dir}")
    print(f"  mode={thermal.mode} gamma={thermal.gamma_mix:.3f} "
          f"tau_th={thermal.tau_th_s:.3f}s "
          f"T_in={thermal.T_in_C:.1f}°C n_cycles={args.n_cycles} "
          f"grid={args.n_grid}")
    if F_max_source:
        print(f"  F_max: {F_max_source}")
    print(f"  retry: enabled={retry_cfg.enabled} "
          f"omegas={list(retry_cfg.omega_values)} "
          f"textured_only={retry_cfg.textured_only}")
    print("=" * 60)

    t0 = time.time()
    results = run_transient(
        F_max=F_max_used,
        debug=False,
        thermal=thermal,
        configs=configs,
        n_grid=args.n_grid,
        n_cycles=args.n_cycles,
        d_phi_base_deg=args.d_phi_base,
        d_phi_peak_deg=args.d_phi_peak,
        retry_config=retry_cfg,
    )
    dt = time.time() - t0
    if args.max_wall_sec is not None and dt > args.max_wall_sec:
        print(f"  [WARN] runtime {dt:.0f}s exceeded "
              f"--max-wall-sec={args.max_wall_sec}")

    _save_data(run_dir, results, thermal, retry_cfg)
    _write_summary(run_dir, results, thermal, retry_cfg,
                    grid=int(args.n_grid),
                    n_cycles=int(args.n_cycles),
                    d_phi_base=float(args.d_phi_base),
                    d_phi_peak=float(args.d_phi_peak),
                    runtime_s=dt,
                    cli_args=" ".join(sys.argv[1:]),
                    F_max_source=F_max_source)

    title_g = f"gamma={thermal.gamma_mix:.2f} tau={thermal.tau_th_s:.2f}s"
    _plot_last_cycle(results, "T_eff", "T_eff (deg C)",
                      "T_eff_vs_phi.png",
                      f"Effective oil temperature ({title_g})", run_dir)
    _plot_last_cycle(results, "T_target", "T_target (deg C)",
                      "T_target_vs_phi.png",
                      f"Static energy target ({title_g})", run_dir)
    _plot_last_cycle(results, "eta_eff", "eta_eff (Pa*s)",
                      "eta_eff_vs_phi.png",
                      f"Effective viscosity ({title_g})", run_dir)
    _plot_last_cycle(results, "P_loss", "P_loss (W)",
                      "P_loss_vs_phi.png",
                      f"Friction power loss ({title_g})", run_dir)
    _plot_last_cycle(results, "mdot", "mdot (kg/s)",
                      "mdot_vs_phi.png",
                      f"Side-leakage mass flow ({title_g})", run_dir)
    _plot_last_cycle(results, "hmin", "h_min (um)",
                      "hmin_vs_phi.png",
                      "Minimum film thickness", run_dir, scale=1e6)
    _plot_last_cycle(results, "pmax", "p_max (MPa)",
                      "pmax_vs_phi.png",
                      "Peak pressure", run_dir, scale=1e-6)
    sl = _last_cycle_slice(results)
    eps_mag = np.sqrt(np.asarray(results["eps_x"][:, sl]) ** 2
                       + np.asarray(results["eps_y"][:, sl]) ** 2)
    # eps plot is hand-built since results doesn't store eps_mag
    fig, ax = plt.subplots(figsize=(10, 5))
    phi = _phi_last_mod(results)
    for ic, cfg in enumerate(results["configs"]):
        ax.plot(phi, eps_mag[ic], color=cfg.get("color"),
                linestyle=cfg.get("ls", "-"), linewidth=1.4,
                label=cfg["label"])
    ax.set_xlabel("Crank angle phi (deg)", fontsize=12)
    ax.set_ylabel("|eps|", fontsize=12)
    ax.set_title("Eccentricity magnitude — last cycle", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "eps_vs_phi.png"), dpi=150)
    plt.close(fig)

    _plot_orbit(results, run_dir)
    _plot_valid_status(results, run_dir)
    _plot_retry_status(results, run_dir)

    print(f"  done in {dt:.1f}s")
    return run_dir


def main(argv=None):
    pa = argparse.ArgumentParser(
        description="Stage Diesel Transient THD-0 BelAZ")
    pa.add_argument("--mode", default="global_relax",
                    choices=["off", "global_static", "global_relax"])
    pa.add_argument("--gamma", type=float, default=0.7)
    pa.add_argument("--gamma-sweep", default=None,
                    help="comma-separated list, e.g. 0.6,0.7,0.8")
    pa.add_argument("--tau-th", dest="tau_th", type=float, default=0.5,
                    help="Thermal time constant (s) for global_relax. "
                         "Default 0.5.")
    pa.add_argument("--T-in", dest="T_in", type=float, default=105.0)
    pa.add_argument("--cp", type=float, default=2000.0)
    pa.add_argument("--mdot-floor", type=float, default=1e-4)
    pa.add_argument("--n-grid", type=int, default=120)
    pa.add_argument("--n-cycles", type=int, default=2)
    pa.add_argument("--d-phi-base", type=float, default=10.0,
                    help="Base crank-angle step (deg).")
    pa.add_argument("--d-phi-peak", type=float, default=2.0,
                    help="Adaptive step at firing peak (deg).")
    pa.add_argument("--cavitation", default="half_sommerfeld")
    pa.add_argument("--configs", default=None,
                    help="comma-separated keys: "
                         + ", ".join(sorted(CONFIG_KEYS)))
    pa.add_argument("--max-wall-sec", type=int, default=1800)
    pa.add_argument("--out-base",
                    default=os.path.join(ROOT, "results",
                                            "diesel_thd_transient"))
    pa.add_argument("--F-max", dest="F_max", type=float, default=None,
                    help="Override load-cycle peak in N. Default uses "
                         "params.F_max (production BelAZ, 850 kN).")
    pa.add_argument("--F-max-debug", dest="F_max_debug",
                    action="store_true",
                    help="Use params.F_max_debug (200 kN).")
    pa.add_argument("--F-max-scale", dest="F_max_scale", type=float,
                    default=None,
                    help="Multiplicative scale on F_max.")
    pa.add_argument("--retry-omega", dest="retry_omega",
                    default="1.70,1.55",
                    help="Comma-separated SOR omega values for the "
                         "conservative retry sequence on textured "
                         "failures. Defaults to '1.70,1.55'.")
    pa.add_argument("--retry-max-iter", dest="retry_max_iter",
                    type=int, default=100_000)
    pa.add_argument("--no-texture-retry", dest="no_texture_retry",
                    action="store_true",
                    help="Disable textured-only conservative SOR retry.")
    args = pa.parse_args(argv)

    # Resolve F_max.
    if args.F_max is not None:
        F_max_resolved = float(args.F_max)
        F_max_source = f"explicit override = {F_max_resolved:.1f} N"
    elif args.F_max_debug:
        F_max_resolved = float(getattr(params, "F_max_debug",
                                          params.F_max))
        F_max_source = (f"params.F_max_debug = {F_max_resolved:.1f} N "
                        "(SANITY MODE — not a production result)")
    else:
        F_max_resolved = float(params.F_max)
        F_max_source = (f"params.F_max = {F_max_resolved:.1f} N "
                        "(production BelAZ)")
    if args.F_max_scale is not None:
        scale = float(args.F_max_scale)
        F_max_resolved *= scale
        F_max_source += f" * scale={scale}"
    args.F_max_resolved = F_max_resolved
    args.F_max_source = F_max_source

    # Retry policy.
    if args.no_texture_retry:
        retry_cfg = SolverRetryConfig.disabled()
    else:
        try:
            omegas = tuple(float(x) for x in
                            args.retry_omega.split(",") if x.strip())
        except Exception as exc:
            raise SystemExit(
                f"--retry-omega must be a comma-separated list of "
                f"floats; got {args.retry_omega!r}: {exc}")
        retry_cfg = SolverRetryConfig(
            enabled=True,
            textured_only=True,
            omega_values=omegas,
            max_iter_retry=int(args.retry_max_iter),
            cold_start=True,
        )

    configs = _select_configs(_parse_csv(args.configs))

    if args.gamma_sweep:
        gammas = [float(x) for x in args.gamma_sweep.split(",")]
    else:
        gammas = [float(args.gamma)]

    out_dirs = []
    for g in gammas:
        thermal = ThermalConfig(
            mode=args.mode,
            T_in_C=args.T_in,
            gamma_mix=g,
            cp_J_kgK=args.cp,
            mdot_floor_kg_s=args.mdot_floor,
            tau_th_s=float(args.tau_th),
        )
        out_dirs.append(_run_one(thermal, retry_cfg, args,
                                    configs, args.out_base))

    print("\nFinished runs:")
    for d in out_dirs:
        print("  " + d)


if __name__ == "__main__":
    main()
