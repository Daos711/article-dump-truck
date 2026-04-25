#!/usr/bin/env python3
"""Stage THD-0 — global_static thermal coupling for the BelAZ-class
quasistatic diesel runner.

Thin CLI on top of ``models.diesel_quasistatic.run_diesel_analysis``.
Solves the standard quasistatic cycle but with a per-angle outer
fixed-point iteration on the static energy balance (Section 3 of the
patch spec).

Usage::

    python scripts/run_diesel_thd_quasistatic.py \\
        --mode global_static --gamma 0.7 \\
        --n-crank 36 --n-phi-grid 160 --n-z-grid 60

Sweep mode::

    python scripts/run_diesel_thd_quasistatic.py \\
        --mode global_static --gamma-sweep 0.6,0.7,0.8 \\
        --n-crank 36

Each run lands in
``results/diesel_thd/<timestamp>_BelAZ_QS_static_gamma<gamma>/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.diesel_quasistatic import (
    CONFIG_KEYS, CONFIGS, run_diesel_analysis,
)
from models.thermal_coupling import ThermalConfig
from config import diesel_params as params


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


def _make_run_dir(out_base: str, gamma: float) -> str:
    tag = datetime.now().strftime("%y%m%d_%H%M%S")
    g_str = f"gamma{gamma:.2f}".replace(".", "p")
    name = f"{tag}_BelAZ_QS_static_{g_str}"
    d = os.path.join(out_base, name)
    os.makedirs(d, exist_ok=True)
    return d


def _plot_per_config(phi, data, ylabel, fname, title, configs, run_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for ic in range(data.shape[0]):
        cfg = configs[ic]
        ax.plot(phi, data[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1.6, label=cfg["label"])
    ax.set_xlabel("Crank angle phi (deg)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(phi[0], phi[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, fname), dpi=150)
    plt.close(fig)


def _stat_lines(arr_full, mask, *, scale=1.0, prefix, unit):
    """Return formatted min/mean/max lines for masked + full arrays."""
    out = []
    full = np.asarray(arr_full, dtype=float)
    finite_full = full[np.isfinite(full)]
    if mask.any():
        sub = full[mask]
        sub = sub[np.isfinite(sub)]
        if sub.size > 0:
            out.append(
                f"  {prefix} valid_fullfilm ({unit}): "
                f"min={sub.min()*scale:.4g}  "
                f"mean={sub.mean()*scale:.4g}  "
                f"max={sub.max()*scale:.4g}"
            )
        else:
            out.append(f"  {prefix} valid_fullfilm ({unit}): no finite values")
    else:
        out.append(f"  {prefix} valid_fullfilm ({unit}): n=0")
    if finite_full.size > 0:
        out.append(
            f"  {prefix} all diagnostic ({unit}): "
            f"min={finite_full.min()*scale:.4g}  "
            f"mean={finite_full.mean()*scale:.4g}  "
            f"max={finite_full.max()*scale:.4g}"
        )
    else:
        out.append(f"  {prefix} all diagnostic ({unit}): no finite values")
    return out


def _write_summary(run_dir, results, thermal, configs, *,
                    grid, n_crank, runtime_s, cli_args,
                    F_max_source=""):
    phi = results["phi_crank"]
    n_phi = len(phi)
    F_ext = np.asarray(results["F_ext"])
    lines = [
        "Stage THD-0B BelAZ quasistatic run",
        f"  thermal mode    : {thermal.mode}",
        f"  gamma           : {thermal.gamma_mix:.4f}",
        f"  T_in_C          : {thermal.T_in_C:.2f}",
        f"  cp_J_kgK        : {thermal.cp_J_kgK:.1f}",
        f"  mdot_floor_kg_s : {thermal.mdot_floor_kg_s:.2e} "
        "(mdot policy: max(rho*|Q|, mdot_floor))",
        f"  tol_T_C         : {thermal.tol_T_C:.3f}",
        f"  max_outer       : {thermal.max_outer}",
        f"  underrelax_T    : {thermal.underrelax_T:.2f}",
        f"  grid            : {grid[0]} x {grid[1]} (N_phi x N_Z)",
        f"  n_crank         : {n_crank}",
        f"  runtime_s       : {runtime_s:.1f}",
        f"  F_max_used      : {results.get('F_max_used', float('nan')):.1f} N",
        f"  F_max_source    : {F_max_source}" if F_max_source else "",
        f"  eps_max_hydro   : {results.get('eps_max_hydro', float('nan')):.3f}",
        f"  F_ext min/max   : {F_ext.min()/1000:.1f} kN / "
        f"{F_ext.max()/1000:.1f} kN",
        f"  cli             : {cli_args}",
        "",
    ]
    # Drop any blank-string placeholders.
    lines = [ln for ln in lines if ln != ""] + [""]
    for ic, cfg in enumerate(configs):
        valid = np.asarray(results["valid_fullfilm"][ic], dtype=bool)
        load_status = np.asarray(results["load_status"][ic])
        load_match = np.asarray(results["load_match_ratio"][ic],
                                  dtype=float)
        finite_match = load_match[np.isfinite(load_match)]
        lines.append(f"[{cfg['label']}]")
        lines.extend(_stat_lines(results["T_eff"][ic], valid,
                                    prefix="T_eff",   unit="C"))
        lines.extend(_stat_lines(results["eta_eff"][ic], valid,
                                    prefix="eta_eff", unit="Pa*s"))
        lines.extend(_stat_lines(results["P_loss"][ic], valid,
                                    prefix="P_loss",  unit="W"))
        lines.extend(_stat_lines(results["Q"][ic], valid,
                                    prefix="Q",       unit="m^3/s"))
        lines.extend(_stat_lines(results["mdot"][ic], valid,
                                    prefix="mdot",    unit="kg/s"))
        lines.extend(_stat_lines(results["hmin"][ic], valid,
                                    scale=1e6,
                                    prefix="h_min",   unit="um"))
        lines.extend(_stat_lines(results["pmax"][ic], valid,
                                    scale=1e-6,
                                    prefix="p_max",   unit="MPa"))
        lines.extend([
            "  load:",
            f"    F_ext min/max : {F_ext.min()/1000:.1f} kN / "
            f"{F_ext.max()/1000:.1f} kN",
            f"    W_table max   : "
            f"{float(results['W_table_max'][ic])/1000:.1f} kN",
            f"    W_table_finite: {bool(results['W_table_finite'][ic])}",
            f"    valid_fullfilm: {int(valid.sum())}/{n_phi}",
            f"    above_range   : {int(np.sum(load_status == 'above_range'))}",
            f"    below_range   : {int(np.sum(load_status == 'below_range'))}",
            f"    solver_failed : {int(np.sum(load_status == 'solver_failed'))}",
            f"    wtable_failed : "
            f"{int(np.sum(load_status == 'wtable_failed'))}",
        ])
        if finite_match.size > 0:
            lines.append(
                f"    load_match_ratio min/mean: "
                f"{finite_match.min():.3f} / {finite_match.mean():.3f}"
            )
        else:
            lines.append("    load_match_ratio min/mean: no finite values")
        lines.extend([
            "  thermal:",
            f"    energy_converged   : "
            f"{int(np.asarray(results['thermal_energy_converged'][ic]).sum())}"
            f"/{n_phi}",
            f"    thermal_converged  : "
            f"{int(np.asarray(results['thermal_converged'][ic]).sum())}"
            f"/{n_phi}",
            f"    mdot_floor_hit     : "
            f"{int(np.asarray(results['mdot_floor_hit'][ic]).sum())}/{n_phi}",
        ])
        outer_arr_ic = np.asarray(results["thermal_outer"][ic])
        max_outer_hit = int(np.sum(outer_arr_ic >= thermal.max_outer))
        lines.append(f"    max_outer hit count: {max_outer_hit}")
        lines.append("")
    with open(os.path.join(run_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_data(run_dir, results, thermal, grid):
    phi = results["phi_crank"]
    np.savez(
        os.path.join(run_dir, "data.npz"),
        phi_crank=phi,
        F_ext=results["F_ext"],
        F_max_used=results.get("F_max_used", float("nan")),
        eps_max_hydro=results.get("eps_max_hydro", float("nan")),
        epsilon=results["epsilon"],
        hmin=results["hmin"],
        f=results["f"],
        pmax=results["pmax"],
        F_hyd=results["F_hyd"],
        T_eff=results["T_eff"],
        T_target=results["T_target"],
        eta_eff=results["eta_eff"],
        P_loss=results["P_loss"],
        Q=results["Q"],
        mdot=results["mdot"],
        F_tr=results["F_tr"],
        thermal_outer=results["thermal_outer"],
        thermal_energy_converged=results["thermal_energy_converged"],
        thermal_converged=results["thermal_converged"],
        mdot_floor_hit=results["mdot_floor_hit"],
        load_status=results["load_status"],
        load_match_ratio=results["load_match_ratio"],
        valid_fullfilm=results["valid_fullfilm"],
        W_table_max=results["W_table_max"],
        W_table_finite=results["W_table_finite"],
        thermal_mode=thermal.mode,
        gamma=thermal.gamma_mix,
        T_in_C=thermal.T_in_C,
        cp_J_kgK=thermal.cp_J_kgK,
        grid=np.asarray(grid, dtype=np.int32),
        labels=[c["label"] for c in results["configs"]],
    )


def _run_one(thermal: ThermalConfig, args, configs, out_base) -> str:
    run_dir = _make_run_dir(out_base, thermal.gamma_mix)
    F_max_used = float(getattr(args, "F_max_resolved", 0.0)) or None
    F_max_source = getattr(args, "F_max_source", "")
    eps_max_hydro = getattr(args, "eps_max_hydro", None)
    print("=" * 60)
    print(f"  Stage THD-0B run -> {run_dir}")
    print(f"  mode={thermal.mode} gamma={thermal.gamma_mix:.3f} "
          f"T_in={thermal.T_in_C:.1f}°C n_crank={args.n_crank} "
          f"grid={args.n_phi_grid}x{args.n_z_grid}")
    if F_max_source:
        print(f"  F_max: {F_max_source}")
    if eps_max_hydro is not None:
        print(f"  eps_max_hydro override: {eps_max_hydro}")
    print("=" * 60)

    phi = np.linspace(0.0, 720.0, int(args.n_crank), endpoint=False)
    t0 = time.time()
    results = run_diesel_analysis(
        cavitation=args.cavitation,
        thermal=thermal,
        phi_crank=phi,
        n_phi_grid=args.n_phi_grid,
        n_z_grid=args.n_z_grid,
        configs=configs,
        F_max=F_max_used,
        eps_max_hydro=eps_max_hydro,
    )
    dt = time.time() - t0
    if args.max_wall_sec is not None and dt > args.max_wall_sec:
        print(f"  [WARN] runtime {dt:.0f}s exceeded "
              f"--max-wall-sec={args.max_wall_sec}")

    _save_data(run_dir, results, thermal,
                grid=(args.n_phi_grid, args.n_z_grid))
    _write_summary(run_dir, results, thermal, configs,
                    grid=(args.n_phi_grid, args.n_z_grid),
                    n_crank=int(args.n_crank), runtime_s=dt,
                    cli_args=" ".join(sys.argv[1:]),
                    F_max_source=F_max_source)

    cfg_for_plot = results["configs"]
    _plot_per_config(phi, results["T_eff"],
                     "T_eff (deg C)",
                     "T_eff_vs_phi.png",
                     f"Effective oil temperature (gamma={thermal.gamma_mix})",
                     cfg_for_plot, run_dir)
    _plot_per_config(phi, results["eta_eff"],
                     "eta_eff (Pa*s)",
                     "eta_eff_vs_phi.png",
                     f"Effective viscosity (gamma={thermal.gamma_mix})",
                     cfg_for_plot, run_dir)
    _plot_per_config(phi, results["P_loss"],
                     "P_loss (W)",
                     "P_loss_vs_phi.png",
                     f"Friction power loss (gamma={thermal.gamma_mix})",
                     cfg_for_plot, run_dir)
    _plot_per_config(phi, results["hmin"] * 1e6,
                     "h_min (um)",
                     "hmin_vs_phi.png",
                     "Minimum film thickness",
                     cfg_for_plot, run_dir)
    _plot_per_config(phi, results["pmax"] / 1e6,
                     "p_max (MPa)",
                     "pmax_vs_phi.png",
                     "Peak pressure",
                     cfg_for_plot, run_dir)

    print(f"  done in {dt:.1f}s")
    return run_dir


def main(argv=None):
    pa = argparse.ArgumentParser(description="Stage THD-0 BelAZ quasistatic")
    pa.add_argument("--mode", default="global_static",
                    choices=["off", "global_static"])
    pa.add_argument("--gamma", type=float, default=0.7)
    pa.add_argument("--gamma-sweep", default=None,
                    help="comma-separated list, e.g. 0.6,0.7,0.8")
    pa.add_argument("--n-crank", type=int, default=36)
    pa.add_argument("--n-phi-grid", type=int, default=160)
    pa.add_argument("--n-z-grid", type=int, default=60)
    pa.add_argument("--T-in", dest="T_in", type=float, default=105.0)
    pa.add_argument("--cp", type=float, default=2000.0)
    pa.add_argument("--mdot-floor", type=float, default=1e-4)
    pa.add_argument("--tol-T", type=float, default=0.5)
    pa.add_argument("--max-outer", type=int, default=5)
    pa.add_argument("--underrelax-T", type=float, default=0.6)
    pa.add_argument("--cavitation", default="half_sommerfeld")
    pa.add_argument("--configs", default=None,
                    help="comma-separated keys: "
                         + ", ".join(sorted(CONFIG_KEYS)))
    pa.add_argument("--max-wall-sec", type=int, default=1800)
    pa.add_argument("--out-base", default=os.path.join(ROOT, "results",
                                                          "diesel_thd"))
    pa.add_argument("--F-max", dest="F_max", type=float, default=None,
                    help="Override load-cycle peak in N. Default uses "
                         "params.F_max (production BelAZ, 850 kN).")
    pa.add_argument("--F-max-debug", dest="F_max_debug",
                    action="store_true",
                    help="Use params.F_max_debug (200 kN) — sanity / "
                         "thermal-plumbing only, NOT a dissertation run.")
    pa.add_argument("--F-max-scale", dest="F_max_scale", type=float,
                    default=None,
                    help="Multiplicative scale on F_max (after "
                         "--F-max / --F-max-debug). For sensitivity "
                         "studies; defaults to 1.0.")
    pa.add_argument("--eps-max-hydro", dest="eps_max_hydro",
                    type=float, default=None,
                    help="High-eps cap for the hydrodynamic load matcher. "
                         "Default uses params.eps_max (0.95).")
    args = pa.parse_args(argv)

    # Resolve F_max precedence: explicit --F-max > --F-max-debug > default;
    # --F-max-scale applies on top of whichever was selected.
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
            tol_T_C=args.tol_T,
            max_outer=args.max_outer,
            underrelax_T=args.underrelax_T,
        )
        out_dirs.append(_run_one(thermal, args, configs, args.out_base))

    print("\nFinished runs:")
    for d in out_dirs:
        print("  " + d)


if __name__ == "__main__":
    main()
