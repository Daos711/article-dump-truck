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


def _write_summary(run_dir, results, thermal, configs, *,
                    grid, n_crank, runtime_s, cli_args):
    phi = results["phi_crank"]
    n_phi = len(phi)
    lines = [
        "Stage THD-0 BelAZ quasistatic run",
        f"  thermal mode    : {thermal.mode}",
        f"  gamma           : {thermal.gamma_mix:.4f}",
        f"  T_in_C          : {thermal.T_in_C:.2f}",
        f"  cp_J_kgK        : {thermal.cp_J_kgK:.1f}",
        f"  mdot_floor_kg_s : {thermal.mdot_floor_kg_s:.2e} (mdot policy: "
        "max(rho*|Q|, mdot_floor))",
        f"  tol_T_C         : {thermal.tol_T_C:.3f}",
        f"  max_outer       : {thermal.max_outer}",
        f"  underrelax_T    : {thermal.underrelax_T:.2f}",
        f"  grid            : {grid[0]} x {grid[1]} (N_phi x N_Z)",
        f"  n_crank         : {n_crank}",
        f"  runtime_s       : {runtime_s:.1f}",
        f"  cli             : {cli_args}",
        "",
    ]
    for ic, cfg in enumerate(configs):
        T_eff = results["T_eff"][ic]
        eta = results["eta_eff"][ic]
        Ploss = results["P_loss"][ic]
        hmin = results["hmin"][ic]
        pmax = results["pmax"][ic]
        conv = results["thermal_converged"][ic]
        floor = results["mdot_floor_hit"][ic]
        lines.extend([
            f"[{cfg['label']}]",
            f"  T_eff (C):  min={T_eff.min():.2f}  "
            f"mean={T_eff.mean():.2f}  max={T_eff.max():.2f}",
            f"  eta_eff (Pa*s): min={eta.min():.4e}  mean={eta.mean():.4e}",
            f"  P_loss (W):   mean={Ploss.mean():.1f}  max={Ploss.max():.1f}",
            f"  h_min (um):   min={hmin.min()*1e6:.2f}",
            f"  p_max (MPa):  max={pmax.max()/1e6:.2f}",
            f"  thermal_converged: {int(conv.sum())}/{n_phi}",
            f"  mdot_floor_hit:    {int(floor.sum())}/{n_phi}",
            "",
        ])
    with open(os.path.join(run_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_data(run_dir, results, thermal, grid):
    phi = results["phi_crank"]
    np.savez(
        os.path.join(run_dir, "data.npz"),
        phi_crank=phi,
        F_ext=results["F_ext"],
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
        thermal_converged=results["thermal_converged"],
        mdot_floor_hit=results["mdot_floor_hit"],
        thermal_mode=thermal.mode,
        gamma=thermal.gamma_mix,
        T_in_C=thermal.T_in_C,
        cp_J_kgK=thermal.cp_J_kgK,
        grid=np.asarray(grid, dtype=np.int32),
        labels=[c["label"] for c in results["configs"]],
    )


def _run_one(thermal: ThermalConfig, args, configs, out_base) -> str:
    run_dir = _make_run_dir(out_base, thermal.gamma_mix)
    print("=" * 60)
    print(f"  Stage THD-0 run -> {run_dir}")
    print(f"  mode={thermal.mode} gamma={thermal.gamma_mix:.3f} "
          f"T_in={thermal.T_in_C:.1f}°C n_crank={args.n_crank} "
          f"grid={args.n_phi_grid}x{args.n_z_grid}")
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
                    cli_args=" ".join(sys.argv[1:]))

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
    args = pa.parse_args(argv)

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
