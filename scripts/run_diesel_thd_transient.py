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
    CONFIGS, CONFIG_KEYS, EnvelopeAbortConfig, run_transient,
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


def _plot_orbit_lastcycle(results, run_dir, *,
                            fname: str = "orbit_lastcycle.png"):
    """Per-config last-cycle orbit on (eps_x, eps_y) with the
    eps=1 contact circle.

    Smooth configs are drawn solid, textured dashed (per Section 4
    of the patch spec). All configs share one figure so smooth vs
    textured overlay is direct.
    """
    sl = _last_cycle_slice(results)
    cfgs = results["configs"]
    eps_x = np.asarray(results["eps_x"])[:, sl]
    eps_y = np.asarray(results["eps_y"])[:, sl]
    fig, ax = plt.subplots(figsize=(7, 7))
    for ic, cfg in enumerate(cfgs):
        ls = "--" if cfg.get("textured") else "-"
        ax.plot(eps_x[ic], eps_y[ic],
                color=cfg.get("color", None),
                linestyle=ls, linewidth=1.4,
                label=cfg["label"])
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.6, alpha=0.4,
            label="eps=1 contact")
    ax.set_xlabel("eps_x = e_x / c", fontsize=11)
    ax.set_ylabel("eps_y = e_y / c", fontsize=11)
    ax.set_title("Shaft orbit — last cycle", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
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
    any_aborted = bool(np.any(np.asarray(results.get("aborted",
                                                          False))))
    save_partial = bool(
        results.get("envelope_abort_config", {}).get(
            "save_partial_on_abort", True))
    fname = ("data_partial.npz" if (any_aborted and save_partial)
              else "data.npz")
    np.savez(
        os.path.join(run_dir, fname),
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
        contact_clamp_event_count=results.get(
            "contact_clamp_event_count",
            np.zeros_like(results["contact_clamp"], dtype=np.int32)),
        # Stage Diesel Transient AngleWeighted Metrics — per-step
        # angular increment + crank position, plus the angle-
        # weighted dicts (object arrays) for regenerate-summary.
        d_phi_per_step=results.get(
            "d_phi_per_step",
            np.zeros_like(results["contact_clamp"], dtype=float)),
        phi_mod_per_step=results.get(
            "phi_mod_per_step",
            np.zeros_like(results["contact_clamp"], dtype=float)),
        angle_weighted_full=np.array(
            results.get("angle_weighted_full", []), dtype=object),
        angle_weighted_last_cycle=np.array(
            results.get("angle_weighted_last_cycle", []), dtype=object),
        # Stage J — persist cavitation/texture/Ausas fields so
        # ``--regenerate-summary`` can rebuild the Stage J block from
        # an existing data.npz without rerunning the solver.
        cavitation_model=str(results.get("cavitation_model",
                                              "half_sommerfeld")),
        texture_kind=str(results.get("texture_kind", "dimple")),
        groove_preset=str(results.get("groove_preset") or "N/A"),
        groove_preset_resolved=np.asarray(
            [results.get("groove_preset_resolved") or {}], dtype=object),
        groove_relief_stats=np.asarray(
            [results.get("groove_relief_stats") or {}], dtype=object),
        fidelity=str(results.get("fidelity") or "custom"),
        ausas_options=np.asarray(
            [results.get("ausas_options") or {}], dtype=object),
        ausas_converged=results.get(
            "ausas_converged",
            np.zeros_like(results["contact_clamp"], dtype=bool)),
        ausas_n_inner=results.get(
            "ausas_n_inner",
            np.zeros_like(results["contact_clamp"], dtype=np.int32)),
        ausas_cav_frac=results.get(
            "ausas_cav_frac",
            np.zeros_like(results["contact_clamp"], dtype=float)),
        ausas_theta_min=results.get(
            "ausas_theta_min",
            np.ones_like(results["contact_clamp"], dtype=float)),
        ausas_theta_max=results.get(
            "ausas_theta_max",
            np.ones_like(results["contact_clamp"], dtype=float)),
        ausas_state_reset_count=results.get(
            "ausas_state_reset_count",
            np.zeros(len(results["configs"]), dtype=np.int32)),
        ausas_failed_step_count=results.get(
            "ausas_failed_step_count",
            np.zeros(len(results["configs"]), dtype=np.int32)),
        ausas_rejected_commit_count=results.get(
            "ausas_rejected_commit_count",
            np.zeros(len(results["configs"]), dtype=np.int32)),
        retry_used=results["retry_used"],
        retry_omega_used=results["retry_omega_used"],
        contact_clamp_count=results["contact_clamp_count"],
        solver_failed_count=results["solver_failed_count"],
        retry_recovered_count=results["retry_recovered_count"],
        retry_exhausted_count=results["retry_exhausted_count"],
        thermal_cycle_delta=results["thermal_cycle_delta"],
        thermal_periodic_converged=results["thermal_periodic_converged"],
        aborted=results["aborted"],
        abort_reason=results["abort_reason"],
        first_clamp_phi=results["first_clamp_phi"],
        first_solver_failed_phi=results["first_solver_failed_phi"],
        first_invalid_phi=results["first_invalid_phi"],
        steps_attempted=results["steps_attempted"],
        steps_completed=results["steps_completed"],
        applicable=results["applicable"],
        applicable_reason=results["applicable_reason"],
        thermal_mode=thermal.mode,
        gamma=thermal.gamma_mix,
        tau_th_s=thermal.tau_th_s,
        T_in_C=thermal.T_in_C,
        cp_J_kgK=thermal.cp_J_kgK,
        mdot_floor_kg_s=thermal.mdot_floor_kg_s,
        labels=[c["label"] for c in results["configs"]],
        textured_flags=[bool(c.get("textured", False))
                         for c in results["configs"]],
    )


def _write_summary(run_dir, results, thermal, retry_cfg, *,
                    grid: int, n_cycles: int,
                    d_phi_base: float, d_phi_peak: float,
                    runtime_s: float, cli_args: str,
                    F_max_source: str = "",
                    peak_lo_deg: Optional[float] = None,
                    peak_hi_deg: Optional[float] = None):
    sl = _last_cycle_slice(results)
    n_steps_last = int(results["n_steps_per_cycle"])
    cfgs = results["configs"]
    env_records = _build_envelope_records_from_results(results)
    global_status = classify_global_status(env_records)
    if global_status == "production_result":
        header = "Stage Diesel Transient THD-0 run — PRODUCTION RESULT"
    elif global_status == "aborted_outside_envelope":
        header = "Stage Diesel Transient THD-0 run — ABORTED OUTSIDE ENVELOPE"
    elif global_status == "completed_boundary_limited_result":
        header = ("Stage Diesel Transient THD-0 run — "
                   "COMPLETED BOUNDARY-LIMITED RESULT")
    else:
        header = "Stage Diesel Transient THD-0 run — PARTIAL PRODUCTION RESULT"
    # Stage Diesel Transient PeakWindow GridDiagnostic — prefer per-
    # run anisotropic grid + peak window from the results dict, fall
    # back to legacy isotropic ``grid`` and the kwargs.
    n_phi_grid = int(results.get("N_phi_grid", grid))
    n_z_grid = int(results.get("N_z_grid", grid))
    peak_lo = float(results.get(
        "peak_lo_deg",
        peak_lo_deg if peak_lo_deg is not None else 330.0))
    peak_hi = float(results.get(
        "peak_hi_deg",
        peak_hi_deg if peak_hi_deg is not None else 480.0))
    tex_diag = results.get("texture_resolution_diagnostic") or {}
    lines = [
        header,
        f"  thermal mode    : {thermal.mode}",
        f"  gamma           : {thermal.gamma_mix:.4f}",
        f"  tau_th_s        : {thermal.tau_th_s:.4f}",
        f"  T_in_C          : {thermal.T_in_C:.2f}",
        f"  cp_J_kgK        : {thermal.cp_J_kgK:.1f}",
        f"  mdot_floor_kg_s : {thermal.mdot_floor_kg_s:.2e} "
        "(mdot policy: max(rho*|Q|, mdot_floor))",
        f"  spatial grid    : {n_phi_grid} x {n_z_grid} (N_phi x N_Z)",
        f"  n_cycles        : {n_cycles}",
        f"  d_phi_base/peak : {d_phi_base}° / {d_phi_peak}°",
        f"  fine d_phi window : {peak_lo:.1f}° - {peak_hi:.1f}°",
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

    # Stage Diesel Transient PeakWindow GridDiagnostic — texture
    # pocket resolution block (independent of configs; same for all
    # textured rows since geometry + grid are run-global).
    if tex_diag:
        cpp_phi = float(tex_diag.get("cells_per_pocket_phi", float("nan")))
        cpp_z = float(tex_diag.get("cells_per_pocket_z", float("nan")))
        status_str = str(tex_diag.get("resolution_status", "unknown"))
        rec_n_phi = int(tex_diag.get("recommended_n_phi_min", 0))
        lines.append("texture resolution :")
        lines.append(
            f"  cells_per_pocket_phi : {cpp_phi:.2f}  "
            f"(N_phi={n_phi_grid})"
        )
        lines.append(
            f"  cells_per_pocket_z   : {cpp_z:.2f}  "
            f"(N_z={n_z_grid})"
        )
        lines.append(f"  resolution_status    : {status_str}")
        lines.append(f"  recommended_N_phi_min (cells>=4) : {rec_n_phi}")
        if status_str == "insufficient":
            lines.append(
                "  [WARN] texture pocket under-resolved "
                f"(cells_per_pocket_phi={cpp_phi:.2f} < 4); "
                f"increase --n-phi to >= {rec_n_phi}."
            )
        lines.append("")

    # Stage J — Ausas + groove block (echoed regardless of cavitation
    # path so a half-Sommerfeld run still tells the reader that
    # Ausas is *not* in use).
    cav_model = str(results.get("cavitation_model", "half_sommerfeld"))
    tex_kind = str(results.get("texture_kind", "dimple"))
    groove_preset_name = results.get("groove_preset")
    groove_preset_resolved = results.get("groove_preset_resolved") or {}
    groove_relief_stats_dict = results.get("groove_relief_stats") or {}
    ausas_opts = results.get("ausas_options") or {}
    fidelity_used = results.get("fidelity")
    lines.append("Stage J: Ausas dynamic + grooves")
    lines.append(f"  cavitation model      : {cav_model}")
    lines.append(f"  texture kind          : {tex_kind}")
    lines.append(
        "  groove preset         : "
        f"{groove_preset_name if groove_preset_name else 'N/A'}"
    )
    if fidelity_used:
        lines.append(f"  fidelity preset       : {fidelity_used}")
    if cav_model == "ausas_dynamic":
        lines.append(
            "  Ausas scheme          : "
            f"{ausas_opts.get('scheme', 'rb')}")
        lines.append(
            "  Ausas omega_p/theta   : "
            f"{ausas_opts.get('omega_p', 'default')} / "
            f"{ausas_opts.get('omega_theta', 'default')}")
        lines.append(
            "  Ausas tol/max_inner   : "
            f"{ausas_opts.get('tol', 'default')} / "
            f"{ausas_opts.get('max_inner', 'default')}")
        lines.append(
            "  Ausas check_every     : "
            f"{ausas_opts.get('check_every', 'default')}")
    lines.append("")

    # Groove geometry block — only emitted when the run actually
    # uses grooves so half-Sommerfeld dimple regressions stay quiet.
    if tex_kind == "groove" and groove_preset_resolved:
        gpr = groove_preset_resolved
        rs = groove_relief_stats_dict
        lines.append("groove geometry:")
        lines.append(
            "  N_branch_per_side : "
            f"{int(gpr.get('N_branch_per_side', 0))}")
        lines.append(
            f"  beta_deg          : {float(gpr.get('beta_deg', 0.0)):.1f}")
        lines.append(
            f"  d_g_um            : {float(gpr.get('d_g_um', 0.0)):.2f}")
        lines.append(
            f"  d_g_over_c        : {float(gpr.get('d_g_over_c', 0.0)):.4f}")
        lines.append(
            f"  w_g_mm            : {float(gpr.get('w_g_mm', 0.0)):.2f}")
        lines.append(
            f"  w_g_over_D        : {float(gpr.get('w_g_over_D', 0.0)):.4f}")
        lines.append(
            f"  belt_half         : {float(gpr.get('belt_half_nondim', 0.0)):.3f}")
        lines.append(
            f"  taper_ratio       : {float(gpr.get('taper_ratio', 0.0)):.3f}")
        lines.append(
            f"  ramp_frac         : {float(gpr.get('ramp_frac', 0.0)):.3f}")
        lines.append(
            f"  chirality         : {gpr.get('chirality', '?')}")
        lines.append(
            f"  coverage_mode     : {gpr.get('coverage_mode', '?')}")
        if rs:
            lines.append(
                "  relief_min/max    : "
                f"{rs.get('relief_min', float('nan')):.4e} / "
                f"{rs.get('relief_max', float('nan')):.4e}")
            lines.append(
                "  relief nonzero %  : "
                f"{100.0 * float(rs.get('relief_nonzero_frac', 0.0)):.1f}%")
        warns = []
        if rs.get("has_nan"):
            warns.append("NaN in relief")
        if rs.get("has_inf"):
            warns.append("inf in relief")
        if rs.get("relief_min", 0.0) < 0.0:
            warns.append(
                f"min < 0 ({rs['relief_min']:.3e})")
        lines.append(
            "  geometry warnings : "
            f"{', '.join(warns) if warns else 'none'}")
        lines.append("")

    if global_status == "aborted_outside_envelope":
        # All configs aborted — the legacy diagnostic header.
        first_aborted = next(
            (ic for ic, c in enumerate(cfgs)
             if bool(results["aborted"][ic])), None)
        if first_aborted is not None:
            n_done = int(results["steps_completed"][first_aborted])
            phi_at = float(results["phi_crank_deg"][min(
                n_done, len(results["phi_crank_deg"]) - 1)])
            lines.extend([
                f"abort triggered at step {n_done} (phi={phi_at:.1f} deg)",
                "This is a load-envelope diagnostic, not a THD result.",
                "",
            ])
    elif global_status == "completed_boundary_limited_result":
        # Stage Diesel Transient AngleWeighted Metrics — every config
        # ran to completion but every config sits outside the
        # applicability gate (firing-sector clamp saturation).
        lines.extend([
            "Global status: completed_boundary_limited_result",
            "",
            "Per-config:",
        ])
        for ic, cfg in enumerate(cfgs):
            vnc = np.asarray(
                results["valid_no_clamp"][ic, sl], dtype=bool)
            frac = (float(vnc.sum()) / float(n_steps_last)
                     if n_steps_last > 0 else float("nan"))
            tag = per_config_status_line(env_records[ic], frac)
            lines.append(f"  {cfg['label']:<32}  {tag}")
        lines.extend([
            "",
            "Interpretation:",
            "  All configurations completed the full transient cycle, "
            "but every",
            "  configuration is outside the full-film applicability "
            "gate. The firing",
            "  sector contains a boundary-limited regime where epsilon "
            "reaches the",
            "  clamp cap and h_min approaches the clamp floor. This is "
            "reported as a",
            "  boundary-limited diagnostic, not as a clean full-film "
            "paired comparison.",
            "  Texture mitigation in the firing peak is not separable "
            "from clamp",
            "  saturation.",
            "",
        ])
    elif global_status == "partial_production_result":
        # Mixed: some configs full, some aborted. Report per-config
        # status + an interpretation note so the downstream reader
        # doesn't conflate the two regimes (Section 2 of the patch).
        lines.extend([
            "Global status: partial_production_result",
            "",
            "Per-config:",
        ])
        for ic, cfg in enumerate(cfgs):
            vnc = np.asarray(
                results["valid_no_clamp"][ic, sl], dtype=bool)
            frac = (float(vnc.sum()) / float(n_steps_last)
                     if n_steps_last > 0 else float("nan"))
            tag = per_config_status_line(env_records[ic], frac)
            lines.append(f"  {cfg['label']:<32}  {tag}")
        lines.extend([
            "",
            "Interpretation:",
            "  Some configurations completed the full transient cycle "
            "and support",
            "  paired smooth-vs-textured comparison. Other "
            "configurations exited",
            "  the full-film transient envelope and are reported as "
            "failure-boundary",
            "  diagnostics rather than paired metrics.",
            f"  All-config paired comparison is unavailable at "
            f"F={float(results['F_max'])/1e3:.1f} kN.",
            "",
        ])
    else:
        # production_result: everyone OK. List per-config status so
        # the reader sees the near-edge band visibly.
        lines.extend([
            "Global status: production_result",
            "",
            "Per-config:",
        ])
        for ic, cfg in enumerate(cfgs):
            vnc = np.asarray(
                results["valid_no_clamp"][ic, sl], dtype=bool)
            frac = (float(vnc.sum()) / float(n_steps_last)
                     if n_steps_last > 0 else float("nan"))
            tag = per_config_status_line(env_records[ic], frac)
            lines.append(f"  {cfg['label']:<32}  {tag}")
        lines.append("")

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
        # Stage Diesel Transient ClampAccounting Fix — show
        # ``Contact steps`` (unique-step mask) and ``Contact events``
        # (predictor + substep + final, up to ~3 per step) on
        # separate lines so the legacy "events / steps" output
        # cannot be misread as a clamp-step fraction.
        ev_arr = results.get("contact_clamp_event_count")
        if ev_arr is not None:
            ev = np.asarray(ev_arr[ic, sl])
            n_clamp_events_last = int(np.nansum(ev))
        else:
            n_clamp_events_last = 0
        lines.extend([
            f"  solver_success     : {int(solver_ok.sum())}/{n_steps_last}",
            f"  valid_dynamic      : {int(valid_dyn.sum())}/{n_steps_last}",
            f"  valid_no_clamp     : {int(valid_noc.sum())}/{n_steps_last}",
            f"  Contact steps      : {int(contact.sum())}/{n_steps_last} "
            f"(unique-step mask)",
            f"  Contact events     : {n_clamp_events_last} "
            f"(predictor + substep + final, up to ~3/step)",
            f"  mdot_floor_hit     : {int(floor.sum())}/{n_steps_last}",
            f"  retry_recovered    : {int(retry_used.sum())}/{n_steps_last}",
            f"  retry_exhausted    : "
            f"{int(results['retry_exhausted_count'][ic])}",
            f"  retry omega hits   : {omega_str}",
            f"  thermal_cycle_delta: "
            f"{delta_t if np.isfinite(delta_t) else float('nan'):.3f} C",
            f"  thermal_periodic_converged: {per_conv}",
        ])
        # Stage Diesel Transient Load-Envelope-0 — envelope block.
        applicable = bool(results["applicable"][ic])
        applicable_reason = str(results["applicable_reason"][ic])
        aborted = bool(results["aborted"][ic])
        abort_reason = str(results["abort_reason"][ic])
        n_attempted = int(results["steps_attempted"][ic])
        n_completed = int(results["steps_completed"][ic])
        first_clamp = float(results["first_clamp_phi"][ic])
        first_invalid = float(results["first_invalid_phi"][ic])
        first_solver_fail = float(results["first_solver_failed_phi"][ic])
        status = "aborted_outside_envelope" if aborted else "ok"
        lines.extend([
            "  envelope:",
            f"    applicable          : "
            f"{'yes' if applicable else 'no'}",
            f"    reason              : {applicable_reason}",
            f"    status              : {status}",
            f"    abort_reason        : {abort_reason or '-'}",
            f"    steps_completed     : {n_completed}/{n_attempted}",
        ])
        # Stage Diesel Transient AngleWeighted Metrics — print both
        # count-based (legacy diagnostic) and angle-weighted (the
        # gate now keys on these) statistics so the reader sees both
        # views and can locate the count-vs-angle delta directly.
        vnc_count_last_frac = (float(valid_noc.sum()) / float(n_steps_last)
                                if n_steps_last > 0 else float("nan"))
        cc_count_last_frac = (float(contact.sum()) / float(n_steps_last)
                               if n_steps_last > 0 else float("nan"))
        lines.extend([
            "    count-based (legacy diagnostic, last cycle):",
            f"      valid_no_clamp_count_frac : "
            f"{vnc_count_last_frac:.3f} "
            f"({int(valid_noc.sum())}/{n_steps_last})",
            f"      contact_clamp_count_frac  : "
            f"{cc_count_last_frac:.3f} "
            f"({int(contact.sum())}/{n_steps_last})",
        ])
        aw_full_list = results.get("angle_weighted_full") or []
        aw_last_list = results.get("angle_weighted_last_cycle") or []
        aw_full = aw_full_list[ic] if ic < len(aw_full_list) else {}
        aw_last = aw_last_list[ic] if ic < len(aw_last_list) else {}
        if aw_full:
            lines.extend([
                "    angle-weighted (full run):",
                f"      cycle_angle_deg              : "
                f"{aw_full.get('cycle_angle_deg', float('nan')):.2f}",
                f"      valid_no_clamp_angle_deg     : "
                f"{aw_full.get('valid_no_clamp_angle_deg', float('nan')):.2f}",
                f"      valid_no_clamp_angle_frac    : "
                f"{aw_full.get('valid_no_clamp_angle_frac', float('nan')):.3f}",
                f"      contact_angle_deg            : "
                f"{aw_full.get('contact_angle_deg', float('nan')):.2f}",
                f"      contact_angle_frac           : "
                f"{aw_full.get('contact_angle_frac', float('nan')):.3f}",
            ])
        if aw_last:
            lines.extend([
                "    angle-weighted (last cycle):",
                f"      cycle_angle_deg              : "
                f"{aw_last.get('cycle_angle_deg', float('nan')):.2f}",
                f"      valid_no_clamp_angle_deg     : "
                f"{aw_last.get('valid_no_clamp_angle_deg', float('nan')):.2f}",
                f"      valid_no_clamp_angle_frac    : "
                f"{aw_last.get('valid_no_clamp_angle_frac', float('nan')):.3f}",
                f"      contact_angle_deg            : "
                f"{aw_last.get('contact_angle_deg', float('nan')):.2f}",
                f"      contact_angle_frac           : "
                f"{aw_last.get('contact_angle_frac', float('nan')):.3f}",
            ])
            fs = results.get("firing_sector_deg", (340.0, 480.0))
            lines.extend([
                f"    angle-weighted (firing sector "
                f"{fs[0]:.1f}-{fs[1]:.1f}°, last cycle):",
                f"      firing_angle_deg               : "
                f"{aw_last.get('firing_angle_deg', float('nan')):.2f}",
                f"      valid_no_clamp_angle_firing_deg: "
                f"{aw_last.get('valid_no_clamp_angle_firing_deg', float('nan')):.2f}",
                f"      valid_no_clamp_angle_frac      : "
                f"{aw_last.get('valid_no_clamp_angle_firing_frac', float('nan')):.3f}",
                f"      contact_angle_firing_deg       : "
                f"{aw_last.get('contact_angle_firing_deg', float('nan')):.2f}",
                f"      contact_angle_frac             : "
                f"{aw_last.get('contact_angle_firing_frac', float('nan')):.3f}",
            ])
        lines.extend([
            f"    first_clamp_phi     : "
            f"{first_clamp if np.isfinite(first_clamp) else float('nan'):.1f} deg",
            f"    first_invalid_phi   : "
            f"{first_invalid if np.isfinite(first_invalid) else float('nan'):.1f} deg",
            f"    first_solver_fail_phi: "
            f"{first_solver_fail if np.isfinite(first_solver_fail) else float('nan'):.1f} deg",
            "",
        ])

        # Stage J — per-config Ausas diagnostics. Quiet on the
        # legacy half-Sommerfeld path (zero-filled arrays render as
        # 100% converged with cav_frac=0; we suppress them so the
        # block stays meaningful).
        if cav_model == "ausas_dynamic":
            ausas_conv = np.asarray(
                results.get("ausas_converged",
                                np.zeros_like(valid_dyn))[ic, sl],
                dtype=bool)
            ausas_n_inner_arr = np.asarray(
                results.get("ausas_n_inner",
                                np.zeros_like(valid_dyn,
                                                  dtype=np.int32))[ic, sl])
            ausas_cav_arr = np.asarray(
                results.get("ausas_cav_frac",
                                np.zeros_like(valid_dyn,
                                                  dtype=float))[ic, sl])
            ausas_tmin_arr = np.asarray(
                results.get("ausas_theta_min",
                                np.ones_like(valid_dyn,
                                                 dtype=float))[ic, sl])
            ausas_tmax_arr = np.asarray(
                results.get("ausas_theta_max",
                                np.ones_like(valid_dyn,
                                                 dtype=float))[ic, sl])
            n_attempted_ic = max(int(ausas_conv.size), 1)
            conv_frac = (float(ausas_conv.sum())
                         / float(n_attempted_ic))
            n_inner_finite = ausas_n_inner_arr[
                np.isfinite(ausas_n_inner_arr)]
            p50 = float(np.percentile(n_inner_finite, 50)) \
                if n_inner_finite.size else float("nan")
            p95 = float(np.percentile(n_inner_finite, 95)) \
                if n_inner_finite.size else float("nan")
            p_max_inner = (float(np.max(n_inner_finite))
                            if n_inner_finite.size else float("nan"))
            cav_finite = ausas_cav_arr[np.isfinite(ausas_cav_arr)]
            cav_mean = float(np.mean(cav_finite)) \
                if cav_finite.size else float("nan")
            cav_max = float(np.max(cav_finite)) \
                if cav_finite.size else float("nan")
            tmin_arr_finite = ausas_tmin_arr[
                np.isfinite(ausas_tmin_arr)]
            tmax_arr_finite = ausas_tmax_arr[
                np.isfinite(ausas_tmax_arr)]
            theta_min = (float(np.min(tmin_arr_finite))
                         if tmin_arr_finite.size else float("nan"))
            theta_max = (float(np.max(tmax_arr_finite))
                         if tmax_arr_finite.size else float("nan"))
            lines.extend([
                "  Ausas diagnostics:",
                f"    converged fraction       : {conv_frac:.3f}",
                f"    n_inner p50/p95/max      : "
                f"{p50:.0f} / {p95:.0f} / {p_max_inner:.0f}",
                f"    cav_frac mean/max        : "
                f"{cav_mean:.3f} / {cav_max:.3f}",
                f"    theta min/max            : "
                f"{theta_min:.3f} / {theta_max:.3f}",
                f"    state_reset_count        : "
                f"{int(results.get('ausas_state_reset_count', np.zeros(len(cfgs)))[ic])}",
                f"    failed_step_count        : "
                f"{int(results.get('ausas_failed_step_count', np.zeros(len(cfgs)))[ic])}",
                f"    rejected_commit_count    : "
                f"{int(results.get('ausas_rejected_commit_count', np.zeros(len(cfgs)))[ic])}",
                "",
            ])

    # Paired comparison block.
    paired = results.get("paired_comparison") or []
    lines.append("=" * 60)
    if global_status == "completed_boundary_limited_result":
        # Stage Diesel Transient AngleWeighted Metrics — when every
        # config is outside the applicability gate, the smooth-vs-
        # textured numbers below are conditional diagnostics, not a
        # clean full-film paired comparison. Make this explicit in
        # the block header so downstream readers don't conflate the
        # regimes.
        lines.append(
            "Conditional paired diagnostics "
            "(common_valid_no_clamp mask only — same time steps both "
            "sides; boundary-limited regime, not clean full-film "
            "comparison)"
        )
    else:
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

    # ── Per-config production metrics block ────────────────────────
    prod = results.get("production_metrics") or []
    fs = results.get("firing_sector_deg", (340.0, 480.0))
    if prod:
        lines.extend([
            "=" * 60,
            "Per-config production transient metrics (last cycle)",
            f"firing sector: {fs[0]:.1f}-{fs[1]:.1f} deg",
            "=" * 60,
        ])
        for ic, cfg in enumerate(cfgs):
            rec = prod[ic]
            lines.extend([
                f"[{cfg['label']}]",
                "  pressure (firing sector, valid_no_clamp):",
                f"    p95 p_max : "
                f"{rec.get('pmax_firing_p95', float('nan'))/1e6:.2f} MPa",
                f"    p99 p_max : "
                f"{rec.get('pmax_firing_p99', float('nan'))/1e6:.2f} MPa",
                f"    max p_max : "
                f"{rec.get('pmax_firing_max', float('nan'))/1e6:.2f} MPa  "
                f"(n_steps={int(rec.get('pmax_firing_count', 0))})",
                "  film thickness (last cycle, valid_no_clamp):",
                f"    P5 h_min  : "
                f"{rec.get('hmin_p5', float('nan'))*1e6:.2f} um",
                f"    min h_min : "
                f"{rec.get('hmin_min', float('nan'))*1e6:.2f} um",
                f"    steps below 10/8/6 um : "
                f"{int(rec.get('steps_hmin_below_10um',0))}/"
                f"{int(rec.get('steps_hmin_below_8um',0))}/"
                f"{int(rec.get('steps_hmin_below_6um',0))}",
                f"    angle below 10/8/6 um : "
                f"{rec.get('angle_hmin_below_10um',0):.1f}/"
                f"{rec.get('angle_hmin_below_8um',0):.1f}/"
                f"{rec.get('angle_hmin_below_6um',0):.1f} deg",
                "  orbit (last cycle, valid_dynamic):",
                f"    max |eps|         : "
                f"{rec.get('max_eps_lastcycle', float('nan')):.4f} "
                f"@ phi={rec.get('phi_at_max_eps', float('nan')):.1f} deg",
                f"    eps at phi=421°   : "
                f"{rec.get('eps_at_phi_421', float('nan')):.4f}",
                f"    recovery to 0.90  : "
                f"{rec.get('angle_recovery_to_0p9', float('nan')):.1f} deg "
                f"(failed={rec.get('recovery_failed_0p9', False)})",
                f"    recovery to 0.85  : "
                f"{rec.get('angle_recovery_to_0p85', float('nan')):.1f} deg "
                f"(failed={rec.get('recovery_failed_0p85', False)})",
                f"    recovery to 0.80  : "
                f"{rec.get('angle_recovery_to_0p8', float('nan')):.1f} deg "
                f"(failed={rec.get('recovery_failed_0p8', False)})",
                f"    AUC eps on 360-480°: "
                f"{rec.get('auc_eps_360_480', float('nan')):.2f} deg",
                "  power loss (firing sector, valid_no_clamp):",
                f"    impulse  : "
                f"{rec.get('ploss_impulse_firing_J', float('nan')):.3f} J",
                f"    mean P_loss : "
                f"{rec.get('ploss_firing_mean_W', float('nan')):.1f} W",
                f"    max P_loss  : "
                f"{rec.get('ploss_firing_max_W', float('nan')):.1f} W",
                "",
            ])

    # ── Paired transient metrics (extended) ─────────────────────────
    paired_ext = results.get("paired_extended") or []
    if paired_ext:
        lines.extend([
            "=" * 60,
            "Paired transient metrics "
            "(common_valid_no_clamp mask only)",
            f"firing sector: {fs[0]:.1f}-{fs[1]:.1f} deg",
            "=" * 60,
        ])
        for r in paired_ext:
            lines.extend([
                f"[{r['oil_name']}]",
                f"  smooth        : {r['smooth_label']}",
                f"  textured      : {r['textured_label']}",
                f"  common_valid_no_clamp count : "
                f"{r['common_valid_no_clamp_count']}",
                f"  common_valid_dynamic count  : "
                f"{r['common_valid_dynamic_count']}",
                "",
                "  Pressure mitigation (firing sector):",
                f"    p95 p_max  smooth/textured: "
                f"{r['smooth_pmax_firing_p95']/1e6:.2f} / "
                f"{r['textured_pmax_firing_p95']/1e6:.2f}  "
                f"(delta = {r['delta_pmax_firing_p95']/1e6:+.2f} MPa)",
                f"    p99 p_max  smooth/textured: "
                f"{r['smooth_pmax_firing_p99']/1e6:.2f} / "
                f"{r['textured_pmax_firing_p99']/1e6:.2f}  "
                f"(delta = {r['delta_pmax_firing_p99']/1e6:+.2f} MPa)",
                f"    max p_max  smooth/textured: "
                f"{r['smooth_pmax_firing_max']/1e6:.2f} / "
                f"{r['textured_pmax_firing_max']/1e6:.2f}  "
                f"(delta = {r['delta_pmax_firing_max']/1e6:+.2f} MPa)",
                "",
                "  Film thickness:",
                f"    P5 h_min  smooth/textured : "
                f"{r['smooth_hmin_p5']*1e6:.2f} / "
                f"{r['textured_hmin_p5']*1e6:.2f} um  "
                f"(delta = {r['delta_hmin_p5']*1e6:+.2f} um)",
                f"    min h_min smooth/textured : "
                f"{r['smooth_hmin_min']*1e6:.2f} / "
                f"{r['textured_hmin_min']*1e6:.2f} um  "
                f"(delta = {r['delta_hmin_min']*1e6:+.2f} um)",
                f"    steps below 10 um  smooth/textured: "
                f"{r['smooth_steps_hmin_below_10um']} / "
                f"{r['textured_steps_hmin_below_10um']}  "
                f"(delta = {r['delta_steps_hmin_below_10um']:+d})",
                f"    steps below 8 um   smooth/textured: "
                f"{r['smooth_steps_hmin_below_8um']} / "
                f"{r['textured_steps_hmin_below_8um']}  "
                f"(delta = {r['delta_steps_hmin_below_8um']:+d})",
                f"    steps below 6 um   smooth/textured: "
                f"{r['smooth_steps_hmin_below_6um']} / "
                f"{r['textured_steps_hmin_below_6um']}  "
                f"(delta = {r['delta_steps_hmin_below_6um']:+d})",
                "",
                "  Orbit dynamics:",
                f"    max |eps|     smooth/textured: "
                f"{r['smooth_max_eps_lastcycle']:.4f} / "
                f"{r['textured_max_eps_lastcycle']:.4f}  "
                f"(delta = {r['delta_max_eps_lastcycle']:+.4f})",
                f"    eps at 421°   smooth/textured: "
                f"{r['smooth_eps_at_phi_421']:.4f} / "
                f"{r['textured_eps_at_phi_421']:.4f}  "
                f"(delta = {r['delta_eps_at_phi_421']:+.4f})",
                f"    AUC eps 360-480°  smooth/textured: "
                f"{r['smooth_auc_eps_360_480']:.2f} / "
                f"{r['textured_auc_eps_360_480']:.2f} deg  "
                f"(delta = {r['delta_auc_eps_360_480']:+.2f} deg)",
                "",
                "  Power loss in firing sector:",
                f"    impulse smooth/textured (J): "
                f"{r['smooth_ploss_impulse_firing_J']:.3f} / "
                f"{r['textured_ploss_impulse_firing_J']:.3f}  "
                f"(delta = {r['delta_ploss_impulse_firing_J']:+.3f} J)",
                f"    mean    smooth/textured (W): "
                f"{r['smooth_ploss_firing_mean_W']:.1f} / "
                f"{r['textured_ploss_firing_mean_W']:.1f}  "
                f"(delta = {r['delta_ploss_firing_mean_W']:+.1f} W)",
                "",
            ])

    with open(os.path.join(run_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


_ENVELOPE_CSV_COLUMNS = (
    "F_scale", "F_max_used", "config",
    "n_steps_attempted", "n_steps_completed",
    "solver_success_count", "valid_dynamic_count",
    "valid_no_clamp_count",
    "contact_clamp_count", "solver_failed_count",
    "retry_recovered_count", "retry_exhausted_count",
    "first_clamp_phi", "first_invalid_phi",
    "max_epsilon", "min_h_min_valid",
    "p95_pmax_valid", "p99_pmax_valid", "max_pmax_valid",
    "min_T_eff_valid", "mean_T_eff_valid", "max_T_eff_valid",
    "mean_P_loss_valid", "mdot_floor_hit_count",
    "thermal_cycle_delta",
    "aborted", "abort_reason",
    "applicable", "applicable_reason",
)


def _envelope_rows_from_results(scale, F_max_used, results) -> List[Dict[str, Any]]:
    """One row per config for the load-envelope CSV (Section 2)."""
    cfgs = results["configs"]
    rows = []
    for ic, cfg in enumerate(cfgs):
        n_attempted = int(results["steps_attempted"][ic])
        n_completed = int(results["steps_completed"][ic])
        sl = slice(0, n_completed) if n_completed > 0 else slice(0, 0)
        sol_ok = np.asarray(results["solver_success"])[ic, sl]
        vd = np.asarray(results["valid_dynamic"])[ic, sl]
        vnc = np.asarray(results["valid_no_clamp"])[ic, sl]
        cc = np.asarray(results["contact_clamp"])[ic, sl]
        eps = np.sqrt(
            np.asarray(results["eps_x"])[ic, sl] ** 2
            + np.asarray(results["eps_y"])[ic, sl] ** 2
        )
        hmin = np.asarray(results["hmin"])[ic, sl]
        pmax = np.asarray(results["pmax"])[ic, sl]
        T_eff = np.asarray(results["T_eff"])[ic, sl]
        P_loss = np.asarray(results["P_loss"])[ic, sl]
        mdot_fh = np.asarray(results["mdot_floor_hit"])[ic, sl]
        valid_mask = vd & np.isfinite(hmin) & np.isfinite(pmax)

        def _pct(a, q):
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan")
            return float(np.percentile(a, q))

        def _safe_min(a, mask):
            v = a[mask]
            v = v[np.isfinite(v)]
            return float(v.min()) if v.size else float("nan")

        def _safe_max(a, mask):
            v = a[mask]
            v = v[np.isfinite(v)]
            return float(v.max()) if v.size else float("nan")

        def _safe_mean(a, mask):
            v = a[mask]
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")

        rows.append({
            "F_scale": float(scale),
            "F_max_used": float(F_max_used),
            "config": cfg["label"],
            "n_steps_attempted": n_attempted,
            "n_steps_completed": n_completed,
            "solver_success_count": int(sol_ok.sum()),
            "valid_dynamic_count": int(vd.sum()),
            "valid_no_clamp_count": int(vnc.sum()),
            "contact_clamp_count": int(cc.sum()),
            "solver_failed_count": int(results["solver_failed_count"][ic]),
            "retry_recovered_count": int(
                results["retry_recovered_count"][ic]),
            "retry_exhausted_count": int(
                results["retry_exhausted_count"][ic]),
            "first_clamp_phi": float(results["first_clamp_phi"][ic]),
            "first_invalid_phi": float(results["first_invalid_phi"][ic]),
            "max_epsilon": float(np.nanmax(eps)) if eps.size else float("nan"),
            "min_h_min_valid": _safe_min(hmin, valid_mask),
            "p95_pmax_valid": _pct(pmax[valid_mask], 95)
                if valid_mask.any() else float("nan"),
            "p99_pmax_valid": _pct(pmax[valid_mask], 99)
                if valid_mask.any() else float("nan"),
            "max_pmax_valid": _safe_max(pmax, valid_mask),
            "min_T_eff_valid": _safe_min(T_eff, valid_mask),
            "mean_T_eff_valid": _safe_mean(T_eff, valid_mask),
            "max_T_eff_valid": _safe_max(T_eff, valid_mask),
            "mean_P_loss_valid": _safe_mean(P_loss, valid_mask),
            "mdot_floor_hit_count": int(mdot_fh.sum()),
            "thermal_cycle_delta": float(
                results["thermal_cycle_delta"][ic]),
            "aborted": bool(results["aborted"][ic]),
            "abort_reason": str(results["abort_reason"][ic]),
            "applicable": bool(results["applicable"][ic]),
            "applicable_reason": str(results["applicable_reason"][ic]),
        })
    return rows


def _write_envelope_csv(sweep_root: str,
                         rows: List[Dict[str, Any]]) -> None:
    import csv as _csv
    path = os.path.join(sweep_root, "load_envelope.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(_ENVELOPE_CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _ENVELOPE_CSV_COLUMNS})


def _regenerate_summary_from_dir(run_dir: str) -> None:
    """Rewrite summary.txt from an existing data.npz / data_partial.npz
    in ``run_dir``. Does NOT recompute physics or touch the npz file.

    Used by ``--regenerate-summary`` to refresh wording (e.g. after a
    summary writer change like Stage Transient Summary Wording Fix)
    without re-running the solver.
    """
    if not os.path.isdir(run_dir):
        raise SystemExit(f"--regenerate-summary: directory not found: "
                          f"{run_dir!r}")
    npz_full = os.path.join(run_dir, "data.npz")
    npz_part = os.path.join(run_dir, "data_partial.npz")
    if os.path.isfile(npz_full):
        npz_path = npz_full
    elif os.path.isfile(npz_part):
        npz_path = npz_part
    else:
        raise SystemExit(
            f"--regenerate-summary: no data.npz or data_partial.npz "
            f"found in {run_dir!r}")
    with np.load(npz_path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    labels = list(data.get("labels", []))
    textured_flags = list(data.get("textured_flags",
                                       [False] * len(labels)))
    # Rebuild a thin configs list — only label / textured / oil-name
    # are needed by the summary writer + classifier; oil dict is
    # only used for paired_extended which already lives in npz when
    # present (we don't recompute it).
    cfgs = []
    for ic, lbl in enumerate(labels):
        textured = bool(textured_flags[ic]
                          if ic < len(textured_flags) else False)
        # Heuristic oil name: derive from label substring so paired
        # smooth-vs-textured grouping still works for legacy npz.
        oil_name = ("rapeseed" if "рапс" in str(lbl).lower()
                                  or "rapeseed" in str(lbl).lower()
                                  else "mineral")
        cfgs.append({
            "label": str(lbl), "textured": textured,
            "color": "blue" if not textured else "red",
            "ls": "-" if not textured else "--",
            "oil": {"name": oil_name},
        })

    # Reconstruct a results dict that _write_summary can consume.
    # Per-step arrays live in npz directly.
    def _arr(name, default=None):
        if name in data:
            return data[name]
        return default

    results = dict(
        configs=cfgs,
        phi_crank_deg=_arr("phi_crank_deg"),
        phi_last=_arr("phi_last"),
        last_start=int(np.asarray(_arr("last_start", 0)).item()),
        n_steps_per_cycle=int(
            np.asarray(_arr("n_steps_per_cycle", 0)).item()),
        eps_x=_arr("eps_x"), eps_y=_arr("eps_y"),
        hmin=_arr("hmin"), pmax=_arr("pmax"),
        f=_arr("f"), F_tr=_arr("F_tr"),
        N_loss=_arr("N_loss"),
        T_eff_used=_arr("T_eff_used"), T_eff=_arr("T_eff"),
        T_target=_arr("T_target"),
        eta_eff=_arr("eta_eff"),
        P_loss=_arr("P_loss"), Q=_arr("Q"), mdot=_arr("mdot"),
        mdot_floor_hit=_arr("mdot_floor_hit"),
        solver_success=_arr("solver_success"),
        valid_dynamic=_arr("valid_dynamic"),
        valid_no_clamp=_arr("valid_no_clamp"),
        contact_clamp=_arr("contact_clamp"),
        contact_clamp_event_count=_arr("contact_clamp_event_count"),
        d_phi_per_step=_arr("d_phi_per_step"),
        phi_mod_per_step=_arr("phi_mod_per_step"),
        angle_weighted_full=(
            list(_arr("angle_weighted_full", []))
            if _arr("angle_weighted_full") is not None else []),
        angle_weighted_last_cycle=(
            list(_arr("angle_weighted_last_cycle", []))
            if _arr("angle_weighted_last_cycle") is not None else []),
        retry_used=_arr("retry_used"),
        retry_omega_used=_arr("retry_omega_used"),
        contact_clamp_count=_arr("contact_clamp_count"),
        solver_failed_count=_arr("solver_failed_count"),
        retry_recovered_count=_arr("retry_recovered_count"),
        retry_exhausted_count=_arr("retry_exhausted_count"),
        thermal_cycle_delta=_arr("thermal_cycle_delta"),
        thermal_periodic_converged=_arr("thermal_periodic_converged"),
        aborted=_arr("aborted"),
        abort_reason=_arr("abort_reason"),
        first_clamp_phi=_arr("first_clamp_phi"),
        first_solver_failed_phi=_arr("first_solver_failed_phi"),
        first_invalid_phi=_arr("first_invalid_phi"),
        steps_attempted=_arr("steps_attempted"),
        steps_completed=_arr("steps_completed"),
        applicable=_arr("applicable"),
        applicable_reason=_arr("applicable_reason"),
        F_max=float(np.asarray(_arr("F_max", 0.0)).item()),
        # Production metrics + paired_extended are not stored in
        # legacy npz; if absent, the writer skips those blocks.
        production_metrics=[],
        paired_extended=[],
        paired_comparison=[],
        firing_sector_deg=(340.0, 480.0),
        envelope_abort_config={"save_partial_on_abort": True},
    )

    # Reconstruct minimal thermal + retry stubs.
    from models.thermal_coupling import ThermalConfig
    from models.diesel_quasistatic import SolverRetryConfig
    thermal = ThermalConfig(
        mode=str(np.asarray(_arr("thermal_mode", "global_relax")).item()),
        T_in_C=float(np.asarray(_arr("T_in_C", 105.0)).item()),
        gamma_mix=float(np.asarray(_arr("gamma", 0.7)).item()),
        cp_J_kgK=float(np.asarray(_arr("cp_J_kgK", 2000.0)).item()),
        mdot_floor_kg_s=float(
            np.asarray(_arr("mdot_floor_kg_s", 1e-4)).item()),
        tau_th_s=float(np.asarray(_arr("tau_th_s", 0.5)).item()),
    )
    retry_cfg = SolverRetryConfig()  # exact retry settings unrecorded
    grid_arr = _arr("grid")
    grid = (int(np.asarray(grid_arr).ravel()[0])
             if grid_arr is not None else 0)
    _write_summary(
        run_dir, results, thermal, retry_cfg,
        grid=grid, n_cycles=0,
        d_phi_base=float("nan"), d_phi_peak=float("nan"),
        runtime_s=float("nan"),
        cli_args="(regenerated from data.npz)",
        F_max_source="(regenerated)",
        peak_lo_deg=None,
        peak_hi_deg=None,
    )
    print(f"summary.txt regenerated in {run_dir}")


def classify_global_status(envelope_records) -> str:
    """Stage Transient Summary Wording Fix + ClampAccounting Fix +
    Stage Diesel Transient AngleWeighted Metrics.

    ``envelope_records``: list of per-config dicts with at least a
    ``status`` and ``applicable`` field. Returns one of:

      * ``production_result``                — every config applicable=True
      * ``aborted_outside_envelope``         — every config aborted
      * ``completed_boundary_limited_result`` — every config ran to
        completion but every config is outside the applicability gate
        (e.g. valid_no_clamp_angle_frac < 0.85). Distinguishes the
        boundary-limited regime from a clean production_result so
        the header does not over-promise.
      * ``partial_production_result``        — any other mixture
        (some applicable, some aborted/outside)
    """
    if not envelope_records:
        return "production_result"
    n_total = len(envelope_records)

    def _aborted(rec) -> bool:
        return (str(rec.get("status", "unknown"))
                == "aborted_outside_envelope")

    def _applicable(rec) -> bool:
        # An aborted config is never applicable, regardless of any
        # ``applicable`` field the caller may have left over.
        if _aborted(rec):
            return False
        # Otherwise default to True for backward-compat with legacy
        # records that only carry ``status``.
        return bool(rec.get("applicable", True))

    def _completed_outside_gate(rec) -> bool:
        return (not _aborted(rec)) and (not _applicable(rec))

    n_applicable = sum(1 for rec in envelope_records if _applicable(rec))
    n_aborted = sum(1 for rec in envelope_records if _aborted(rec))
    n_completed_outside = sum(
        1 for rec in envelope_records if _completed_outside_gate(rec))
    if n_applicable == n_total:
        return "production_result"
    if n_aborted == n_total:
        return "aborted_outside_envelope"
    if n_completed_outside == n_total:
        return "completed_boundary_limited_result"
    return "partial_production_result"


def per_config_status_line(env_record,
                             valid_no_clamp_frac: float) -> str:
    """One-line per-config envelope status string for the global
    header.

    * ``aborted_outside_envelope`` if the config aborted (overrides
      everything else).
    * ``full / outside applicability gate`` if status=="ok" but the
      envelope's ``applicable`` flag is False (Stage Diesel Transient
      ClampAccounting Fix — distinguishes a completed-but-not-
      applicable run from a near-edge applicable one).
    * ``full / applicable`` if status=="ok", applicable=True and
      valid_no_clamp_frac >= 0.95.
    * ``full / near-edge applicable`` if status=="ok",
      applicable=True but valid_no_clamp_frac < 0.95.
    """
    status = str(env_record.get("status", "unknown"))
    if status == "aborted_outside_envelope":
        return "aborted_outside_envelope"
    applicable = bool(env_record.get("applicable", True))
    if not applicable:
        return "full / outside applicability gate"
    if not np.isfinite(valid_no_clamp_frac):
        return "full / unknown"
    if float(valid_no_clamp_frac) < 0.95:
        return "full / near-edge applicable"
    return "full / applicable"


def _build_envelope_records_from_results(results) -> List[Dict[str, Any]]:
    """Adapter: produce the list of dicts ``classify_global_status``
    expects from a ``run_transient`` results dict.

    Stage Diesel Transient ClampAccounting Fix — also surfaces the
    per-config ``applicable`` boolean so the global classifier can
    distinguish "completed but outside applicability gate" from a
    full production result.
    """
    cfgs = results.get("configs", [])
    n_cfg = len(cfgs)
    aborted = np.asarray(results.get("aborted",
                                         np.zeros(n_cfg, dtype=bool)))
    applicable_arr = np.asarray(
        results.get("applicable", np.ones(n_cfg, dtype=bool)))
    out = []
    for ic, cfg in enumerate(cfgs):
        is_ab = bool(aborted[ic])
        out.append({
            "config": cfg.get("label", f"cfg_{ic}"),
            "status": ("aborted_outside_envelope" if is_ab else "ok"),
            "applicable": (False if is_ab else bool(applicable_arr[ic])),
        })
    return out


def _run_one(thermal: ThermalConfig, retry_cfg: SolverRetryConfig,
              args, configs, out_base,
              *,
              envelope_abort: Optional[EnvelopeAbortConfig] = None,
              return_results: bool = False):
    F_max_used = float(getattr(args, "F_max_resolved", 0.0)) or None
    F_max_source = getattr(args, "F_max_source", "")
    extra_tag = ""
    if getattr(args, "F_max_debug", False):
        extra_tag = "Fmaxdebug"
    if not retry_cfg.enabled:
        extra_tag = (extra_tag + "_no_retry").strip("_")
    # Sweep mode lands every scale in its own Fscale_<N>p<MM> folder
    # under a parent _load_envelope dir (out_base is already the parent).
    scale_for_dir = getattr(args, "_scale_for_dir", None)
    if scale_for_dir is not None:
        run_dir = os.path.join(
            out_base,
            f"Fscale_{scale_for_dir:.2f}".replace(".", "p"),
        )
        os.makedirs(run_dir, exist_ok=True)
    else:
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
    # Stage J — assemble ausas_options from any --ausas-* CLI flags.
    ausas_options: Dict[str, Any] = {}
    for src_attr, kw in (
        ("ausas_omega_p", "omega_p"),
        ("ausas_omega_theta", "omega_theta"),
        ("ausas_tol", "tol"),
        ("ausas_max_inner", "max_inner"),
        ("ausas_check_every", "check_every"),
        ("ausas_scheme", "scheme"),
    ):
        v = getattr(args, src_attr, None)
        if v is not None:
            ausas_options[kw] = v
    results = run_transient(
        F_max=F_max_used,
        debug=False,
        thermal=thermal,
        configs=configs,
        cavitation=str(args.cavitation),
        n_grid=args.n_grid,
        n_cycles=args.n_cycles,
        d_phi_base_deg=args.d_phi_base,
        d_phi_peak_deg=args.d_phi_peak,
        peak_lo_deg=float(args.peak_lo_deg),
        peak_hi_deg=float(args.peak_hi_deg),
        n_phi_grid=args.n_phi,
        n_z_grid=args.n_z,
        retry_config=retry_cfg,
        envelope_abort=envelope_abort,
        firing_sector_deg=getattr(args, "firing_sector_resolved", None),
        texture_kind=str(args.texture_kind),
        groove_preset=args.groove_preset,
        fidelity=args.fidelity,
        ausas_options=(ausas_options if ausas_options else None),
        save_field_checkpoints=bool(args.save_field_checkpoints),
        debug_first_steps=int(getattr(args, "debug_first_steps", 0) or 0),
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
                    F_max_source=F_max_source,
                    peak_lo_deg=float(args.peak_lo_deg),
                    peak_hi_deg=float(args.peak_hi_deg))

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
    _plot_orbit_lastcycle(results, run_dir)
    _plot_valid_status(results, run_dir)
    _plot_retry_status(results, run_dir)

    print(f"  done in {dt:.1f}s")
    if return_results:
        return run_dir, results, dt
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
    # Stage Diesel Transient PeakWindow GridDiagnostic.
    pa.add_argument("--peak-lo-deg", dest="peak_lo_deg",
                    type=float, default=330.0,
                    help="Lower bound of the fine-d_phi window (deg). "
                         "Default 330.")
    pa.add_argument("--peak-hi-deg", dest="peak_hi_deg",
                    type=float, default=480.0,
                    help="Upper bound of the fine-d_phi window (deg). "
                         "Default 480 covers the full production "
                         "metrics window 340°-480° including post-"
                         "peak recovery; legacy default was 420°.")
    pa.add_argument("--n-phi", dest="n_phi", type=int, default=None,
                    help="Circumferential grid resolution Nφ. If "
                         "unspecified, falls back to --n-grid (legacy "
                         "isotropic). Use a larger Nφ to resolve the "
                         "texture pocket (cells_per_pocket >= 4).")
    pa.add_argument("--n-z", dest="n_z", type=int, default=None,
                    help="Axial grid resolution N_Z. If unspecified, "
                         "falls back to --n-grid (legacy isotropic).")
    pa.add_argument("--cavitation", default="half_sommerfeld",
                    choices=["half_sommerfeld", "ausas_dynamic"],
                    help="Cavitation closure. 'half_sommerfeld' "
                         "(default, legacy) goes through "
                         "solve_reynolds; 'ausas_dynamic' (Stage J) "
                         "wires the dynamic Ausas JFO solver via "
                         "the per-config DieselAusasState adapter.")
    # Stage J — texture / groove / fidelity / ausas-options.
    pa.add_argument("--texture-kind", dest="texture_kind",
                    default="dimple",
                    choices=["dimple", "groove", "none"],
                    help="Texture family: 'dimple' (legacy default), "
                         "'groove' (Stage J ramped/tapered "
                         "herringbone), or 'none' (smooth even "
                         "for textured configs).")
    pa.add_argument("--groove-preset", dest="groove_preset",
                    default=None,
                    help="Named groove preset (Stage J). Required "
                         "when --texture-kind=groove. See "
                         "config.diesel_groove_presets.GROOVE_PRESETS.")
    pa.add_argument("--fidelity", dest="fidelity", default=None,
                    choices=["low", "medium", "high", "custom"],
                    help="Stage J fidelity preset. Sets defaults "
                         "for n-phi/n-z/n-cycles/d-phi-base/peak. "
                         "Explicit CLI flags override the preset.")
    pa.add_argument("--ausas-omega-p", dest="ausas_omega_p",
                    type=float, default=None)
    pa.add_argument("--ausas-omega-theta", dest="ausas_omega_theta",
                    type=float, default=None)
    pa.add_argument("--ausas-tol", dest="ausas_tol",
                    type=float, default=None)
    pa.add_argument("--ausas-max-inner", dest="ausas_max_inner",
                    type=int, default=None)
    pa.add_argument("--ausas-check-every", dest="ausas_check_every",
                    type=int, default=None)
    pa.add_argument("--ausas-scheme", dest="ausas_scheme",
                    choices=["rb", "jacobi"], default=None)
    pa.add_argument("--save-field-checkpoints",
                    dest="save_field_checkpoints",
                    action="store_true", default=False,
                    help="Save peak-pressure / min-hmin / max-eps "
                         "field checkpoints (Stage J).")
    pa.add_argument("--debug-first-steps", dest="debug_first_steps",
                    type=int, default=0,
                    help="Stage J followup §4 — when running with "
                         "--cavitation ausas_dynamic, print per-step "
                         "diagnostics (phi, dt_s, dt_tau, eps, F_ext, "
                         "F_hyd, dot_norm, p_nd_max, p_dim_max, theta, "
                         "residual, n_inner, converged) for each Verlet "
                         "trial substep AND the accepted commit, for "
                         "the first N steps. Default 0 (off). "
                         "Recommended N=5 for smoke debugging.")
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
    # Stage Diesel Transient Load-Envelope-0 — abort + sweep policy.
    pa.add_argument("--F-max-scale-sweep", dest="F_max_scale_sweep",
                    default=None,
                    help="Comma-separated list of F_max scales for a "
                         "load-envelope sweep, e.g. "
                         "'0.3,0.5,0.7,0.85,1.0'. Overrides "
                         "--F-max-scale. Each scale lands in its own "
                         "subfolder; sweep aggregator writes "
                         "load_envelope.csv at the parent directory.")
    pa.add_argument("--abort-on-clamp-frac", dest="abort_clamp_frac",
                    type=float, default=0.30,
                    help="Abort a config if the clamp fraction so far "
                         "exceeds this threshold. Default 0.30.")
    pa.add_argument("--abort-on-solver-fail-frac",
                    dest="abort_solver_fail_frac", type=float,
                    default=0.30,
                    help="Abort a config if the solver-failed fraction "
                         "so far exceeds this threshold. Default 0.30.")
    pa.add_argument("--abort-after-consecutive-invalid",
                    dest="abort_consec_invalid", type=int, default=30,
                    help="Abort a config after this many consecutive "
                         "not-valid_dynamic steps. Default 30.")
    pa.add_argument("--save-partial-on-abort", dest="save_partial",
                    action="store_true", default=True,
                    help="On abort, write data_partial.npz with the "
                         "completed-step prefix instead of dropping it. "
                         "Default ON.")
    pa.add_argument("--no-save-partial-on-abort",
                    dest="save_partial", action="store_false",
                    help="Override: do NOT save partial data on abort.")
    pa.add_argument("--no-envelope-abort", dest="no_envelope_abort",
                    action="store_true",
                    help="Disable the envelope-abort policy entirely "
                         "(legacy 'run to the end no matter what').")
    pa.add_argument("--firing-sector", dest="firing_sector",
                    default=None,
                    help="Firing sector in crank-angle degrees, "
                         "comma-separated 'lo,hi'. Default uses the "
                         "BelAZ-class window (340.0,480.0).")
    pa.add_argument("--regenerate-summary", dest="regenerate_summary",
                    default=None,
                    help="Path to an existing run dir containing "
                         "data.npz / data_partial.npz. Rewrite "
                         "summary.txt from that data with the current "
                         "wording rules — does NOT recompute physics, "
                         "does NOT touch data.npz.")
    args = pa.parse_args(argv)

    if args.regenerate_summary:
        return _regenerate_summary_from_dir(args.regenerate_summary)

    # Stage J — fidelity preset resolution. Only fills slots that
    # were not explicitly overridden on the command line. Default
    # (None) leaves every CLI default unchanged so legacy invocations
    # continue to behave bit-for-bit.
    _FIDELITY_PRESETS = {
        "low": dict(n_phi=160, n_z=60, n_cycles=2,
                    d_phi_base=4.0, d_phi_peak=1.0),
        "medium": dict(n_phi=320, n_z=120, n_cycles=2,
                       d_phi_base=2.0, d_phi_peak=0.5),
        "high": dict(n_phi=480, n_z=160, n_cycles=2,
                     d_phi_base=1.0, d_phi_peak=0.1),
    }
    if args.fidelity in _FIDELITY_PRESETS:
        preset = _FIDELITY_PRESETS[args.fidelity]
        # n-phi / n-z fall back to the preset if the user did not
        # explicitly pass --n-phi / --n-z (they default to None).
        if args.n_phi is None:
            args.n_phi = preset["n_phi"]
        if args.n_z is None:
            args.n_z = preset["n_z"]
        # n-cycles / d-phi defaults are integer/float so we can't
        # distinguish "user passed default" from "user explicit". The
        # safe rule: only override the script-level defaults
        # (n_cycles=2, d_phi_base=10.0, d_phi_peak=2.0). Other values
        # were explicitly chosen by the user.
        if args.n_cycles == 2:
            args.n_cycles = preset["n_cycles"]
        if args.d_phi_base == 10.0:
            args.d_phi_base = preset["d_phi_base"]
        if args.d_phi_peak == 2.0:
            args.d_phi_peak = preset["d_phi_peak"]
    if args.texture_kind == "groove" and not args.groove_preset:
        raise SystemExit(
            "--texture-kind=groove requires --groove-preset (see "
            "config.diesel_groove_presets.GROOVE_PRESETS).")

    # Resolve base F_max (without scale).
    if args.F_max is not None:
        F_max_base = float(args.F_max)
        F_max_base_source = f"explicit override = {F_max_base:.1f} N"
    elif args.F_max_debug:
        F_max_base = float(getattr(params, "F_max_debug",
                                       params.F_max))
        F_max_base_source = (
            f"params.F_max_debug = {F_max_base:.1f} N "
            "(SANITY MODE — not a production result)")
    else:
        F_max_base = float(params.F_max)
        F_max_base_source = (f"params.F_max = {F_max_base:.1f} N "
                              "(production BelAZ)")

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

    # Envelope-abort policy (Stage Diesel Transient Load-Envelope-0).
    if args.no_envelope_abort:
        envelope_abort = EnvelopeAbortConfig.disabled()
    else:
        envelope_abort = EnvelopeAbortConfig(
            enabled=True,
            clamp_frac_max=float(args.abort_clamp_frac),
            solver_fail_frac_max=float(args.abort_solver_fail_frac),
            consecutive_invalid_max=int(args.abort_consec_invalid),
            save_partial_on_abort=bool(args.save_partial),
        )
    args.envelope_abort = envelope_abort

    # Firing-sector resolution.
    if args.firing_sector:
        try:
            lo, hi = (float(x) for x in
                       args.firing_sector.split(",", 1))
            args.firing_sector_resolved = (lo, hi)
        except Exception as exc:
            raise SystemExit(
                f"--firing-sector must be 'lo,hi'; got "
                f"{args.firing_sector!r}: {exc}")
    else:
        args.firing_sector_resolved = None

    configs = _select_configs(_parse_csv(args.configs))

    if args.gamma_sweep:
        gammas = [float(x) for x in args.gamma_sweep.split(",")]
    else:
        gammas = [float(args.gamma)]

    # Resolve F_max scale list (Section 2 of the patch spec).
    if args.F_max_scale_sweep:
        try:
            scales = [float(x) for x in
                       args.F_max_scale_sweep.split(",")
                       if x.strip()]
        except Exception as exc:
            raise SystemExit(
                f"--F-max-scale-sweep must be a comma-separated list "
                f"of floats; got {args.F_max_scale_sweep!r}: {exc}")
    elif args.F_max_scale is not None:
        scales = [float(args.F_max_scale)]
    else:
        scales = [1.0]

    if args.F_max_scale_sweep:
        # Single parent dir for the whole sweep.
        sweep_ts = datetime.now().strftime("%y%m%d_%H%M%S")
        sweep_root = os.path.join(args.out_base,
                                     f"{sweep_ts}_load_envelope")
        os.makedirs(sweep_root, exist_ok=True)
        envelope_rows: List[Dict[str, Any]] = []
    else:
        sweep_root = None
        envelope_rows = None

    out_dirs = []
    for scale in scales:
        F_max_resolved = F_max_base * float(scale)
        F_max_source = (f"{F_max_base_source} * scale={scale}"
                          if scale != 1.0 else F_max_base_source)
        args.F_max_resolved = F_max_resolved
        args.F_max_source = F_max_source
        args._scale_for_dir = float(scale) if sweep_root else None
        for g in gammas:
            thermal = ThermalConfig(
                mode=args.mode,
                T_in_C=args.T_in,
                gamma_mix=g,
                cp_J_kgK=args.cp,
                mdot_floor_kg_s=args.mdot_floor,
                tau_th_s=float(args.tau_th),
            )
            out_base_for_run = sweep_root or args.out_base
            run_dir, results, dt = _run_one(
                thermal, retry_cfg, args, configs,
                out_base_for_run,
                envelope_abort=envelope_abort,
                return_results=True,
            )
            out_dirs.append(run_dir)
            if envelope_rows is not None:
                envelope_rows.extend(_envelope_rows_from_results(
                    scale, F_max_resolved, results))

    if envelope_rows is not None and sweep_root:
        _write_envelope_csv(sweep_root, envelope_rows)
        print(f"\nLoad-envelope CSV: "
              f"{os.path.join(sweep_root, 'load_envelope.csv')}")

    print("\nFinished runs:")
    for d in out_dirs:
        print("  " + d)


if __name__ == "__main__":
    main()
