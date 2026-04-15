#!/usr/bin/env python3
"""Сравнение smooth vs textured: overlay графики + JSON + MD report.

Читает последние cycle_N.npz из обоих директорий, рисует графики,
считает metrics (включая time-below-hcrit), выводит GO/NO-GO оценку.
"""
import sys
import os
import json
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(CASE_DIR))
sys.path.insert(0, CASE_DIR)
sys.path.insert(0, THIS)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import case_config as cfg
from scaling import (
    omega_from_rpm, pressure_scale, force_scale,
    CYCLE_TAU, tau_to_crank_deg, dt_from_crank_step,
)


def load_last_cycle(run_dir):
    files = sorted([f for f in os.listdir(run_dir)
                     if f.startswith("cycle_") and f.endswith(".npz")])
    if not files:
        raise FileNotFoundError(f"Нет cycle_*.npz в {run_dir}")
    d = np.load(os.path.join(run_dir, files[-1]))
    return {k: d[k] for k in d.files}, len(files)


def time_below(threshold, arr, dt_val):
    """Доля цикла (%) где arr < threshold."""
    return float(np.sum(np.asarray(arr) < threshold) * dt_val
                  / CYCLE_TAU * 100)


def time_above(threshold, arr, dt_val):
    return float(np.sum(np.asarray(arr) > threshold) * dt_val
                  / CYCLE_TAU * 100)


def compute_metrics(data, p_scale, dt_val):
    h_nd = np.asarray(data["h_min"])
    p_nd = np.asarray(data["p_max"])
    h_um = h_nd * cfg.c * 1e6
    p_MPa = p_nd * p_scale / 1e6
    eps = np.sqrt(np.asarray(data["X"])**2 + np.asarray(data["Y"])**2)

    m = {
        "min_hmin_nd": float(np.min(h_nd)),
        "min_hmin_um": float(np.min(h_um)),
        "max_pmax_nd": float(np.max(p_nd)),
        "max_pmax_MPa": float(np.max(p_MPa)),
        "mean_cav_frac": float(np.mean(data["cav_frac"])),
        "max_cav_frac": float(np.max(data["cav_frac"])),
        "e_max": float(np.max(eps)),
        "e_min": float(np.min(eps)),
    }
    for h_crit in cfg.h_crit_um:
        m[f"time_below_h{h_crit}um_pct"] = time_below(h_crit, h_um, dt_val)
    for p_crit in cfg.p_crit_MPa:
        m[f"time_above_p{p_crit}MPa_pct"] = time_above(p_crit, p_MPa, dt_val)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rpm", type=int, default=cfg.n_rpm)
    parser.add_argument("--load-pct", type=int, default=cfg.load_pct)
    parser.add_argument("--zone", type=str, default="upstream")
    args = parser.parse_args()

    res_dir = os.path.join(CASE_DIR, "results")
    smooth_dir = os.path.join(res_dir,
                                f"smooth_{args.n_rpm}rpm_{args.load_pct}pct")
    tex_dir = os.path.join(
        res_dir, f"textured_{args.n_rpm}rpm_{args.load_pct}pct_{args.zone}")
    out_dir = os.path.join(
        res_dir, f"compare_{args.n_rpm}rpm_{args.load_pct}pct_{args.zone}")
    os.makedirs(out_dir, exist_ok=True)

    s_data, s_cycles = load_last_cycle(smooth_dir)
    t_data, t_cycles = load_last_cycle(tex_dir)

    omega = omega_from_rpm(args.n_rpm)
    p_sc = pressure_scale(cfg.eta, omega, cfg.R, cfg.c)
    dt_val, _ = dt_from_crank_step(cfg.crank_step_deg)

    # CA axis
    t_nd = np.asarray(s_data["t"])
    ca = tau_to_crank_deg(t_nd - t_nd[0])

    # Metrics
    m_s = compute_metrics(s_data, p_sc, dt_val)
    m_t = compute_metrics(t_data, p_sc, dt_val)

    # Benefit
    def pct(tex, sm):
        return (tex - sm) / max(abs(sm), 1e-10) * 100

    benefit = {
        "delta_hmin_pct": pct(m_t["min_hmin_um"], m_s["min_hmin_um"]),
        "delta_pmax_pct": pct(m_t["max_pmax_MPa"], m_s["max_pmax_MPa"]),
        "delta_cav_frac_pct": pct(m_t["mean_cav_frac"],
                                    m_s["mean_cav_frac"]),
    }
    for h_crit in cfg.h_crit_um:
        key = f"time_below_h{h_crit}um_pct"
        benefit[f"delta_{key}"] = m_t[key] - m_s[key]

    # Texture config
    tex_cfg = {}
    tex_cfg_path = os.path.join(tex_dir, "texture_config.json")
    if os.path.exists(tex_cfg_path):
        with open(tex_cfg_path, "r", encoding="utf-8") as f:
            tex_cfg = json.load(f)

    # Summary JSON
    summary = {
        "case": f"diesel_sun2019_{args.n_rpm}rpm_{args.load_pct}pct",
        "texture": tex_cfg,
        "smooth": m_s,
        "textured": m_t,
        "benefit": benefit,
        "convergence": {
            "smooth_cycles": s_cycles,
            "textured_cycles": t_cycles,
        },
    }
    with open(os.path.join(out_dir, "summary.json"),
               "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print table
    print("=" * 80)
    print(f"COMPARE: smooth vs textured — {args.zone}")
    print("=" * 80)
    print(f"{'Metric':<30} {'Smooth':>12} {'Textured':>12} {'Δ':>10}")
    print("-" * 80)
    print(f"{'min h_min (μm)':<30} {m_s['min_hmin_um']:>12.3f} "
          f"{m_t['min_hmin_um']:>12.3f} {benefit['delta_hmin_pct']:>+9.2f}%")
    print(f"{'max p_max (MPa)':<30} {m_s['max_pmax_MPa']:>12.2f} "
          f"{m_t['max_pmax_MPa']:>12.2f} {benefit['delta_pmax_pct']:>+9.2f}%")
    print(f"{'mean cav_frac':<30} {m_s['mean_cav_frac']:>12.3f} "
          f"{m_t['mean_cav_frac']:>12.3f} "
          f"{benefit['delta_cav_frac_pct']:>+9.2f}%")
    print(f"{'e_max':<30} {m_s['e_max']:>12.4f} "
          f"{m_t['e_max']:>12.4f}")
    for h_crit in cfg.h_crit_um:
        key = f"time_below_h{h_crit}um_pct"
        print(f"{'time h<'+str(h_crit)+'μm (%)':<30} "
              f"{m_s[key]:>12.2f} {m_t[key]:>12.2f} "
              f"{benefit['delta_'+key]:>+9.2f}")
    print()

    # GO/NO-GO
    go = False
    reasons = []
    if benefit["delta_hmin_pct"] >= 3.0:
        go = True
        reasons.append(f"min h_min +{benefit['delta_hmin_pct']:.1f}% ≥ 3%")
    if benefit["delta_pmax_pct"] <= -3.0:
        go = True
        reasons.append(f"max p_max {benefit['delta_pmax_pct']:.1f}% ≤ −3%")
    for h_crit in cfg.h_crit_um:
        key = f"time_below_h{h_crit}um_pct"
        rel = benefit[f"delta_{key}"] / max(m_s[key], 1e-3) * 100
        if rel <= -10.0:
            go = True
            reasons.append(f"time h<{h_crit}μm reduced {rel:.0f}%")
    print(f"GO/NO-GO: {'GO ✓' if go else 'NO-GO ✗'}")
    for r in reasons:
        print(f"  ✓ {r}")
    print()

    # Графики
    _plot_overlay(ca, s_data, t_data, p_sc, out_dir)
    _plot_orbit(s_data, t_data, out_dir)

    # Markdown report
    _export_markdown(summary, out_dir)

    print(f"Выход → {out_dir}")


def _plot_overlay(ca, s, t, p_sc, out_dir):
    # h_min
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ca, s["h_min"] * cfg.c * 1e6, "b-", lw=1.2, label="Smooth")
    ax.plot(ca, t["h_min"] * cfg.c * 1e6, "r-", lw=1.2, label="Textured")
    for h_c in cfg.h_crit_um:
        ax.axhline(h_c, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("h_min (μm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay_hmin.png"), dpi=150)
    plt.close(fig)

    # p_max
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ca, s["p_max"] * p_sc / 1e6, "b-", lw=1.2, label="Smooth")
    ax.plot(ca, t["p_max"] * p_sc / 1e6, "r-", lw=1.2, label="Textured")
    for p_c in cfg.p_crit_MPa:
        ax.axhline(p_c, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("p_max (MPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay_pmax.png"), dpi=150)
    plt.close(fig)

    # cav_frac
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ca, s["cav_frac"], "b-", lw=1.2, label="Smooth")
    ax.plot(ca, t["cav_frac"], "r-", lw=1.2, label="Textured")
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("cav_frac")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay_cav.png"), dpi=150)
    plt.close(fig)

    # ε
    fig, ax = plt.subplots(figsize=(10, 5))
    eps_s = np.sqrt(s["X"]**2 + s["Y"]**2)
    eps_t = np.sqrt(t["X"]**2 + t["Y"]**2)
    ax.plot(ca, eps_s, "b-", lw=1.2, label="Smooth")
    ax.plot(ca, eps_t, "r-", lw=1.2, label="Textured")
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("ε")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay_eps.png"), dpi=150)
    plt.close(fig)

    # Load
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ca, s["WX"], "b-", lw=1.2, label="WaX")
    ax.plot(ca, s["WY"], "r-", lw=1.2, label="WaY")
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("W (nondim)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "load.png"), dpi=150)
    plt.close(fig)


def _plot_orbit(s, t, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(s["X"], s["Y"], "b-", lw=1.0, label="Smooth")
    ax.plot(t["X"], t["Y"], "r-", lw=1.0, label="Textured")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay_orbit.png"), dpi=150)
    plt.close(fig)


def _export_markdown(s, out_dir):
    tex = s["texture"]
    ms = s["smooth"]
    mt = s["textured"]
    b = s["benefit"]
    lines = [
        "# Diesel Main Bearing — Smooth vs Textured",
        f"## Case: {s['case']}",
        f"## Texture: {tex.get('zone', '?')} zone, "
        f"{tex.get('phi_start_deg', 0):.1f}°–{tex.get('phi_end_deg', 0):.1f}°",
        f"- Dimple diameter: {tex.get('dimple_diameter_um', '?')} μm",
        f"- Dimple depth: {tex.get('dimple_depth_um', '?')} μm",
        f"- Area density: {tex.get('area_density', 0) * 100:.0f}%",
        f"- Number of dimples: {tex.get('N_tex', '?')}",
        "",
        "| Metric              | Smooth  | Textured | Benefit |",
        "|---------------------|---------|----------|---------|",
        f"| min h_min (μm)      | {ms['min_hmin_um']:.3f} | "
        f"{mt['min_hmin_um']:.3f} | {b['delta_hmin_pct']:+.2f}% |",
        f"| max p_max (MPa)     | {ms['max_pmax_MPa']:.2f} | "
        f"{mt['max_pmax_MPa']:.2f} | {b['delta_pmax_pct']:+.2f}% |",
        f"| mean cav_frac       | {ms['mean_cav_frac']:.3f} | "
        f"{mt['mean_cav_frac']:.3f} | {b['delta_cav_frac_pct']:+.2f}% |",
        f"| e_max               | {ms['e_max']:.4f} | "
        f"{mt['e_max']:.4f} | — |",
    ]
    for h_crit in cfg.h_crit_um:
        key = f"time_below_h{h_crit}um_pct"
        lines.append(f"| time h<{h_crit}μm (%)      | {ms[key]:.2f} | "
                      f"{mt[key]:.2f} | {b['delta_'+key]:+.2f} |")
    lines.append("")
    lines.append(f"## Convergence")
    lines.append(f"- Smooth cycles: {s['convergence']['smooth_cycles']}")
    lines.append(f"- Textured cycles: {s['convergence']['textured_cycles']}")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
