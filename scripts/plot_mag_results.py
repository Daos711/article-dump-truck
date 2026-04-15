#!/usr/bin/env python3
"""Графики и MD report для magnetic unloading exploratory case.

Читает JSON из mag_smooth_summary.json и (опционально)
mag_textured_summary.json. Строит:
  1. eps_eq vs mag_share
  2. h_min vs mag_share
  3. p_max vs mag_share
  4. hydro_share / mag_share vs mag_share_target (force sharing)
  5. textured/smooth ratios (h_min, p_max) vs mag_share
Генерирует report.md.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.data_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", "magnetic_pump")
    os.makedirs(out_dir, exist_ok=True)

    sm = load_json(os.path.join(out_dir, "mag_smooth_summary.json"))
    if sm is None:
        print(f"Не найдено {out_dir}/mag_smooth_summary.json")
        sys.exit(1)
    tx = load_json(os.path.join(out_dir, "mag_textured_summary.json"))

    smooth_runs = sm["continuation"]
    cfg = sm["config"]
    targets = np.array([r["mag_share_target"] for r in smooth_runs]) * 100

    def col(rs, key):
        return np.array([r[key] for r in rs])

    eps_s = col(smooth_runs, "eps")
    hmin_s = col(smooth_runs, "h_min") * 1e6   # μm
    pmax_s = col(smooth_runs, "p_max") / 1e6   # MPa
    hydro_share_s = col(smooth_runs, "hydro_load_share")
    mag_share_s = col(smooth_runs, "mag_load_share")

    # --- 1. eps vs mag_share ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, eps_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tx is not None:
        eps_t = col(tx["textured"], "eps")
        ax.plot(targets, eps_t, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("mag_share target (%)")
    ax.set_ylabel("ε_eq")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eps_vs_mag_share.png"), dpi=150)
    plt.close(fig)

    # --- 2. h_min ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, hmin_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tx is not None:
        hmin_t = col(tx["textured"], "h_min") * 1e6
        ax.plot(targets, hmin_t, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("mag_share target (%)")
    ax.set_ylabel("h_min (μm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hmin_vs_mag_share.png"), dpi=150)
    plt.close(fig)

    # --- 3. p_max ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, pmax_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tx is not None:
        pmax_t = col(tx["textured"], "p_max") / 1e6
        ax.plot(targets, pmax_t, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("mag_share target (%)")
    ax.set_ylabel("p_max (MPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pmax_vs_mag_share.png"), dpi=150)
    plt.close(fig)

    # --- 4. Force sharing ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, hydro_share_s, "bo-", lw=2, markersize=7,
            label="hydro_share")
    ax.plot(targets, mag_share_s, "rs-", lw=2, markersize=7,
            label="mag_share")
    ax.plot(targets, hydro_share_s + mag_share_s, "g^--", lw=1.5,
            markersize=5, alpha=0.7, label="total")
    ax.plot(targets, targets / 100, "k:", lw=1, label="target")
    ax.set_xlabel("mag_share target (%)")
    ax.set_ylabel("load share (projection on e_load)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "force_sharing.png"), dpi=150)
    plt.close(fig)

    # --- 5. Ratios textured/smooth ---
    if tx is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        hmin_t = col(tx["textured"], "h_min") * 1e6
        pmax_t = col(tx["textured"], "p_max") / 1e6
        axes[0].plot(targets, hmin_t / hmin_s, "bo-", lw=2, markersize=7)
        axes[0].axhline(1.0, color="gray", ls=":")
        axes[0].set_xlabel("mag_share target (%)")
        axes[0].set_ylabel("h_min_tex / h_min_smooth")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(targets, pmax_t / pmax_s, "rs-", lw=2, markersize=7)
        axes[1].axhline(1.0, color="gray", ls=":")
        axes[1].set_xlabel("mag_share target (%)")
        axes[1].set_ylabel("p_max_tex / p_max_smooth")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ratios_tex_vs_smooth.png"), dpi=150)
        plt.close(fig)

    # --- MD report ---
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Pump bearing + magnetic unloading (surrogate)\n\n")
        f.write(f"- Grid: {cfg['N_phi']}×{cfg['N_Z']}\n")
        f.write(f"- F₀ = {cfg['F0_N']:.1f} Н\n")
        f.write(f"- W_applied = ({cfg['W_applied_N'][0]:.1f}, "
                f"{cfg['W_applied_N'][1]:.1f}) Н\n")
        f.write(f"- Sectors: {cfg['sector_angles_deg']}°\n\n")

        f.write("## Baseline smooth (no magnet)\n\n")
        b = sm["baseline"]
        f.write(f"- ε = {b['eps']:.4f}, X = {b['X']:.4f}, Y = {b['Y']:.4f}\n")
        f.write(f"- attitude = {b['attitude_deg']:.1f}°\n")
        f.write(f"- h_min = {b['h_min']*1e6:.2f} μm\n")
        f.write(f"- p_max = {b['p_max']/1e6:.2f} MPa\n")
        f.write(f"- residual = {b['rel_residual']:.2e}\n\n")

        f.write("## Sanity checks\n\n")
        acc = sm["acceptance"]
        f.write(f"- K_mag=0 reproduces baseline: "
                f"**{'PASS' if acc['baseline_reproduced'] else 'FAIL'}**\n")
        f.write(f"- ε monotonic decrease: "
                f"**{'PASS' if acc['monotonic_eps_decrease'] else 'FAIL'}**\n")
        f.write(f"- max residual = {acc['max_residual']:.2e}\n\n")

        f.write("## Smooth continuation\n\n")
        f.write("| mag_share | ε_eq | h_min, μm | p_max, MPa "
                "| hydro_share | mag_share |\n")
        f.write("|-----------|------|-----------|-----------"
                "|-------------|-----------|\n")
        for r in smooth_runs:
            f.write(f"| {r['mag_share_target']*100:.1f}% "
                    f"| {r['eps']:.4f} "
                    f"| {r['h_min']*1e6:.2f} "
                    f"| {r['p_max']/1e6:.2f} "
                    f"| {r['hydro_load_share']:.3f} "
                    f"| {r['mag_load_share']:.3f} |\n")
        f.write("\n")

        if tx is not None:
            f.write("## Textured comparison\n\n")
            f.write("| mag_share | ε_s | ε_t | Δε "
                    "| h_min_s,μm | h_min_t,μm | h_t/h_s "
                    "| p_max_s,MPa | p_max_t,MPa | p_t/p_s |\n")
            f.write("|-----------|-----|-----|-----"
                    "|-----------|-----------|---------"
                    "|-------------|-------------|---------|\n")
            for rs, rt in zip(sm["continuation"], tx["textured"]):
                dE = rt["eps"] - rs["eps"]
                f.write(f"| {rs['mag_share_target']*100:.1f}% "
                        f"| {rs['eps']:.4f} | {rt['eps']:.4f} | {dE:+.4f} "
                        f"| {rs['h_min']*1e6:.2f} "
                        f"| {rt['h_min']*1e6:.2f} "
                        f"| {rt['h_min']/max(rs['h_min'],1e-12):.4f} "
                        f"| {rs['p_max']/1e6:.2f} "
                        f"| {rt['p_max']/1e6:.2f} "
                        f"| {rt['p_max']/max(rs['p_max'],1e-12):.4f} |\n")

        f.write("\n## Figures\n\n")
        for name in ["eps_vs_mag_share", "hmin_vs_mag_share",
                      "pmax_vs_mag_share", "force_sharing"]:
            f.write(f"![{name}]({name}.png)\n\n")
        if tx is not None:
            f.write(f"![ratios_tex_vs_smooth](ratios_tex_vs_smooth.png)\n")

    print(f"Графики + report.md сохранены в {out_dir}")


if __name__ == "__main__":
    main()
