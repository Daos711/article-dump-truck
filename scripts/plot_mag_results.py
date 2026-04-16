#!/usr/bin/env python3
"""Графики + MD report для magnetic unloading exploratory case.

Читает:
  results/magnetic_pump/mag_smooth_summary.json
  results/magnetic_pump/mag_textured_summary.json (опционально)

Все метки в терминах unload_share (не mag_share).
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
        print(f"Нет {out_dir}/mag_smooth_summary.json")
        sys.exit(1)
    tx = load_json(os.path.join(out_dir, "mag_textured_summary.json"))

    smooth_runs = sm["continuation"]
    if not smooth_runs:
        print("В smooth JSON нет accepted continuation")
        sys.exit(1)

    targets = np.array([r["unload_share_target"] for r in smooth_runs]) * 100

    def col(rs, key):
        return np.array([r[key] for r in rs])

    eps_s = col(smooth_runs, "eps")
    hmin_s = col(smooth_runs, "h_min") * 1e6
    pmax_s = col(smooth_runs, "p_max") / 1e6
    hydro_s = col(smooth_runs, "hydro_share_actual")
    unload_s = col(smooth_runs, "unload_share_actual")

    # 1. eps_eq vs unload_share_target
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, eps_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tx is not None:
        tex_eps = []
        tex_t = []
        for p in tx["pairs"]:
            if p.get("accepted") and p.get("textured"):
                tex_eps.append(p["textured"]["eps"])
                tex_t.append(p["unload_share_target"] * 100)
        if tex_eps:
            ax.plot(tex_t, tex_eps, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("ε_eq")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eps_vs_unload_share.png"), dpi=150)
    plt.close(fig)

    # 2. h_min
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, hmin_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tx is not None:
        tex_h = []
        tex_t = []
        for p in tx["pairs"]:
            if p.get("accepted") and p.get("textured"):
                tex_h.append(p["textured"]["h_min"] * 1e6)
                tex_t.append(p["unload_share_target"] * 100)
        if tex_h:
            ax.plot(tex_t, tex_h, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("h_min (μm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hmin_vs_unload_share.png"), dpi=150)
    plt.close(fig)

    # 3. p_max
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, pmax_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tx is not None:
        tex_p = []
        tex_t = []
        for p in tx["pairs"]:
            if p.get("accepted") and p.get("textured"):
                tex_p.append(p["textured"]["p_max"] / 1e6)
                tex_t.append(p["unload_share_target"] * 100)
        if tex_p:
            ax.plot(tex_t, tex_p, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("p_max (MPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pmax_vs_unload_share.png"), dpi=150)
    plt.close(fig)

    # 4. Force sharing
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, hydro_s, "bo-", lw=2, markersize=7, label="hydro_share")
    ax.plot(targets, unload_s, "rs-", lw=2, markersize=7, label="unload_share")
    ax.plot(targets, hydro_s + unload_s, "g^--", lw=1.5, markersize=5,
            alpha=0.7, label="sum")
    ax.plot(targets, targets / 100, "k:", lw=1, label="target")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("load share (on ê_resist)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "force_sharing.png"), dpi=150)
    plt.close(fig)

    # 5. textured/smooth ratios
    if tx is not None:
        acc_pairs = [p for p in tx["pairs"]
                     if p.get("accepted") and p.get("textured")]
        if acc_pairs:
            t_arr = [p["unload_share_target"] * 100 for p in acc_pairs]
            hr = [p["ratios"]["h_ratio"] for p in acc_pairs]
            pr = [p["ratios"]["p_ratio"] for p in acc_pairs]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(t_arr, hr, "bo-", lw=2, markersize=7)
            axes[0].axhline(1.0, color="gray", ls=":")
            axes[0].set_xlabel("unload_share_target (%)")
            axes[0].set_ylabel("h_min_tex / h_min_smooth")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(t_arr, pr, "rs-", lw=2, markersize=7)
            axes[1].axhline(1.0, color="gray", ls=":")
            axes[1].set_xlabel("unload_share_target (%)")
            axes[1].set_ylabel("p_max_tex / p_max_smooth")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "ratios_tex_vs_smooth.png"),
                         dpi=150)
            plt.close(fig)

    # MD report
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        cfg = sm["config"]
        f.write("# Pump bearing + magnetic unloading (radial surrogate)\n\n")
        f.write(f"- Model: {cfg.get('model', 'radial')}\n")
        f.write(f"- Grid: {cfg['N_phi']}×{cfg['N_Z']}\n")
        f.write(f"- F₀ = {cfg['F0_N']:.1f} Н\n")
        f.write(f"- W_applied = ({cfg['W_applied_N'][0]:.1f}, "
                f"{cfg['W_applied_N'][1]:.1f}) Н\n\n")

        f.write("## Baseline smooth (no magnet)\n\n")
        b = sm["baseline"]
        f.write(f"- ε = {b['eps']:.4f}, X = {b['X']:.4f}, Y = {b['Y']:.4f}\n")
        f.write(f"- h_min = {b['h_min']*1e6:.2f} μm, "
                f"p_max = {b['p_max']/1e6:.2f} MPa\n")
        f.write(f"- residual = {b['rel_residual']:.2e}\n\n")

        f.write("## Acceptance\n\n")
        acc = sm["acceptance"]
        for key in ["baseline_reproduced", "unload_positive", "eps_monotonic",
                     "sum_shares_ok"]:
            v = acc.get(key)
            f.write(f"- {key}: **{'PASS' if v else 'FAIL'}**\n")
        f.write(f"- max residual = {acc['max_residual']:.2e}\n\n")

        f.write("## Smooth continuation (accepted targets)\n\n")
        f.write("| target | ε | h_min, μm | p_max, MPa "
                "| hydro_share | unload_share |\n")
        f.write("|--------|---|-----------|-----------"
                "|-------------|--------------|\n")
        for r in smooth_runs:
            f.write(f"| {r['unload_share_target']*100:.2f}% "
                    f"| {r['eps']:.4f} "
                    f"| {r['h_min']*1e6:.2f} "
                    f"| {r['p_max']/1e6:.2f} "
                    f"| {r['hydro_share_actual']:+.4f} "
                    f"| {r['unload_share_actual']:+.4f} |\n")
        f.write("\n")

        if tx is not None:
            f.write("## Textured comparison (accepted only)\n\n")
            f.write("| target | smooth ε | tex ε | Δε "
                    "| h_t/h_s | p_t/p_s | Δcav |\n")
            f.write("|--------|----------|-------|-----"
                    "|---------|---------|------|\n")
            for p in tx["pairs"]:
                if not (p.get("accepted") and p.get("textured")):
                    continue
                rs = p["smooth"]
                rt = p["textured"]
                rr = p["ratios"]
                f.write(f"| {p['unload_share_target']*100:.2f}% "
                        f"| {rs['eps']:.4f} | {rt['eps']:.4f} "
                        f"| {rr['delta_eps']:+.4f} "
                        f"| {rr['h_ratio']:.4f} "
                        f"| {rr['p_ratio']:.4f} "
                        f"| {rr['delta_cav']:+.4f} |\n")
            f.write("\n")

        f.write("## Figures\n\n")
        for name in ["eps_vs_unload_share", "hmin_vs_unload_share",
                     "pmax_vs_unload_share", "force_sharing"]:
            f.write(f"![{name}]({name}.png)\n\n")
        if tx is not None:
            f.write("![ratios_tex_vs_smooth](ratios_tex_vs_smooth.png)\n")

    print(f"→ {out_dir}/report.md + figures")


if __name__ == "__main__":
    main()
