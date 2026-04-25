#!/usr/bin/env python3
"""Plot L/D sweep results from ld_sweep_pairs.csv + ld_sweep_curves.csv.

6 figures, no titles (project convention).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMA = "herringbone_ld_v1"
EPS_MARKERS = {0.2: "o", 0.5: "s", 0.8: "^"}
EPS_COLORS = {0.2: "tab:blue", 0.5: "tab:orange", 0.8: "tab:red"}


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(
                            os.path.dirname(__file__),
                            "results", "ld_sweep_v1"))
    parser.add_argument("--grid", type=str, default="confirm")
    args = parser.parse_args()

    dd = args.data_dir
    manifest_path = os.path.join(dd, "ld_sweep_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        if m.get("schema_version") != SCHEMA:
            print(f"FAIL: schema={m.get('schema_version')!r}, "
                  f"expected {SCHEMA!r}")
            sys.exit(1)

    pairs = load_csv(os.path.join(dd, "ld_sweep_pairs.csv"))
    curves = load_csv(os.path.join(dd, "ld_sweep_curves.csv"))

    # Filter to requested grid
    pairs = [p for p in pairs if p["grid_name"] == args.grid]
    curves = [c for c in curves if c["grid_name"] == args.grid]

    eps_list = sorted({float(p["eps"]) for p in pairs})
    ld_list = sorted({float(p["ratio_target"]) for p in pairs})

    def ratio_vs_ld(metric, ylabel, fname, axhline=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for eps in eps_list:
            pts = [(float(p["ratio_target"]), float(p[metric]))
                   for p in pairs if abs(float(p["eps"]) - eps) < 1e-6]
            pts.sort()
            if pts:
                ax.plot([x[0] for x in pts], [x[1] for x in pts],
                        marker=EPS_MARKERS.get(eps, "o"),
                        color=EPS_COLORS.get(eps, "gray"),
                        lw=2, markersize=7, label=f"ε = {eps}")
        if axhline is not None:
            ax.axhline(axhline, color="gray", ls=":", lw=1)
        ax.set_xlabel("L / D")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(dd, fname), dpi=150)
        plt.close(fig)
        print(f"→ {fname}")

    ratio_vs_ld("cof_ratio", "COF_herr / COF_conv",
                "cof_ratio_vs_LD.png", axhline=1.0)
    ratio_vs_ld("W_ratio", "W_herr / W_conv",
                "W_ratio_vs_LD.png", axhline=1.0)
    ratio_vs_ld("h_ratio", "h_min_herr / h_min_conv",
                "h_ratio_vs_LD.png", axhline=1.0)
    ratio_vs_ld("p_ratio", "p_max_herr / p_max_conv",
                "p_ratio_vs_LD.png", axhline=1.0)

    # ── Absolute COF vs L/D by eps ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in eps_list:
        for tt, ls, lbl_suffix in [("conventional", "-", "conv"),
                                     ("herringbone_grooves", "--", "herr")]:
            pts = [(float(r["ratio_target"]), float(r["COF"]))
                   for r in curves
                   if abs(float(r["eps"]) - eps) < 1e-6
                   and r["config"] == tt]
            pts.sort()
            if pts:
                ax.plot([x[0] for x in pts], [x[1] for x in pts],
                        marker=EPS_MARKERS.get(eps, "o"),
                        color=EPS_COLORS.get(eps, "gray"),
                        ls=ls, lw=1.5, markersize=6,
                        label=f"ε={eps} {lbl_suffix}")
    ax.set_xlabel("L / D")
    ax.set_ylabel("COF (Couette-only)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dd, "cof_abs_vs_LD_by_eps.png"), dpi=150)
    plt.close(fig)
    print("→ cof_abs_vs_LD_by_eps.png")

    # ── Benefit map (heatmap-like: L/D × eps → cof_ratio) ────────
    if ld_list and eps_list:
        grid_data = np.full((len(eps_list), len(ld_list)), np.nan)
        for p in pairs:
            ei = eps_list.index(float(p["eps"])) if float(p["eps"]) in eps_list else -1
            li = ld_list.index(float(p["ratio_target"])) if float(p["ratio_target"]) in ld_list else -1
            if ei >= 0 and li >= 0:
                grid_data[ei, li] = float(p["cof_ratio"])
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(grid_data, aspect="auto", origin="lower",
                        cmap="RdYlGn_r",
                        extent=[min(ld_list) - 0.025, max(ld_list) + 0.025,
                                min(eps_list) - 0.05, max(eps_list) + 0.05])
        for ei, eps in enumerate(eps_list):
            for li, ld in enumerate(ld_list):
                v = grid_data[ei, li]
                if np.isfinite(v):
                    ax.text(ld, eps, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color="black")
        ax.set_xlabel("L / D")
        ax.set_ylabel("ε")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("COF_herr / COF_conv")
        fig.tight_layout()
        fig.savefig(os.path.join(dd, "benefit_map.png"), dpi=150)
        plt.close(fig)
        print("→ benefit_map.png")


if __name__ == "__main__":
    main()
