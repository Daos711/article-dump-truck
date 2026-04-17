#!/usr/bin/env python3
"""Plot ablation results from ablation_results.json.

Two figures:
  1. ablation_cof_bar.png — grouped bar: COF × 4 configs × 3 loadcases @ B=0.50T
  2. ablation_hmin_vs_bref.png — line: h_min vs B_ref × 4 configs @ L50
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data_dir = os.path.join(
        os.path.dirname(__file__), "results", "ablation_v1")
    src = os.path.join(data_dir, "ablation_results.json")
    if not os.path.exists(src):
        print(f"FAIL: нет {src}")
        sys.exit(1)
    with open(src, "r", encoding="utf-8") as f:
        doc = json.load(f)

    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    lcs = doc["loadcases"]
    CONFIGS = ["conv_nomag", "groove_nomag", "conv_mag", "groove_mag"]
    LABELS = ["Conv", "Groove", "Conv+Mag", "Groove+Mag"]
    COLORS = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b"]
    LC_NAMES = ["L20", "L50", "L80"]

    # ── Fig 1: COF bar @ B=0.50T ─────────────────────────────────
    bkey = "Bref_0.50T"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(LC_NAMES))
    width = 0.18
    for i, (cfg, label, color) in enumerate(zip(CONFIGS, LABELS, COLORS)):
        vals = []
        for lc in LC_NAMES:
            entry = lcs.get(lc, {}).get(bkey, {}).get(cfg, {})
            vals.append(float(entry.get("COF_eq", 0.0)))
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(LC_NAMES)
    ax.set_xlabel("Load case")
    ax.set_ylabel("COF (Couette-only)")
    ax.set_title("2×2 ablation: COF at B_ref = 0.50 T")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out1 = os.path.join(fig_dir, "ablation_cof_bar.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"→ {out1}")

    # ── Fig 2: h_min vs B_ref @ L50 ──────────────────────────────
    bref_values = sorted(
        k for k in lcs.get("L50", {}).keys()
        if k.startswith("Bref_"))
    bref_floats = [float(k.split("_")[1].rstrip("T")) for k in bref_values]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg, label, color in zip(CONFIGS, LABELS, COLORS):
        hmin_vals = []
        for bk in bref_values:
            entry = lcs.get("L50", {}).get(bk, {}).get(cfg, {})
            hmin_vals.append(float(entry.get("h_min", 0.0)) * 1e6)
        ax.plot(bref_floats, hmin_vals, "o-", lw=2, markersize=7,
                color=color, label=label)
    ax.set_xlabel("B_ref (T)")
    ax.set_ylabel("h_min (μm)")
    ax.set_title("h_min vs B_ref — load case L50")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(fig_dir, "ablation_hmin_vs_bref.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"→ {out2}")


if __name__ == "__main__":
    main()
