#!/usr/bin/env python3
"""Plot Fig-6-like COF vs ε from gu_validation_curves.csv."""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(ROOT, "results",
                                             "herringbone_gu_v1"))
    parser.add_argument("--grid", type=str, default="confirm")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "gu_validation_curves.csv")
    if not os.path.exists(csv_path):
        print(f"FAIL: нет {csv_path}")
        sys.exit(1)
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {
        "conventional": dict(marker="o", ls="-", color="black",
                              label="Conventional"),
        "straight_grooves": dict(marker="s", ls="--", color="tab:blue",
                                  label="Straight grooves"),
        "herringbone_grooves": dict(marker="^", ls="-.", color="tab:red",
                                     label="Herringbone grooves"),
    }
    for tt, st in styles.items():
        pts = [(float(r["eps"]), float(r["COF"]))
                for r in rows
                if r["grid"] == args.grid and r["texture_type"] == tt]
        if not pts:
            continue
        pts.sort()
        eps_arr = [p[0] for p in pts]
        cof_arr = [p[1] for p in pts]
        ax.plot(eps_arr, cof_arr, lw=2, markersize=7, **st)
    ax.set_xlabel("Eccentricity ratio ε")
    ax.set_ylabel("COF (Couette-only)")
    ax.set_title(f"Gu 2020 reproduction — {args.grid} grid")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(args.data_dir, "gu_fig6_reproduction.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
