#!/usr/bin/env python3
"""Plot results from gu_loaded_side pipeline.

Generates up to 4 PNGs (no set_title on any plot):
  - cof_headline_vs_Bref.png
  - hmin_headline_vs_Bref.png
  - partial_COF_vs_pattern.png
  - ablation_bars.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps = solve_payvar_salant_cpu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cases.gu_loaded_side.schema import SCHEMA
from cases.gu_loaded_side.common import LOADCASE_NAMES

CONFIGS = ["conv_nomag", "groove_nomag", "conv_mag", "groove_mag"]
LABELS = ["Conv", "Groove", "Conv+Mag", "Groove+Mag"]
COLORS = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b"]


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def plot_cof_headline_vs_bref(rows, fig_dir, lc_names):
    """COF for 4 configs vs B_ref, one subplot per loadcase."""
    bref_set = sorted({_to_float(r.get("B_ref_T", 0)) for r in rows})
    if not bref_set or len(bref_set) < 2:
        return

    n_lc = len(lc_names)
    fig, axes = plt.subplots(1, n_lc, figsize=(5 * n_lc, 5), squeeze=False)

    for col, lc in enumerate(lc_names):
        ax = axes[0, col]
        lc_rows = [r for r in rows if r.get("loadcase") == lc]
        for cfg, label, color in zip(CONFIGS, LABELS, COLORS):
            cfg_rows = [r for r in lc_rows if r.get("config") == cfg]
            if not cfg_rows:
                continue
            xs = [_to_float(r.get("B_ref_T", 0)) for r in cfg_rows]
            ys = [_to_float(r.get("COF_eq", 0)) for r in cfg_rows]
            # Sort by B_ref
            pairs = sorted(zip(xs, ys))
            if pairs:
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, "o-", lw=2, markersize=6,
                        color=color, label=label)
        ax.set_xlabel("B_ref (T)")
        ax.set_ylabel("COF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, lc, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout()
    out = os.path.join(fig_dir, "cof_headline_vs_Bref.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def plot_hmin_headline_vs_bref(rows, fig_dir, lc_names):
    """h_min for 4 configs vs B_ref, one subplot per loadcase."""
    bref_set = sorted({_to_float(r.get("B_ref_T", 0)) for r in rows})
    if not bref_set or len(bref_set) < 2:
        return

    n_lc = len(lc_names)
    fig, axes = plt.subplots(1, n_lc, figsize=(5 * n_lc, 5), squeeze=False)

    for col, lc in enumerate(lc_names):
        ax = axes[0, col]
        lc_rows = [r for r in rows if r.get("loadcase") == lc]
        for cfg, label, color in zip(CONFIGS, LABELS, COLORS):
            cfg_rows = [r for r in lc_rows if r.get("config") == cfg]
            if not cfg_rows:
                continue
            xs = [_to_float(r.get("B_ref_T", 0)) for r in cfg_rows]
            ys = [_to_float(r.get("h_min", 0)) * 1e6 for r in cfg_rows]
            pairs = sorted(zip(xs, ys))
            if pairs:
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, "o-", lw=2, markersize=6,
                        color=color, label=label)
        ax.set_xlabel("B_ref (T)")
        ax.set_ylabel("h_min (um)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, lc, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout()
    out = os.path.join(fig_dir, "hmin_headline_vs_Bref.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def plot_partial_cof_vs_pattern(rows, fig_dir, lc_names):
    """COF for partial patterns at each loadcase (from stageB CSV)."""
    if not rows:
        return

    n_lc = len(lc_names)
    fig, axes = plt.subplots(1, n_lc, figsize=(5 * n_lc, 5.5), squeeze=False)

    for col, lc in enumerate(lc_names):
        ax = axes[0, col]
        lc_rows = [r for r in rows if r.get("loadcase") == lc]
        if not lc_rows:
            continue

        # Group by N_active
        n_active_set = sorted({int(_to_float(r.get("N_active", 0)))
                               for r in lc_rows})
        for na in n_active_set:
            na_rows = [r for r in lc_rows
                       if int(_to_float(r.get("N_active", 0))) == na]
            shifts = [int(_to_float(r.get("shift_cells", 0)))
                      for r in na_rows]
            cofs = [_to_float(r.get("COF_eq", 0)) for r in na_rows]
            feasible = [str(r.get("feasible", "")).lower() in
                        ("true", "1") for r in na_rows]
            for s, co, fe in zip(shifts, cofs, feasible):
                marker = "o" if fe else "x"
                ax.plot(s, co, marker, markersize=8 if fe else 6,
                        label=f"N={na}" if s == shifts[0] else None)

        ax.set_xlabel("shift_cells")
        ax.set_ylabel("COF_eq")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, lc, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    fig.tight_layout()
    out = os.path.join(fig_dir, "partial_COF_vs_pattern.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def plot_ablation_bars(rows, fig_dir, lc_names):
    """Grouped bar chart of 4 configs at max B_ref per loadcase."""
    if not rows:
        return

    # Find max B_ref
    bref_max = max(_to_float(r.get("B_ref_T", 0)) for r in rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(lc_names))
    width = 0.18

    for i, (cfg, label, color) in enumerate(zip(CONFIGS, LABELS, COLORS)):
        vals = []
        for lc in lc_names:
            lc_rows = [r for r in rows
                       if r.get("loadcase") == lc
                       and r.get("config") == cfg
                       and abs(_to_float(r.get("B_ref_T", 0)) - bref_max)
                       < 0.01]
            if lc_rows:
                vals.append(_to_float(lc_rows[0].get("COF_eq", 0)))
            else:
                # For conv_nomag / groove_nomag, B_ref may be 0
                fallback = [r for r in rows
                            if r.get("loadcase") == lc
                            and r.get("config") == cfg]
                if fallback:
                    vals.append(_to_float(fallback[0].get("COF_eq", 0)))
                else:
                    vals.append(0.0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(lc_names)
    ax.set_xlabel("Load case")
    ax.set_ylabel("COF (Couette-only)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(fig_dir, "ablation_bars.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from gu_loaded_side pipeline")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="root of gu_loaded_side_v1 results")
    args = parser.parse_args()

    data_dir = args.data_dir
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Determine which loadcases are present
    lc_names = list(LOADCASE_NAMES)

    print("Plotting gu_loaded_side results")
    print(f"  data_dir: {data_dir}")

    # ── combined_results.csv (Stage D) ───────────────────────────
    combined_csv = os.path.join(data_dir, "combined_results.csv")
    combined_rows = _read_csv(combined_csv)

    # ── partial_results.csv (Stage B) ────────────────────────────
    partial_csv = os.path.join(data_dir, "partial_results.csv")
    partial_rows = _read_csv(partial_csv)

    if combined_rows:
        print(f"  Found combined_results.csv ({len(combined_rows)} rows)")
        plot_cof_headline_vs_bref(combined_rows, fig_dir, lc_names)
        plot_hmin_headline_vs_bref(combined_rows, fig_dir, lc_names)
        plot_ablation_bars(combined_rows, fig_dir, lc_names)
    else:
        print("  No combined_results.csv — skipping headline/ablation plots")

    if partial_rows:
        print(f"  Found partial_results.csv ({len(partial_rows)} rows)")
        plot_partial_cof_vs_pattern(partial_rows, fig_dir, lc_names)
    else:
        print("  No partial_results.csv — skipping partial pattern plot")

    print("Done.")


if __name__ == "__main__":
    main()
