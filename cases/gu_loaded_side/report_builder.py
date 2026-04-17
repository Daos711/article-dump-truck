#!/usr/bin/env python3
"""Build final_report.md from all stage manifests under --root.

Reads working geometry, loadcase, partial groove, sector magnet,
and combined ablation manifests to produce a consolidated Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps = solve_payvar_salant_cpu

from cases.gu_loaded_side.schema import SCHEMA, classify_status
from cases.gu_loaded_side.common import LOADCASE_NAMES


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _fmt(v, fmt_str=".4f"):
    try:
        return format(float(v), fmt_str)
    except (ValueError, TypeError):
        return str(v)


def main():
    parser = argparse.ArgumentParser(
        description="Build final report from gu_loaded_side results")
    parser.add_argument("--root", type=str,
                        default="results/gu_loaded_side_v1",
                        help="results directory (default: results/gu_loaded_side_v1)")
    args = parser.parse_args()

    root = args.root

    # ── Load manifests ───────────────────────────────────────────
    manifest_A = _load_json(os.path.join(root,
                            "working_geometry_manifest.json"))
    manifest_B = _load_json(os.path.join(root,
                            "partial_manifest.json"))
    manifest_C = _load_json(os.path.join(root,
                            "sector_magnet_manifest.json"))
    manifest_D = _load_json(os.path.join(root,
                            "combined_manifest.json"))

    # ── Validate schema versions ─────────────────────────────────
    for label, m in [("A", manifest_A), ("B", manifest_B),
                     ("C", manifest_C), ("D", manifest_D)]:
        if m is not None and m.get("schema_version") != SCHEMA:
            print(f"WARNING: Stage {label} schema_version mismatch "
                  f"(expected {SCHEMA}, got {m.get('schema_version')})")

    # ── Load CSVs ────────────────────────────────────────────────
    anchor_rows = _read_csv(os.path.join(root, "anchor_cases.csv"))
    loadcase_rows = _read_csv(os.path.join(root, "loadcases.csv"))
    partial_rows = _read_csv(os.path.join(root, "partial_results.csv"))
    sector_rows = _read_csv(os.path.join(root,
                            "sector_magnet_results.csv"))
    combined_rows = _read_csv(os.path.join(root,
                              "combined_results.csv"))
    headline_rows = _read_csv(os.path.join(root,
                              "headline_table.csv"))

    best_by_lc_data = _load_json(os.path.join(
        root, "partial_best_by_loadcase.json"))

    lines: List[str] = []

    def w(text: str = ""):
        lines.append(text)

    w("# GU Loaded-Side Study Report")
    w()
    w(f"Schema: `{SCHEMA}`")
    w()

    # ── 1. Working Geometry Table ────────────────────────────────
    w("## 1. Working Geometry")
    w()
    if manifest_A and "working_geometry" in manifest_A:
        wg = manifest_A["working_geometry"]
        w("| Parameter | Value |")
        w("|-----------|-------|")
        for k, v in sorted(wg.items()):
            w(f"| {k} | {v} |")
    else:
        w("_Stage A manifest not found._")
    w()

    if manifest_A and "groove_geometry" in manifest_A:
        gg = manifest_A["groove_geometry"]
        w("### Groove Geometry")
        w()
        w("| Parameter | Value |")
        w("|-----------|-------|")
        for k, v in sorted(gg.items()):
            w(f"| {k} | {v} |")
    w()

    # ── 2. Loadcase Table ────────────────────────────────────────
    w("## 2. Loadcases")
    w()
    if loadcase_rows:
        headers = loadcase_rows[0].keys()
        w("| " + " | ".join(headers) + " |")
        w("| " + " | ".join("---" for _ in headers) + " |")
        for r in loadcase_rows:
            w("| " + " | ".join(str(r[h]) for h in headers) + " |")
    elif manifest_A and "loadcases" in manifest_A:
        lcs = manifest_A["loadcases"]
        w("| Loadcase | eps_source | W_N | phi_loaded_deg |")
        w("|----------|-----------|-----|----------------|")
        for name, lc in lcs.items():
            w(f"| {name} | {lc.get('eps_source','')} "
              f"| {_fmt(lc.get('W_N',''), '.1f')} "
              f"| {_fmt(lc.get('phi_loaded_deg',''), '.1f')} |")
    else:
        w("_Loadcase data not found._")
    w()

    # ── 3. Partial Groove Best Patterns ──────────────────────────
    w("## 3. Partial Groove Best Patterns (Stage B)")
    w()
    if best_by_lc_data:
        w("| Loadcase | N_active | shift_cells | active_cells | COF_eq |")
        w("|----------|----------|-------------|-------------|--------|")
        for lc_name in LOADCASE_NAMES:
            b = best_by_lc_data.get(lc_name)
            if b:
                cells_str = str(b.get("active_cells", []))
                w(f"| {lc_name} "
                  f"| {b.get('N_active','')} "
                  f"| {b.get('shift_cells','')} "
                  f"| {cells_str} "
                  f"| {_fmt(b.get('COF_eq',''))} |")
            else:
                w(f"| {lc_name} | - | - | - | - |")
    elif manifest_B and "best_by_loadcase" in manifest_B:
        bbl = manifest_B["best_by_loadcase"]
        w("| Loadcase | N_active | shift_cells | COF_eq |")
        w("|----------|----------|-------------|--------|")
        for lc_name in LOADCASE_NAMES:
            b = bbl.get(lc_name, {})
            w(f"| {lc_name} "
              f"| {b.get('N_active','-')} "
              f"| {b.get('shift_cells','-')} "
              f"| {_fmt(b.get('COF_eq',''), '.6f')} |")
    else:
        w("_Stage B data not found._")
    w()

    if manifest_B:
        w(f"Total combos: {manifest_B.get('n_combos', '?')}, "
          f"feasible: {manifest_B.get('n_feasible', '?')}")
    w()

    # ── 4. Sector Magnet Force Summary ───────────────────────────
    w("## 4. Sector Magnet Force Summary (Stage C)")
    w()
    if sector_rows:
        w("| Loadcase | B_ref_T | F_mag_N | COF_eq | h_min_um | status |")
        w("|----------|---------|---------|--------|----------|--------|")
        for r in sector_rows:
            w(f"| {r.get('loadcase','')} "
              f"| {_fmt(r.get('B_ref_T',''), '.2f')} "
              f"| {_fmt(r.get('F_mag_N',''), '.3f')} "
              f"| {_fmt(r.get('COF_eq',''))} "
              f"| {_fmt(_to_float(r.get('h_min',0))*1e6, '.1f')} "
              f"| {r.get('status','')} |")
    else:
        w("_Stage C data not found._")
    w()

    # ── 5. Headline Comparison Table ─────────────────────────────
    w("## 5. Headline Comparison (groove_mag vs conv_nomag)")
    w()
    if headline_rows:
        w("| Loadcase | B_ref_T | COF_conv | COF_groove_mag "
          "| dCOF_pct | dh_min_pct | scenario |")
        w("|----------|---------|----------|---------------|"
          "----------|------------|----------|")
        for r in headline_rows:
            w(f"| {r.get('loadcase','')} "
              f"| {_fmt(r.get('B_ref_T',''), '.2f')} "
              f"| {_fmt(r.get('COF_conv_nomag',''))} "
              f"| {_fmt(r.get('COF_groove_mag',''))} "
              f"| {_fmt(r.get('dCOF_pct',''), '+.1f')} "
              f"| {_fmt(r.get('dh_min_pct',''), '+.1f')} "
              f"| {r.get('scenario','')} |")
    else:
        w("_Headline data not found._")
    w()

    # ── 6. Scenario Classification ───────────────────────────────
    w("## 6. Scenario Classification")
    w()
    if manifest_D and "scenarios" in manifest_D:
        scenarios = manifest_D["scenarios"]
        w("| Case | Scenario |")
        w("|------|----------|")
        for case_key, scenario in sorted(scenarios.items()):
            w(f"| {case_key} | {scenario} |")
    elif headline_rows:
        w("| Case | Scenario |")
        w("|------|----------|")
        for r in headline_rows:
            case_key = f"{r.get('loadcase','')}_B{_fmt(r.get('B_ref_T',''), '.2f')}"
            w(f"| {case_key} | {r.get('scenario','')} |")
    else:
        w("_No scenario data available._")
    w()

    # ── Write report ─────────────────────────────────────────────
    report_path = os.path.join(root, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written: {report_path}")
    print(f"  Sections: 6")
    print(f"  Lines: {len(lines)}")


if __name__ == "__main__":
    main()
