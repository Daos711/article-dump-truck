#!/usr/bin/env python3
"""Stage M0 — build load cases from validated Gu fixed-ε runs.

Reads gu_validation_curves.csv, extracts conventional Fx/Fy at
ε = 0.2, 0.5, 0.8 and writes loadcases_gu_aligned.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from cases.gu_2020_magnet.config_gu_magnet import (
    SCHEMA, LOAD_CASE_EPS, LOAD_CASE_NAMES,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=str,
                        default=os.path.join(ROOT, "results",
                                              "herringbone_gu_v1"))
    parser.add_argument("--grid", type=str, default="confirm")
    args = parser.parse_args()

    csv_path = os.path.join(args.validation_dir, "gu_validation_curves.csv")
    if not os.path.exists(csv_path):
        print(f"FAIL: нет {csv_path}")
        sys.exit(1)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    loadcases = {}
    for eps, name in zip(LOAD_CASE_EPS, LOAD_CASE_NAMES):
        matches = [r for r in rows
                   if r["grid"] == args.grid
                   and r["texture_type"] == "conventional"
                   and abs(float(r["eps"]) - eps) < 1e-6]
        if not matches:
            print(f"FAIL: нет conventional @ eps={eps} grid={args.grid}")
            sys.exit(1)
        r = matches[0]
        Fx = float(r["Fx"])
        Fy = float(r["Fy"])
        W = float(r["W"])
        loadcases[name] = dict(
            eps_source=eps,
            Fx_N=Fx, Fy_N=Fy, W_N=W,
            applied_load_N=[Fx, Fy],
            source_grid=args.grid,
        )
        print(f"  {name}: Fx={Fx:.3f} N, Fy={Fy:.3f} N, W={W:.1f} N")

    out_dir = os.path.join(ROOT, "cases", "gu_2020_magnet", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loadcases_gu_aligned.json")
    doc = dict(
        schema_version=SCHEMA,
        source_validation=args.validation_dir,
        source_grid=args.grid,
        loadcases=loadcases,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
