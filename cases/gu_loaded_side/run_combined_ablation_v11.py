#!/usr/bin/env python3
"""Stage D' -- combined 2x2 ablation (v1.1 pipeline).

Reads upstream stages:
  - Stage A  (conv_nomag anchors from anchor_cases.csv)
  - Stage B' (groove_nomag from full_groove_anchor_manifest.json)
  - Stage C' (conv_mag + groove_mag from sector_magnet_results.csv)

Assembles the quartet {conv_nomag, groove_nomag, conv_mag, groove_mag}
for each (loadcase, B_ref).  Invalid quartets (any member not
hard_converged or soft_converged) go to diagnostic CSV only.

Computes headline deltas (groove_mag vs conv_nomag), synergy, and
per-loadcase scenario classification.

Schema: gu_loaded_side_v1_1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from cases.gu_loaded_side.schema import (
    SCHEMA, resolve_stage_dir, classify_status, is_feasible, TOL_HARD,
)
from cases.gu_loaded_side.common import (
    D, R, L, c, n_rpm, eta, sigma,
    w_g, L_g, d_g, beta_deg, N_g,
    LOADCASE_NAMES,
    BREF_SWEEP,
)

CONFIGS = ["conv_nomag", "groove_nomag", "conv_mag", "groove_mag"]


# -- Helpers --------------------------------------------------------------

def _is_float(v):
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _classify_scenario(dCOF_pct: float, dh_pct: float,
                       dp_pct: float) -> str:
    """Scenario tag from headline deltas.

    win_win   : dCOF < 0 AND dh > 0
    adverse   : dh < 0 AND dp > 25%
    marginal  : everything else
    """
    if dCOF_pct < 0.0 and dh_pct > 0.0:
        return "win_win"
    if dh_pct < 0.0 and dp_pct > 25.0:
        return "adverse"
    return "marginal"


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    skip_keys = {"active_cells"}
    fields = sorted({k for r in rows for k in r.keys()
                     if k not in skip_keys
                     and not isinstance(r.get(k), (dict, list, np.ndarray))})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8e}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})


# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage D': combined 2x2 ablation (v1.1)")
    parser.add_argument("--stageA", type=str, required=True,
                        help="path to Stage A output directory")
    parser.add_argument("--stageB", type=str, required=True,
                        help="path to Stage B' output directory")
    parser.add_argument("--stageC", type=str, required=True,
                        help="path to Stage C' output directory")
    parser.add_argument("--out", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()

    # -- Validate Stage A manifest ----------------------------------------
    stageA_dir = resolve_stage_dir(args.stageA)
    manifest_A_path = os.path.join(stageA_dir,
                                   "working_geometry_manifest.json")
    with open(manifest_A_path, encoding="utf-8") as f:
        manifest_A = json.load(f)
    if manifest_A.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageA manifest "
              f"(expected {SCHEMA}, got {manifest_A.get('schema_version')})")
        sys.exit(1)

    loadcases = manifest_A["loadcases"]

    # -- Read anchor CSV (conv_nomag from Stage A) ------------------------
    anchor_csv = os.path.join(stageA_dir, "anchor_cases.csv")
    anchors_by_lc: Dict[str, Dict[str, Any]] = {}
    with open(anchor_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors_by_lc[row["loadcase"]] = {
                k: _try_float(v) for k, v in row.items()
            }

    # -- Validate Stage B' manifest ---------------------------------------
    stageB_dir = resolve_stage_dir(args.stageB)
    manifest_B_path = os.path.join(stageB_dir,
                                   "full_groove_anchor_manifest.json")
    with open(manifest_B_path, encoding="utf-8") as f:
        manifest_B = json.load(f)
    if manifest_B.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageB manifest "
              f"(expected {SCHEMA}, got {manifest_B.get('schema_version')})")
        sys.exit(1)

    # Extract groove_nomag results keyed by loadcase
    groove_nomag_by_lc: Dict[str, Dict[str, Any]] = {}
    results_by_lc_B = manifest_B.get("results_by_loadcase", {})
    for lc_name, lc_results in results_by_lc_B.items():
        gn = lc_results.get("groove_nomag")
        if gn is not None:
            groove_nomag_by_lc[lc_name] = gn

    # -- Validate Stage C' manifest ---------------------------------------
    stageC_dir = resolve_stage_dir(args.stageC)
    manifest_C_path = os.path.join(stageC_dir,
                                   "sector_magnet_manifest.json")
    with open(manifest_C_path, encoding="utf-8") as f:
        manifest_C = json.load(f)
    if manifest_C.get("schema_version") != SCHEMA:
        print(f"FAIL: schema_version mismatch in stageC manifest "
              f"(expected {SCHEMA}, got {manifest_C.get('schema_version')})")
        sys.exit(1)

    bref_list = manifest_C.get("bref_sweep", BREF_SWEEP)

    # -- Read Stage C' CSV for conv_mag + groove_mag lookup ---------------
    stageC_csv = os.path.join(stageC_dir, "sector_magnet_results.csv")
    # Key: (loadcase, config, B_ref_T) -> row dict
    stageC_lookup: Dict[tuple, Dict[str, Any]] = {}
    with open(stageC_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lc = row["loadcase"]
            cfg = row["config"]
            b = float(row["B_ref_T"])
            stageC_lookup[(lc, cfg, b)] = {
                k: _try_float(v) for k, v in row.items()
            }

    os.makedirs(args.out, exist_ok=True)

    print(f"Stage D': combined 2x2 ablation (v1.1)")
    print(f"B_ref sweep: {bref_list}")

    flat_rows: List[Dict[str, Any]] = []
    headline_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    for lc_name in LOADCASE_NAMES:
        lc = loadcases[lc_name]
        W_applied = np.array(lc["applied_load_N"], dtype=float)
        Wa_norm = float(np.linalg.norm(W_applied))

        # conv_nomag anchor
        anchor = anchors_by_lc.get(lc_name)
        if anchor is None:
            print(f"  {lc_name}: no conv_nomag anchor -- skipping")
            continue

        cn = dict(anchor)
        cn["config"] = "conv_nomag"
        cn.setdefault("COF_eq",
                      float(cn.get("COF",
                            cn.get("COF_eq",
                            float(cn.get("friction", 0)) /
                            max(Wa_norm, 1e-20)))))

        # groove_nomag
        gn_data = groove_nomag_by_lc.get(lc_name)

        print(f"\n{'='*60}")
        print(f"Loadcase: {lc_name}  W={Wa_norm:.1f}N")
        print(f"{'='*60}")

        for B_ref in bref_list:
            print(f"\n  B_ref = {B_ref:.2f} T")

            results_4: Dict[str, Dict[str, Any]] = {}

            # -- conv_nomag (from Stage A) --------------------------------
            results_4["conv_nomag"] = cn

            # -- groove_nomag (from Stage B') -----------------------------
            if gn_data is not None:
                gn = dict(gn_data)
                gn["config"] = "groove_nomag"
                gn["B_ref_T"] = 0.0
                results_4["groove_nomag"] = gn
            else:
                results_4["groove_nomag"] = None

            # -- conv_mag (from Stage C') ---------------------------------
            cm_data = stageC_lookup.get((lc_name, "conv_mag", B_ref))
            if cm_data is not None:
                cm = dict(cm_data)
                cm["config"] = "conv_mag"
                cm["B_ref_T"] = float(B_ref)
                results_4["conv_mag"] = cm
            else:
                results_4["conv_mag"] = None

            # -- groove_mag (from Stage C') -------------------------------
            gm_data = stageC_lookup.get((lc_name, "groove_mag", B_ref))
            if gm_data is not None:
                gm = dict(gm_data)
                gm["config"] = "groove_mag"
                gm["B_ref_T"] = float(B_ref)
                results_4["groove_mag"] = gm
            else:
                results_4["groove_mag"] = None

            # -- Classify statuses for quartet validity -------------------
            quartet_valid = True
            for cfg in CONFIGS:
                d = results_4.get(cfg)
                if d is None:
                    quartet_valid = False
                    continue
                # Determine status for this config
                if cfg == "conv_nomag":
                    # Anchors are always valid
                    d_status = d.get("classify_status",
                                     d.get("status", "anchor"))
                    if d_status == "anchor":
                        continue  # anchors always valid
                else:
                    d_status = d.get("classify_status",
                                     d.get("status", "failed"))
                if not is_feasible(d_status) and d_status != "anchor":
                    quartet_valid = False

            # -- Print quartet summary ------------------------------------
            for cfg in CONFIGS:
                d = results_4.get(cfg)
                if d is None:
                    print(f"    {cfg:>16s}: MISSING")
                    continue
                status_str = d.get("classify_status",
                                   d.get("status", "?"))
                print(f"    {cfg:>16s}: eps={float(d.get('eps',0)):.4f} "
                      f"h_min={float(d.get('h_min',0))*1e6:.1f}um "
                      f"p_max={float(d.get('p_max',0))/1e6:.2f}MPa "
                      f"COF={float(d.get('COF_eq',0)):.6f} "
                      f"[{status_str}]")

            # -- Flatten all 4 configs into combined rows -----------------
            for cfg in CONFIGS:
                d = results_4.get(cfg)
                if d is None:
                    continue
                row = dict(d)
                row["loadcase"] = lc_name
                row["B_ref_T"] = float(B_ref) if cfg not in ("conv_nomag",
                                                               "groove_nomag") else 0.0
                flat_rows.append(row)

            # -- Compute headline if quartet valid ------------------------
            if not quartet_valid:
                diag_row = dict(
                    loadcase=lc_name,
                    B_ref_T=B_ref,
                    reason="incomplete_or_failed_quartet",
                )
                for cfg in CONFIGS:
                    d = results_4.get(cfg)
                    if d is not None:
                        diag_row[f"status_{cfg}"] = d.get(
                            "classify_status", d.get("status", "?"))
                    else:
                        diag_row[f"status_{cfg}"] = "missing"
                diagnostic_rows.append(diag_row)
                print(f"    QUARTET INVALID -- diagnostic only")
                continue

            cn_d = results_4["conv_nomag"]
            gn_d = results_4["groove_nomag"]
            cm_d = results_4["conv_mag"]
            gm_d = results_4["groove_mag"]

            cn_cof = float(cn_d.get("COF_eq", cn_d.get("COF", 0)))
            gn_cof = float(gn_d.get("COF_eq", 0))
            cm_cof = float(cm_d.get("COF_eq", 0))
            gm_cof = float(gm_d.get("COF_eq", 0))

            cn_hmin = float(cn_d.get("h_min", 0))
            gm_hmin = float(gm_d.get("h_min", 0))

            cn_pmax = float(cn_d.get("p_max", 0))
            gm_pmax = float(gm_d.get("p_max", 0))

            # -- Headline deltas: groove_mag vs conv_nomag ----------------
            dCOF_pct = ((gm_cof - cn_cof) / max(abs(cn_cof), 1e-20)
                        * 100.0)
            dh_pct = ((gm_hmin - cn_hmin) / max(abs(cn_hmin), 1e-20)
                      * 100.0)
            dp_pct = ((gm_pmax - cn_pmax) / max(abs(cn_pmax), 1e-20)
                      * 100.0)

            # -- Synergy --------------------------------------------------
            # delta_synergy = (COF_gm - COF_cn)
            #   - [(COF_gn - COF_cn) + (COF_cm - COF_cn)]
            delta_synergy = ((gm_cof - cn_cof)
                             - ((gn_cof - cn_cof) + (cm_cof - cn_cof)))

            # -- Scenario classification per loadcase ---------------------
            scenario = _classify_scenario(dCOF_pct, dh_pct, dp_pct)

            hl = dict(
                loadcase=lc_name,
                B_ref_T=B_ref,
                COF_conv_nomag=cn_cof,
                COF_groove_nomag=gn_cof,
                COF_conv_mag=cm_cof,
                COF_groove_mag=gm_cof,
                dCOF_pct=dCOF_pct,
                dh_min_pct=dh_pct,
                dp_max_pct=dp_pct,
                delta_synergy=delta_synergy,
                scenario=scenario,
            )
            headline_rows.append(hl)

            print(f"    HEADLINE: dCOF={dCOF_pct:+.1f}%  "
                  f"dh_min={dh_pct:+.1f}%  dp_max={dp_pct:+.1f}%  "
                  f"synergy={delta_synergy:.6f}  "
                  f"scenario={scenario}")

    # -- Write combined_results.csv ---------------------------------------
    _write_csv(os.path.join(args.out, "combined_results.csv"), flat_rows)

    # -- Write headline_table.csv -----------------------------------------
    _write_csv(os.path.join(args.out, "headline_table.csv"), headline_rows)

    # -- Write diagnostic CSV for invalid quartets ------------------------
    if diagnostic_rows:
        _write_csv(os.path.join(args.out, "diagnostic_invalid.csv"),
                   diagnostic_rows)

    # -- Write combined_manifest.json -------------------------------------
    manifest = dict(
        schema_version=SCHEMA,
        stage="Dp_combined_ablation_v11",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        stageA_manifest=manifest_A_path,
        stageB_manifest=manifest_B_path,
        stageC_manifest=manifest_C_path,
        bref_sweep=[float(b) for b in bref_list],
        configs=CONFIGS,
        tol_accept=TOL_HARD,
        n_headline_rows=len(headline_rows),
        n_diagnostic_rows=len(diagnostic_rows),
        n_combined_rows=len(flat_rows),
        scenarios={f"{hl['loadcase']}_B{hl['B_ref_T']:.2f}":
                   hl["scenario"] for hl in headline_rows},
    )
    with open(os.path.join(args.out, "combined_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nStage D' complete: {len(headline_rows)} headline rows, "
          f"{len(diagnostic_rows)} diagnostic rows")
    print(f"Artifacts: {args.out}/")


if __name__ == "__main__":
    main()
