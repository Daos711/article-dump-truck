#!/usr/bin/env python3
"""Diagnostic C — equilibrium на трёх Wy ровно для того, чтобы
проверить: появляется ли положительный эффект текстуры, если
принудительно опустить ε_eq в зону окна (≈0.3).

Диагностика, не полный sweep (ТЗ coexp_v1.1). Hard-coded:
  * Wy ∈ {0.05, 0.10, 0.25}   (LOW_LOAD + CONTROL)
  * tol_accept = 0.01            (relaxed vs 0.005)
  * max_iter_nr = 200            (vs 80)
  * grid = 800x200               (coarse)

Decision signal:
  * PROCEED_TO_V2  — ≥1 candidate useful при Wy=0.05 или Wy=0.10
  * CLOSE_COEXP    — иначе
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(__file__)
for _p in (os.path.join(_HERE, ".."), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.coexp_schema import (  # noqa: E402
    DEFAULT_TOL_ACCEPT, DEFAULT_WY_SHARE, EQUILIBRIUM_TOP_N,
)

DIAGNOSTIC_SCHEMA = "coexp_v1.1"
DIAGNOSTIC_ID = "C"

WY_SHARES = [0.05, 0.10, 0.25]   # low-load diagnostic + control
LOW_LOAD_WY = [0.05, 0.10]       # for decision_signal
CONTROL_WY = 0.25
TOL_ACCEPT = 0.01
MAX_ITER_NR = 200
GRID = "800x200"


def wy_tag(wy: float) -> str:
    return f"{int(round(float(wy) * 100)):03d}"


def compute_decision_signal(results_by_wy: Dict[str, List[Dict[str, Any]]],
                              low_load_keys: List[str]) -> str:
    """Pure function — single source of truth для decision logic.

    PROCEED_TO_V2 iff существует хоть один pair с useful=True при Wy из
    `low_load_keys`. Иначе CLOSE_COEXP.
    """
    for key in low_load_keys:
        pairs = results_by_wy.get(key) or []
        if any(p.get("useful") for p in pairs):
            return "PROCEED_TO_V2"
    return "CLOSE_COEXP"


def build_diagnostic_summary(run_dir: str,
                               wy_shares: List[float],
                               base_run: str) -> Dict[str, Any]:
    """Собрать summary_diagnostic_C.json из трёх подкаталогов."""
    results_by_wy: Dict[str, List[Dict[str, Any]]] = {}
    tol_effective_seen: List[float] = []
    max_iter_seen: List[int] = []

    for wy in wy_shares:
        subdir = os.path.join(run_dir, f"equilibrium_wy{wy_tag(wy)}")
        summary_path = os.path.join(subdir, "equilibrium_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"отсутствует {summary_path} — "
                f"запусти run_coexp_equilibrium.py с --Wy-share-list")
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        if s.get("schema_version") != DIAGNOSTIC_SCHEMA:
            raise RuntimeError(
                f"{summary_path}: schema_version="
                f"{s.get('schema_version')!r}, "
                f"expected {DIAGNOSTIC_SCHEMA!r}")
        # Приводим к лёгкому представлению для агрегата
        lite_pairs = []
        for p in s.get("pairs", []):
            r = p.get("ratios") or {}
            lite_pairs.append(dict(
                profile_hash=p["profile_hash"],
                family=(p.get("profile_spec") or {}).get("family"),
                params=(p.get("profile_spec") or {}).get("params"),
                h_r=r.get("h_r"),
                p_r=r.get("p_r"),
                f_r=r.get("f_r"),
                c_d=r.get("c_d"),
                J_eq=p.get("J_eq"),
                useful=bool(p.get("useful")),
                smooth_accepted=bool(p.get("smooth_accepted")),
                textured_accepted=bool(p.get("textured_accepted")),
                eps_smooth=(p.get("smooth") or {}).get("eps"),
                eps_textured=(p.get("textured") or {}).get("eps"),
                rel_residual_smooth=(p.get("smooth") or {}).get("rel_residual"),
                rel_residual_textured=(p.get("textured") or {}).get("rel_residual"),
            ))
        key = f"{wy:.3f}"
        results_by_wy[key] = lite_pairs
        if "tol_accept_effective" in s:
            tol_effective_seen.append(float(s["tol_accept_effective"]))
        if "max_iter_nr" in s:
            max_iter_seen.append(int(s["max_iter_nr"]))

    # Summary aggregates
    any_useful_per_wy: Dict[str, int] = {}
    best_h_ratio_per_wy: Dict[str, Any] = {}
    best_h_ratio_at_eps: Dict[str, Any] = {}
    for key, pairs in results_by_wy.items():
        any_useful_per_wy[key] = sum(1 for p in pairs if p.get("useful"))
        # Лучший h_r по pairs с known ratios
        hrs = [(p["h_r"], p.get("eps_smooth")) for p in pairs
                if p.get("h_r") is not None]
        if hrs:
            best_h, best_eps = max(hrs, key=lambda t: t[0])
            best_h_ratio_per_wy[key] = float(best_h)
            best_h_ratio_at_eps[key] = (
                float(best_eps) if best_eps is not None else None)
        else:
            best_h_ratio_per_wy[key] = None
            best_h_ratio_at_eps[key] = None

    decision = compute_decision_signal(
        results_by_wy,
        low_load_keys=[f"{wy:.3f}" for wy in LOW_LOAD_WY])

    doc = dict(
        schema_version=DIAGNOSTIC_SCHEMA,
        diagnostic_id=DIAGNOSTIC_ID,
        base_run=base_run,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        wy_shares_tested=[float(x) for x in wy_shares],
        tol_accept_effective=(tol_effective_seen[0]
                               if tol_effective_seen else None),
        max_iter_nr=(max_iter_seen[0] if max_iter_seen else None),
        results_by_wy=results_by_wy,
        summary=dict(
            any_useful_per_wy=any_useful_per_wy,
            best_h_ratio_per_wy=best_h_ratio_per_wy,
            best_h_ratio_at_eps=best_h_ratio_at_eps,
            decision_signal=decision,
            low_load_wy=[float(x) for x in LOW_LOAD_WY],
            control_wy=float(CONTROL_WY),
        ),
    )
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True,
                        help="output run_id (e.g. coexp_2026_04_16)")
    parser.add_argument("--from-confirm", type=str, required=True,
                        help="upstream confirm run_id")
    parser.add_argument("--skip-solve", action="store_true",
                        help="только агрегировать уже существующие "
                             "equilibrium_wy<XXX>/ под run_id")
    args = parser.parse_args()

    base = os.path.join(_HERE, "..", "results", "coexp", args.run_id)
    run_dir = os.path.abspath(base)
    os.makedirs(run_dir, exist_ok=True)

    if not args.skip_solve:
        equilibrium_script = os.path.join(_HERE, "run_coexp_equilibrium.py")
        wy_list_str = ",".join(str(x) for x in WY_SHARES)
        cmd = [
            sys.executable, equilibrium_script,
            "--run-id", args.run_id,
            "--from-confirm", args.from_confirm,
            "--Wy-share-list", wy_list_str,
            "--grid", GRID,
            "--phase", "E1",
            "--tol-accept-override", str(TOL_ACCEPT),
            "--max-iter-nr", str(MAX_ITER_NR),
        ]
        print("=" * 60)
        print("Diagnostic C — running equilibrium on 3 Wy shares")
        print("=" * 60)
        print(f"Command: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"FAIL: equilibrium script exited {res.returncode}")
            sys.exit(res.returncode)

    # Aggregate
    print("\n" + "=" * 60)
    print("Diagnostic C — aggregating")
    print("=" * 60)
    try:
        doc = build_diagnostic_summary(run_dir, WY_SHARES, args.run_id)
    except Exception as e:
        print(f"FAIL aggregation: {e}")
        sys.exit(1)

    diag_dir = os.path.join(run_dir, "diagnostic_C")
    os.makedirs(diag_dir, exist_ok=True)
    out_path = os.path.join(diag_dir, "summary_diagnostic_C.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    s = doc["summary"]
    best_h_fmt = {
        k: (f"{v:.4f}" if isinstance(v, float) else v)
        for k, v in s["best_h_ratio_per_wy"].items()
    }
    print(f"\nsummary_diagnostic_C.json: {out_path}")
    print(f"  any_useful_per_wy: {s['any_useful_per_wy']}")
    print(f"  best_h_ratio_per_wy: {best_h_fmt}")
    print(f"  decision_signal: {s['decision_signal']}")


if __name__ == "__main__":
    main()
