#!/usr/bin/env python3
"""Fine-grid sanity check for top-2 candidates from diagnostic C
(coexp_v1.2).

Гипотеза: эффект `h_r ≈ 1.02-1.03` на coarse grid (800×200) с
soft_converged NR может быть сеточным артефактом. Перед coexp_v2
надо подтвердить, что лучшие candidate'ы из diagnostic_C сохраняют
useful behavior на fine grid (1600×400) при тех же tol_accept=0.01
и max_iter_nr=200.

Для каждого low-load Wy (0.05, 0.10) берётся top candidate по `h_r`,
и решается paired equilibrium (smooth + textured) на fine grid.
Aggregator выдаёт `decision_signal`:

  * PROCEED_TO_V2  — оба candidate useful_fine=True И h_r_fine ≥ 1.015
  * INVESTIGATE_NR — все h_r_fine в диапазоне [1.005, 1.015]
  * CLOSE_COEXP    — хотя бы один h_r_fine < 1.0 (опрокинулся)

Вызов:
  python scripts/run_coexp_finegrid_sanity.py \
      --run-id coexp_2026_04_16 \
      --from-diagnostic-C coexp_2026_04_16
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(__file__)
for _p in (os.path.join(_HERE, ".."), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


SCHEMA_FINEGRID = "coexp_v1.2"
DIAGNOSTIC_C_SCHEMA = "coexp_v1.1"
DIAGNOSTIC_ID = "finegrid_sanity"

LOW_LOAD_WY = [0.05, 0.10]
DEFAULT_GRID_FINE = "1600x400"
DEFAULT_GRID_COARSE = "800x200"
TOL_ACCEPT = 0.01
MAX_ITER_NR = 200

# Decision thresholds (ТЗ §2.5)
H_R_USEFUL_GATE = 1.005          # совпадает с equilibrium_useful gate
H_R_PROCEED_THRESHOLD = 1.015    # PROCEED requires margin
H_R_COLLAPSE_THRESHOLD = 1.0     # < 1.0 → опрокинулся


def wy_tag(wy: float) -> str:
    return f"{int(round(float(wy) * 100)):03d}"


# ─── Pure helpers (testable without solver) ────────────────────────

def pick_top2_candidates(diag_C_doc: Dict[str, Any],
                          low_load_wy: Optional[List[float]] = None
                          ) -> List[Dict[str, Any]]:
    """Из diagnostic_C summary выбрать top candidate по best h_r для
    каждого low-load Wy. Возвращает list dict'ов с ключами:
        profile_hash, family, params, Wy_share,
        h_r_coarse, p_r_coarse, f_r_coarse, c_d_coarse,
        useful_coarse, eps_smooth_coarse

    Если top-кандидат для разных Wy совпадает по hash, второй слот
    заполняется второй best для соответствующего Wy.
    """
    low_load = list(low_load_wy if low_load_wy is not None else LOW_LOAD_WY)
    results_by_wy = diag_C_doc.get("results_by_wy") or {}

    chosen: List[Dict[str, Any]] = []
    chosen_hashes = set()

    for wy in low_load:
        key = f"{wy:.3f}"
        pairs = results_by_wy.get(key) or []
        ranked = sorted(
            [p for p in pairs if p.get("h_r") is not None],
            key=lambda p: p["h_r"], reverse=True)
        chosen_for_this_wy = None
        for p in ranked:
            ph = p.get("profile_hash")
            if ph and ph not in chosen_hashes:
                chosen_for_this_wy = p
                break
        if chosen_for_this_wy is None and ranked:
            # Fallback: дублирующий hash тоже допустим, чтобы не
            # потерять второй слот совсем
            chosen_for_this_wy = ranked[0]
        if chosen_for_this_wy is not None:
            entry = dict(
                profile_hash=chosen_for_this_wy["profile_hash"],
                family=chosen_for_this_wy.get("family"),
                params=chosen_for_this_wy.get("params"),
                Wy_share=float(wy),
                h_r_coarse=chosen_for_this_wy.get("h_r"),
                p_r_coarse=chosen_for_this_wy.get("p_r"),
                f_r_coarse=chosen_for_this_wy.get("f_r"),
                c_d_coarse=chosen_for_this_wy.get("c_d"),
                useful_coarse=bool(chosen_for_this_wy.get("useful")),
                eps_smooth_coarse=chosen_for_this_wy.get("eps_smooth"),
            )
            chosen.append(entry)
            chosen_hashes.add(entry["profile_hash"])
    return chosen


def compute_decision_signal(candidates_fine: List[Dict[str, Any]]) -> str:
    """Branches per ТЗ §2.5.

    Pure function. Любая candidate без `h_r_fine` (None) или с
    `h_r_fine < 1.0` → CLOSE_COEXP.
    PROCEED_TO_V2 требует ALL useful AND ALL >= 1.015.
    INVESTIGATE_NR — ALL >= 1.005, но не все >= 1.015.
    Иначе (например, mixed: один useful, второй ниже 1.005) → CLOSE.
    """
    if not candidates_fine:
        return "CLOSE_COEXP"

    h_values = [c.get("h_r_fine") for c in candidates_fine]
    if any(h is None or h < H_R_COLLAPSE_THRESHOLD for h in h_values):
        return "CLOSE_COEXP"

    all_useful = all(bool(c.get("useful_fine")) for c in candidates_fine)
    all_with_margin = all(h >= H_R_PROCEED_THRESHOLD for h in h_values)
    if all_useful and all_with_margin:
        return "PROCEED_TO_V2"

    if all(h >= H_R_USEFUL_GATE for h in h_values):
        return "INVESTIGATE_NR"

    return "CLOSE_COEXP"


def build_summary(diag_C_doc: Dict[str, Any],
                   candidates_fine: List[Dict[str, Any]],
                   base_run: str, from_diag: str,
                   tol_accept: float, max_iter: int) -> Dict[str, Any]:
    decision = compute_decision_signal(candidates_fine)
    n_useful_fine = sum(1 for c in candidates_fine
                        if c.get("useful_fine"))
    n_coarse_only = sum(1 for c in candidates_fine
                        if c.get("useful_coarse")
                        and not c.get("useful_fine"))
    h_finite = [c.get("h_r_fine") for c in candidates_fine
                if c.get("h_r_fine") is not None]
    return dict(
        schema_version=SCHEMA_FINEGRID,
        diagnostic_id=DIAGNOSTIC_ID,
        base_run=base_run,
        from_diagnostic_C=from_diag,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        tol_accept_effective=float(tol_accept),
        max_iter_nr=int(max_iter),
        candidates_tested=candidates_fine,
        summary=dict(
            n_candidates=len(candidates_fine),
            n_useful_fine=int(n_useful_fine),
            n_useful_coarse_but_not_fine=int(n_coarse_only),
            max_h_r_fine=(max(h_finite) if h_finite else None),
            decision_signal=decision,
        ),
    )


# ─── Solver path (lazy imports — keeps tests PS-free) ──────────────

def _run_one_finegrid(spec_dict: Dict[str, Any],
                       Wy_share: float,
                       tol_accept: float, max_iter: int,
                       grid, run_dir: str, hash_short: str,
                       from_diag: str,
                       ps_solver, relief_fn, constants):
    """One paired equilibrium solve on the fine grid. Writes per-pair
    subdir and returns summary dict."""
    from models.coexp_schema import (
        REFERENCE_TEXTURE,
        EQUILIBRIUM_NR_SEED_X0, EQUILIBRIUM_NR_SEED_Y0,
        DEFAULT_STEP_CAP, DEFAULT_EPS_MAX,
        make_experiment_spec,
    )
    from models.coexp_equilibrium import solve_equilibrium_pair
    from models.coexp_objective import (
        compute_J_eq, equilibrium_useful, compute_incremental_ratios,
    )

    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = grid
    exp = make_experiment_spec(spec_dict, REFERENCE_TEXTURE)

    t0 = time.time()
    result = solve_equilibrium_pair(
        exp, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
        constants["R"], constants["L"], constants["c"],
        constants["sigma"], constants["eta"], constants["omega"],
        constants["F0"], Wy_share,
        ps_solver=ps_solver, texture_relief_fn=relief_fn,
        tol_accept=tol_accept,
        step_cap=DEFAULT_STEP_CAP, eps_max=DEFAULT_EPS_MAX,
        max_iter=max_iter,
    )
    dt = time.time() - t0

    s = result.get("smooth")
    t = result.get("textured")
    if s and t:
        ratios = compute_incremental_ratios(
            dict(h_min=s["h_min"], p_max=s["p_max"],
                 friction=s["friction"], cav_frac=s["cav_frac"]),
            dict(h_min=t["h_min"], p_max=t["p_max"],
                 friction=t["friction"], cav_frac=t["cav_frac"]),
        )
        J_eq = compute_J_eq(ratios)
        useful = bool(equilibrium_useful(ratios))
    else:
        ratios = None
        J_eq = None
        useful = False

    subdir_name = f"finegrid_sanity_wy{wy_tag(Wy_share)}_{hash_short}"
    subdir = os.path.join(run_dir, subdir_name)
    os.makedirs(subdir, exist_ok=True)
    summary_per = dict(
        schema_version=SCHEMA_FINEGRID,
        phase="equilibrium_finegrid_sanity",
        from_diagnostic_C=from_diag,
        Wy_share=float(Wy_share),
        tol_accept=float(tol_accept),
        tol_accept_effective=float(tol_accept),
        max_iter_nr=int(max_iter),
        grid=dict(N_phi=int(Phi.shape[1]), N_Z=int(Phi.shape[0])),
        nr_seed=dict(X0=EQUILIBRIUM_NR_SEED_X0,
                     Y0=EQUILIBRIUM_NR_SEED_Y0),
        pair=dict(
            profile_hash=result["profile_hash"],
            experiment_id=result["experiment_id"],
            profile_spec=result["profile_spec"],
            smooth=s,
            textured=t,
            smooth_accepted=bool(result.get("smooth_accepted")),
            textured_accepted=bool(result.get("textured_accepted")),
            ratios=ratios,
            J_eq=J_eq,
            useful=useful,
        ),
        elapsed_sec=float(dt),
    )
    with open(os.path.join(subdir, "equilibrium_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary_per, f, indent=2, ensure_ascii=False)
    return summary_per, subdir


def _read_existing_subdir(run_dir: str, c: Dict[str, Any]) -> Dict[str, Any]:
    short = c["profile_hash"][:8]
    subdir_name = f"finegrid_sanity_wy{wy_tag(c['Wy_share'])}_{short}"
    sp = os.path.join(run_dir, subdir_name, "equilibrium_summary.json")
    if not os.path.exists(sp):
        raise FileNotFoundError(sp)
    with open(sp, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_entry(c: Dict[str, Any], summary_per: Dict[str, Any],
                 coarse_grid: Dict[str, int]) -> Dict[str, Any]:
    pair = summary_per["pair"]
    ratios = pair.get("ratios") or {}
    return dict(
        profile_hash=c["profile_hash"],
        family=c["family"],
        Wy_share=c["Wy_share"],
        coarse_grid=coarse_grid,
        fine_grid=summary_per["grid"],
        h_r_coarse=c.get("h_r_coarse"),
        p_r_coarse=c.get("p_r_coarse"),
        f_r_coarse=c.get("f_r_coarse"),
        c_d_coarse=c.get("c_d_coarse"),
        useful_coarse=bool(c.get("useful_coarse")),
        h_r_fine=ratios.get("h_r"),
        p_r_fine=ratios.get("p_r"),
        f_r_fine=ratios.get("f_r"),
        c_d_fine=ratios.get("c_d"),
        useful_fine=bool(pair.get("useful")),
        smooth_accepted_fine=bool(pair.get("smooth_accepted")),
        textured_accepted_fine=bool(pair.get("textured_accepted")),
        rel_residual_smooth_fine=(
            pair["smooth"].get("rel_residual")
            if pair.get("smooth") else None),
        rel_residual_textured_fine=(
            pair["textured"].get("rel_residual")
            if pair.get("textured") else None),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--from-diagnostic-C", type=str, required=True)
    parser.add_argument("--grid-fine", type=str,
                        default=DEFAULT_GRID_FINE)
    parser.add_argument("--grid-coarse", type=str,
                        default=DEFAULT_GRID_COARSE)
    parser.add_argument("--skip-solve", action="store_true",
                        help="только агрегировать уже existing subdirs")
    args = parser.parse_args()

    base = os.path.join(_HERE, "..", "results", "coexp", args.run_id)
    run_dir = os.path.abspath(base)

    diag_C_path = os.path.join(
        _HERE, "..", "results", "coexp", args.from_diagnostic_C,
        "diagnostic_C", "summary_diagnostic_C.json")
    diag_C_path = os.path.abspath(diag_C_path)
    if not os.path.exists(diag_C_path):
        print(f"FAIL: нет {diag_C_path}")
        sys.exit(1)
    with open(diag_C_path, "r", encoding="utf-8") as f:
        diag_C = json.load(f)
    if diag_C.get("schema_version") != DIAGNOSTIC_C_SCHEMA:
        print(f"FAIL: diagnostic_C schema="
              f"{diag_C.get('schema_version')!r}, "
              f"expected {DIAGNOSTIC_C_SCHEMA!r}")
        sys.exit(1)

    candidates = pick_top2_candidates(diag_C, low_load_wy=LOW_LOAD_WY)
    if not candidates:
        print("FAIL: нет кандидатов с h_r в diagnostic_C low-load Wy")
        sys.exit(1)
    print(f"Top {len(candidates)} candidates from diagnostic_C "
          f"({args.from_diagnostic_C}):")
    for c in candidates:
        h = c.get("h_r_coarse")
        h_str = f"{h:.4f}" if isinstance(h, (int, float)) else "n/a"
        print(f"  hash={c['profile_hash'][:8]} family={c['family']} "
              f"Wy={c['Wy_share']} h_r_coarse={h_str} "
              f"useful_coarse={c['useful_coarse']}")

    coarse_N_phi, coarse_N_Z = (int(x) for x in args.grid_coarse.split("x"))
    fine_N_phi, fine_N_Z = (int(x) for x in args.grid_fine.split("x"))
    coarse_grid_dict = dict(N_phi=coarse_N_phi, N_Z=coarse_N_Z)

    out_dir = os.path.join(run_dir, "finegrid_sanity")
    os.makedirs(out_dir, exist_ok=True)

    candidates_fine: List[Dict[str, Any]] = []
    if not args.skip_solve:
        from _coexp_runtime import (
            load_ps_solver, load_relief_fn, mineral_constants, make_grid,
        )
        ps_solver = load_ps_solver()
        relief_fn = load_relief_fn()
        constants = mineral_constants()
        grid = make_grid(fine_N_phi, fine_N_Z)

        print(f"\nFine grid: {fine_N_phi}x{fine_N_Z}, "
              f"tol_accept={TOL_ACCEPT}, max_iter_nr={MAX_ITER_NR}")
        for c in candidates:
            spec = dict(family=c["family"], params=dict(c["params"]))
            short = c["profile_hash"][:8]
            print(f"\nSolving: {short} (family={c['family']}, "
                  f"Wy={c['Wy_share']})...")
            summary_per, subdir = _run_one_finegrid(
                spec, c["Wy_share"], TOL_ACCEPT, MAX_ITER_NR,
                grid, run_dir, short, args.from_diagnostic_C,
                ps_solver, relief_fn, constants)
            entry = _make_entry(c, summary_per, coarse_grid_dict)
            candidates_fine.append(entry)
            print(f"  → h_r_fine="
                  f"{entry['h_r_fine']!r}, "
                  f"useful_fine={entry['useful_fine']}, "
                  f"sm_acc={entry['smooth_accepted_fine']}, "
                  f"tx_acc={entry['textured_accepted_fine']}, "
                  f"sm_res={entry['rel_residual_smooth_fine']}, "
                  f"tx_res={entry['rel_residual_textured_fine']}")
            print(f"  subdir: {subdir}")
    else:
        for c in candidates:
            try:
                summary_per = _read_existing_subdir(run_dir, c)
            except FileNotFoundError as e:
                print(f"FAIL --skip-solve: нет {e}")
                sys.exit(1)
            entry = _make_entry(c, summary_per, coarse_grid_dict)
            candidates_fine.append(entry)

    # Aggregate
    doc = build_summary(diag_C, candidates_fine, args.run_id,
                         args.from_diagnostic_C,
                         TOL_ACCEPT, MAX_ITER_NR)
    out_path = os.path.join(out_dir, "summary_finegrid_sanity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"\nsummary_finegrid_sanity.json: {out_path}")

    # CSV
    csv_path = os.path.join(out_dir, "finegrid_results.csv")
    flat_fields = sorted({k for c in candidates_fine for k in c.keys()
                          if not isinstance(c[k], dict)})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for c in candidates_fine:
            w.writerow({k: c.get(k, "") for k in flat_fields})
    print(f"finegrid_results.csv: {csv_path}")

    s = doc["summary"]
    print("\n" + "=" * 60)
    print(f"  n_candidates: {s['n_candidates']}")
    print(f"  n_useful_fine: {s['n_useful_fine']}")
    print(f"  n_useful_coarse_but_not_fine: "
          f"{s['n_useful_coarse_but_not_fine']}")
    print(f"  max_h_r_fine: {s['max_h_r_fine']}")
    print(f"  decision_signal: {s['decision_signal']}")


if __name__ == "__main__":
    main()
