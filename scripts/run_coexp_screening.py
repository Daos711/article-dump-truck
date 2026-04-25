#!/usr/bin/env python3
"""Phase S1 — fixed-ε screening (ТЗ §6).

DOE: 48 LHS + 8×2 local refinement per family × 3 families = 192.
Grid 800×200, ε ∈ {0.30, 0.40, 0.50}. Oil mineral, noPV.

За-один-проход:
  1) cylindrical reference (once per ε)   ← clarification 2
  2) initial DOE generation (LHS)         ← clarification 1 (cyclic phases)
  3) geometry hard-filter                  ← §4.1
  4) smooth-vs-cylindrical sanity filter   ← §4.2
  5) paired smooth+textured solve          ← §2 invariant (single bore_profile_fn)
  6) per-ε ratios + J_eps
  7) profile classification (screen_fail?)
  8) write screening_manifest.json / CSV / JSONL

Local refinement:
  Взять top-2 в каждой семье → ±10% amp, ±10° phase → 8 per base.
  Итого 192 профилей (см. §5).

Все hash-based IDs детерминированы (clarification 3).
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(__file__)
for _p in (os.path.join(_HERE, ".."), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from models.bore_profiles import (
    profile_hash, hard_geometry_fail, make_bore_profile,
)
from models.coexp_schema import (
    SCHEMA_VERSION, REFERENCE_TEXTURE, SCREENING_EPS,
    SCREENING_LHS_SEED, SCREENING_N_LHS_PER_FAMILY,
    SCREENING_N_REFINE_PER_FAMILY, SCREENING_N_REFINE_BASES,
    SCREENING_FAMILIES, HARD_GEOMETRY, SMOOTH_SANITY_VS_CYL,
    CONFIRM_TOP_N, PASS_RATE_OK, PASS_RATE_WARN,
    make_experiment_spec,
)
from models.coexp_screening import (
    generate_initial_doe, generate_local_refinement,
    filter_hard_fail, select_top_global, select_top_per_family,
    pass_rate_status,
)
from models.coexp_pairing import make_H_pair_builders
from models.coexp_objective import (
    compute_incremental_ratios, compute_J_eps,
    classify_profile,
)

from _coexp_runtime import (
    load_ps_solver, load_relief_fn, mineral_constants,
    make_grid, solve_case_metrics, git_sha,
)


def cylindrical_reference(eps_values, Phi, phi_1D, Z_1D, d_phi, d_Z,
                           constants, ps_solver, sigma_over_c):
    """Smooth cylindrical baseline, computed ONCE per screening
    (clarification 2). Shared denominator for §4.2 sanity checks."""
    ref = {}
    for eps in eps_values:
        H0 = 1.0 + float(eps) * np.cos(Phi)
        H = np.sqrt(H0 ** 2 + sigma_over_c ** 2)
        m = solve_case_metrics(H, Phi, phi_1D, Z_1D, d_phi, d_Z,
                                constants, ps_solver)
        ref[float(eps)] = dict(
            h_min=m["h_min"], p_max=m["p_max"],
            friction=m["friction"], cav_frac=m["cav_frac"])
    return ref


def smooth_sanity_fail(metrics_sm: Dict[str, float],
                        cyl: Dict[str, float]) -> (bool, str):
    r_h = metrics_sm["h_min"] / max(cyl["h_min"], 1e-30)
    r_p = metrics_sm["p_max"] / max(cyl["p_max"], 1e-30)
    r_f = metrics_sm["friction"] / max(cyl["friction"], 1e-30)
    if r_h < SMOOTH_SANITY_VS_CYL["h_min_ratio_min"]:
        return True, f"h_min_sm/h_min_cyl={r_h:.3f} < {SMOOTH_SANITY_VS_CYL['h_min_ratio_min']}"
    if r_p > SMOOTH_SANITY_VS_CYL["p_max_ratio_max"]:
        return True, f"p_max_sm/p_max_cyl={r_p:.3f} > {SMOOTH_SANITY_VS_CYL['p_max_ratio_max']}"
    if r_f > SMOOTH_SANITY_VS_CYL["friction_ratio_max"]:
        return True, f"fr_sm/fr_cyl={r_f:.3f} > {SMOOTH_SANITY_VS_CYL['friction_ratio_max']}"
    return False, ""


def solve_profile(profile_spec, eps_values, grid, constants,
                   ps_solver, relief_fn, cyl_ref):
    """Вернуть per-profile dict со всеми метриками + классификацией."""
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = grid
    exp = make_experiment_spec(profile_spec, REFERENCE_TEXTURE)
    pair = make_H_pair_builders(
        exp, constants["R"], constants["L"], constants["c"],
        constants["sigma"], texture_relief_fn=relief_fn)

    metrics_sm: Dict[float, Dict[str, float]] = {}
    metrics_tx: Dict[float, Dict[str, float]] = {}
    ratios_per_eps: Dict[float, Dict[str, float]] = {}
    J_per_eps: Dict[float, float] = {}

    smooth_sanity_rejected = False
    sanity_reason = ""
    for eps in eps_values:
        H_sm = pair.smooth(eps, Phi, Zm)
        m_sm = solve_case_metrics(H_sm, Phi, phi_1D, Z_1D, d_phi, d_Z,
                                    constants, ps_solver)
        metrics_sm[float(eps)] = m_sm
        # §4.2 sanity
        fail, reason = smooth_sanity_fail(m_sm, cyl_ref[float(eps)])
        if fail:
            smooth_sanity_rejected = True
            sanity_reason = f"eps={eps}: {reason}"
            break

    result = dict(
        profile_id=exp.profile_id,
        experiment_id=exp.experiment_id,
        family=exp.family,
        profile_spec=exp.profile.as_dict(),
    )

    if smooth_sanity_rejected:
        result.update(dict(
            smooth_sanity_fail=True,
            reject_stage="smooth_sanity",
            reject_reason=sanity_reason,
            screen_fail=True,
            J_screen=float("nan"),
        ))
        return result

    # Textured pass
    for eps in eps_values:
        H_tx = pair.textured(eps, Phi, Zm)
        m_tx = solve_case_metrics(H_tx, Phi, phi_1D, Z_1D, d_phi, d_Z,
                                    constants, ps_solver)
        metrics_tx[float(eps)] = m_tx
        r = compute_incremental_ratios(metrics_sm[float(eps)], m_tx)
        ratios_per_eps[float(eps)] = r
        J_per_eps[float(eps)] = compute_J_eps(r)

    classification = classify_profile(J_per_eps, ratios_per_eps)
    result.update(dict(
        smooth_sanity_fail=False,
        metrics_smooth={str(k): v for k, v in metrics_sm.items()},
        metrics_textured={str(k): v for k, v in metrics_tx.items()},
        ratios_per_eps={str(k): v for k, v in ratios_per_eps.items()},
        **classification,
    ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--grid", type=str, default="800x200")
    parser.add_argument("--seed", type=int, default=SCREENING_LHS_SEED)
    parser.add_argument("--n-lhs", type=int,
                        default=SCREENING_N_LHS_PER_FAMILY)
    parser.add_argument("--n-refine", type=int,
                        default=SCREENING_N_REFINE_PER_FAMILY)
    parser.add_argument("--n-refine-bases", type=int,
                        default=SCREENING_N_REFINE_BASES)
    parser.add_argument("--skip-refine", action="store_true",
                        help="отладочный режим: только initial DOE")
    args = parser.parse_args()

    try:
        N_phi_s, N_Z_s = args.grid.split("x")
        N_phi = int(N_phi_s)
        N_Z = int(N_Z_s)
    except Exception:
        print(f"FAIL: --grid format must be WxH, got {args.grid!r}")
        sys.exit(1)

    run_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "coexp", args.run_id, "screening")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Screening run dir: {run_dir}")
    print(f"Grid: {N_phi}x{N_Z}, seed: {args.seed}")

    ps_solver = load_ps_solver()
    relief_fn = load_relief_fn()
    constants = mineral_constants()
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    grid = (phi_1D, Z_1D, Phi, Zm, d_phi, d_Z)

    print("\nCylindrical reference (computed ONCE, clarification 2)...")
    t0 = time.time()
    cyl_ref = cylindrical_reference(
        SCREENING_EPS, Phi, phi_1D, Z_1D, d_phi, d_Z,
        constants, ps_solver,
        sigma_over_c=constants["sigma"] / constants["c"])
    print(f"  done in {time.time()-t0:.1f}s:")
    for eps, m in cyl_ref.items():
        print(f"    eps={eps}: h_min={m['h_min']*1e6:.2f}μm, "
              f"p_max={m['p_max']/1e6:.2f}MPa, fr={m['friction']:.2f}")

    # ── Stage 1: initial DOE ──────────────────────────────────────
    print("\nStage 1 DOE: LHS × families")
    doe = generate_initial_doe(
        seed=args.seed,
        n_per_family=args.n_lhs,
        families=SCREENING_FAMILIES)
    print(f"  {len(doe)} initial samples "
          f"({args.n_lhs} × {len(SCREENING_FAMILIES)} families)")

    # Geometry hard filter (§4.1)
    surv, rej = filter_hard_fail(doe, eps_values=list(SCREENING_EPS))
    print(f"  geometry filter: {len(surv)} survived, {len(rej)} rejected")

    # Solve each survivor
    print("\nSolving initial DOE (6 PS per profile)...")
    records: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, spec in enumerate(surv):
        t0 = time.time()
        r = solve_profile(spec, SCREENING_EPS, grid, constants,
                          ps_solver, relief_fn, cyl_ref)
        r["stage"] = "initial"
        records.append(r)
        dt = time.time() - t0
        tag = "screen_fail" if r.get("screen_fail") else "ok"
        print(f"  [{i+1}/{len(surv)}] {r['family']:>14} "
              f"hash={r['profile_id'][:8]} "
              f"{tag}  J={r.get('J_screen', float('nan')):.4f}  "
              f"{dt:.1f}s")

    # ── Stage 2: local refinement ──────────────────────────────────
    refine_records: List[Dict[str, Any]] = []
    if not args.skip_refine:
        print("\nStage 2 local refinement...")
        # top-2 per family (ТЗ §5)
        bests = select_top_per_family(
            records, args.n_refine_bases, score_key="J_screen")
        print(f"  refinement bases: {[b['profile_id'][:8] for b in bests]}")
        refined_specs = generate_local_refinement(
            [b["profile_spec"] for b in bests],
            n_per_base=args.n_refine,
            seed=args.seed + 1)
        # Geometry pre-filter
        r_surv, r_rej = filter_hard_fail(refined_specs,
                                           eps_values=list(SCREENING_EPS))
        rej.extend(r_rej)
        print(f"  refined samples: {len(refined_specs)} "
              f"({len(r_surv)} survived geom, {len(r_rej)} rejected)")
        for i, spec in enumerate(r_surv):
            t0 = time.time()
            r = solve_profile(spec, SCREENING_EPS, grid, constants,
                              ps_solver, relief_fn, cyl_ref)
            r["stage"] = "refine"
            refine_records.append(r)
            dt = time.time() - t0
            tag = "screen_fail" if r.get("screen_fail") else "ok"
            print(f"  [refine {i+1}/{len(r_surv)}] {r['family']:>14} "
                  f"hash={r['profile_id'][:8]} {tag} "
                  f"J={r.get('J_screen', float('nan')):.4f}  {dt:.1f}s")

    all_records = records + refine_records
    total_time = time.time() - t_start

    # ── Pass-rate status (clarification 5) ────────────────────────
    n_total = len(all_records) + len(rej)
    n_pass = sum(1 for r in all_records if not r.get("screen_fail"))
    status = pass_rate_status(n_total, n_pass)
    print(f"\nPass rate: {n_pass}/{n_total} → {status} "
          f"(ok≥{PASS_RATE_OK}, warn≥{PASS_RATE_WARN})")

    # ── Top-12 selection (hard-fail excluded via T9) ───────────────
    top12 = select_top_global(all_records, n_top=CONFIRM_TOP_N,
                                score_key="J_screen")
    print(f"\nTop-{CONFIRM_TOP_N} global:")
    for i, r in enumerate(top12):
        print(f"  [{i+1}] {r['family']:>14} hash={r['profile_id'][:8]} "
              f"J_screen={r['J_screen']:.4f}  "
              f"params={r['profile_spec']['params']}")

    # ── Artifacts ────────────────────────────────────────────────
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        phase="screening",
        run_id=args.run_id,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        git_sha=git_sha(),
        solver="PS_GPU",
        oil="mineral",
        pv=False,
        reference_texture=REFERENCE_TEXTURE,
        families=SCREENING_FAMILIES,
        screening_eps=list(SCREENING_EPS),
        seed=int(args.seed),
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        hard_geometry=HARD_GEOMETRY,
        smooth_sanity_vs_cyl=SMOOTH_SANITY_VS_CYL,
        cylindrical_reference={str(k): v for k, v in cyl_ref.items()},
        stage1_n_lhs=args.n_lhs,
        stage2_n_refine_per_base=args.n_refine,
        stage2_n_bases=args.n_refine_bases,
        pass_rate=dict(n_total=n_total, n_pass=n_pass, status=status),
        total_time_sec=total_time,
    )
    with open(os.path.join(run_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nmanifest.json: {run_dir}/manifest.json")

    # JSONL — all records + rejects
    jsonl_path = os.path.join(run_dir, "screening_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"screening_results.jsonl: {jsonl_path}")

    rej_path = os.path.join(run_dir, "screening_rejects.jsonl")
    with open(rej_path, "w", encoding="utf-8") as f:
        for r in rej:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV flat
    csv_path = os.path.join(run_dir, "screening_results.csv")
    flat_rows = []
    for r in all_records:
        base = dict(
            profile_id=r["profile_id"], family=r["family"],
            stage=r.get("stage", ""),
            screen_fail=bool(r.get("screen_fail")),
            J_screen=float(r.get("J_screen", float("nan"))),
            n_eps_fail=int(r.get("n_eps_fail", 0)),
        )
        for k, v in r["profile_spec"]["params"].items():
            base[f"param_{k}"] = v
        for eps_key, ratios in r.get("ratios_per_eps", {}).items():
            for rk, rv in ratios.items():
                base[f"{rk}@{eps_key}"] = rv
        flat_rows.append(base)
    fieldnames = sorted({k for row in flat_rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in flat_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"screening_results.csv: {csv_path}")

    # top12 candidates
    top12_path = os.path.join(run_dir, "top12_candidates.json")
    with open(top12_path, "w", encoding="utf-8") as f:
        json.dump(dict(
            schema_version=SCHEMA_VERSION,
            source_manifest=os.path.relpath(
                os.path.join(run_dir, "manifest.json")),
            top_n=CONFIRM_TOP_N,
            score_key="J_screen",
            candidates=top12,
        ), f, indent=2, ensure_ascii=False)
    print(f"top12_candidates.json: {top12_path}")

    print(f"\nDone in {total_time:.1f}s. "
          f"{n_pass}/{n_total} passed, top-{CONFIRM_TOP_N} selected.")


if __name__ == "__main__":
    main()
