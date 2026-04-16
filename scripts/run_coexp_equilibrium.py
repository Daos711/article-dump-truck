#!/usr/bin/env python3
"""Phase E1/E2 — equilibrium на applied load (ТЗ §6).

E1: top-6 confirmed → 800×200, Wy/F0 = 0.25.
E2 (only if E1 has ≥1 useful candidate): top-3 → 1600×400.

Пара smooth/textured equilibrium для ОДНОГО bore profile использует
один и тот же hash (T6/T13). NR seed FIXED — (0, -0.4) (clarification 4).
"""
from __future__ import annotations

import argparse
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

from models.coexp_schema import (
    SCHEMA_VERSION, REFERENCE_TEXTURE,
    DEFAULT_WY_SHARE, DEFAULT_TOL_ACCEPT, DEFAULT_STEP_CAP, DEFAULT_EPS_MAX,
    EQUILIBRIUM_NR_SEED_X0, EQUILIBRIUM_NR_SEED_Y0,
    EQUILIBRIUM_TOP_N, EQUILIBRIUM_FINE_TOP_N,
    make_experiment_spec,
)
from models.coexp_objective import (
    compute_incremental_ratios, compute_J_eq, equilibrium_useful,
)
from models.coexp_equilibrium import solve_equilibrium_pair

from _coexp_runtime import (
    load_ps_solver, load_relief_fn, mineral_constants, make_grid, git_sha,
)


def _ratios_from_pair(pair_dict: Dict[str, Any]) -> Dict[str, float]:
    s = pair_dict["smooth"]
    t = pair_dict["textured"]
    if s is None or t is None:
        return {}
    return compute_incremental_ratios(
        dict(h_min=s["h_min"], p_max=s["p_max"],
             friction=s["friction"], cav_frac=s["cav_frac"]),
        dict(h_min=t["h_min"], p_max=t["p_max"],
             friction=t["friction"], cav_frac=t["cav_frac"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--from-confirm", type=str, required=True)
    parser.add_argument("--Wy-share", type=float, default=DEFAULT_WY_SHARE)
    parser.add_argument("--grid", type=str, default="800x200")
    parser.add_argument("--phase", choices=["E1", "E2"], default="E1")
    parser.add_argument("--tol-accept", type=float, default=DEFAULT_TOL_ACCEPT)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..", "results", "coexp")
    src = os.path.join(base, args.from_confirm, "confirm",
                        "top6_confirmed.json")
    if not os.path.exists(src):
        print(f"FAIL: нет {src}")
        sys.exit(1)
    with open(src, "r", encoding="utf-8") as f:
        top6 = json.load(f)
    if top6.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: top6 schema={top6.get('schema_version')!r}")
        sys.exit(1)
    candidates = top6["candidates"]

    # E2: pre-filter from prior E1
    if args.phase == "E2":
        e1_summary = os.path.join(
            base, args.from_confirm, "equilibrium",
            "equilibrium_summary.json")
        if not os.path.exists(e1_summary):
            print(f"FAIL E2: нет E1 summary {e1_summary}")
            sys.exit(1)
        with open(e1_summary, "r", encoding="utf-8") as f:
            e1 = json.load(f)
        useful_ids = {p["profile_id"] for p in e1["pairs"]
                      if p.get("useful")}
        if not useful_ids:
            print("E2 skipped: нет useful candidates в E1.")
            sys.exit(0)
        sorted_e1 = sorted(
            [p for p in e1["pairs"] if p.get("useful")],
            key=lambda p: p.get("J_eq", 0.0), reverse=True)
        top3_ids = {p["profile_id"]
                     for p in sorted_e1[:EQUILIBRIUM_FINE_TOP_N]}
        candidates = [c for c in candidates
                      if c["profile_id"] in top3_ids]
        print(f"E2: top-{EQUILIBRIUM_FINE_TOP_N} useful "
              f"({len(candidates)} candidates)")

    try:
        N_phi_s, N_Z_s = args.grid.split("x")
        N_phi = int(N_phi_s)
        N_Z = int(N_Z_s)
    except Exception:
        print(f"FAIL: --grid format must be WxH, got {args.grid!r}")
        sys.exit(1)

    out_dir = os.path.join(base, args.run_id, "equilibrium")
    os.makedirs(out_dir, exist_ok=True)

    ps_solver = load_ps_solver()
    relief_fn = load_relief_fn()
    constants = mineral_constants()
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)

    print(f"\nEquilibrium phase={args.phase}, grid={N_phi}x{N_Z}, "
          f"Wy_share={args.Wy_share}")
    print(f"NR seed FIXED: (X0={EQUILIBRIUM_NR_SEED_X0}, "
          f"Y0={EQUILIBRIUM_NR_SEED_Y0}) — clarification 4")

    pairs: List[Dict[str, Any]] = []
    t_start = time.time()
    for i, c in enumerate(candidates):
        spec = c["profile_spec"]
        exp = make_experiment_spec(spec, REFERENCE_TEXTURE)
        t0 = time.time()
        result = solve_equilibrium_pair(
            exp, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
            constants["R"], constants["L"], constants["c"],
            constants["sigma"], constants["eta"], constants["omega"],
            constants["F0"], args.Wy_share,
            ps_solver=ps_solver, texture_relief_fn=relief_fn,
            tol_accept=args.tol_accept,
            step_cap=DEFAULT_STEP_CAP, eps_max=DEFAULT_EPS_MAX,
        )
        ratios = _ratios_from_pair(result)
        result["ratios"] = ratios
        if ratios:
            result["J_eq"] = compute_J_eq(ratios)
            result["useful"] = bool(equilibrium_useful(ratios))
        else:
            result["J_eq"] = None
            result["useful"] = False
        # T13 bit-check: smooth and textured must share profile_hash
        result["pair_hash_matches"] = (result["profile_hash"]
                                        == exp.profile_id)
        pairs.append(result)
        dt = time.time() - t0
        useful_tag = "USEFUL" if result["useful"] else "—"
        accepted_tag = (
            f"sm:{'✓' if result['smooth_accepted'] else '✗'}"
            f"/tx:{'✓' if result['textured_accepted'] else '✗'}")
        if ratios:
            print(f"  [{i+1}/{len(candidates)}] hash={result['profile_hash'][:8]} "
                  f"{accepted_tag} h_r={ratios['h_r']:.4f} "
                  f"p_r={ratios['p_r']:.4f} J_eq={result['J_eq']:.4f} "
                  f"{useful_tag} {dt:.1f}s")
        else:
            print(f"  [{i+1}/{len(candidates)}] hash={result['profile_hash'][:8]} "
                  f"{accepted_tag} (ratios n/a — accept failed) {dt:.1f}s")

    n_useful = sum(1 for p in pairs if p["useful"])
    summary = dict(
        schema_version=SCHEMA_VERSION,
        phase=f"equilibrium_{args.phase}",
        run_id=args.run_id,
        from_confirm=args.from_confirm,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        git_sha=git_sha(),
        oil="mineral", pv=False,
        reference_texture=REFERENCE_TEXTURE,
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        Wy_share=float(args.Wy_share),
        nr_seed=dict(X0=EQUILIBRIUM_NR_SEED_X0, Y0=EQUILIBRIUM_NR_SEED_Y0),
        tol_accept=float(args.tol_accept),
        step_cap=float(DEFAULT_STEP_CAP),
        eps_max=float(DEFAULT_EPS_MAX),
        n_pairs=len(pairs),
        n_useful=n_useful,
        any_useful=bool(n_useful > 0),
        pairs=pairs,
        total_time_sec=time.time() - t_start,
    )

    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(
            schema_version=SCHEMA_VERSION,
            phase=f"equilibrium_{args.phase}",
            run_id=args.run_id,
            from_confirm=args.from_confirm,
            created_utc=summary["created_utc"],
            git_sha=summary["git_sha"],
            grid=summary["grid"],
            Wy_share=summary["Wy_share"],
            nr_seed=summary["nr_seed"],
            tol_accept=summary["tol_accept"],
        ), f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "equilibrium_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"equilibrium_summary.json: {out_dir}/equilibrium_summary.json")

    csv_path = os.path.join(out_dir, "equilibrium_results.csv")
    rows = []
    for p in pairs:
        row = dict(
            profile_id=p["profile_hash"],
            experiment_id=p["experiment_id"],
            family=p["profile_spec"]["family"],
            useful=bool(p["useful"]),
            smooth_accepted=bool(p["smooth_accepted"]),
            textured_accepted=bool(p["textured_accepted"]),
            J_eq=p.get("J_eq"),
        )
        if p.get("ratios"):
            for k, v in p["ratios"].items():
                row[k] = v
        for k, v in p["profile_spec"]["params"].items():
            row[f"param_{k}"] = v
        rows.append(row)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"equilibrium_results.csv: {csv_path}")

    print(f"\nUseful candidates (h_r>1.005, p_r≤1.000, "
          f"f_r≤1.02, c_d≤0.02): {n_useful}/{len(pairs)}")
    if n_useful == 0:
        print("Negative result honestly reported. Infrastructure PASS, "
              "physics NEGATIVE.")


if __name__ == "__main__":
    main()
