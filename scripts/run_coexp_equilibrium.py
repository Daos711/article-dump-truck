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


SCHEMA_DIAGNOSTIC = "coexp_v1.1"


def wy_tag(wy: float) -> str:
    """'0.05' → '005', '0.10' → '010', '0.25' → '025'. Консистентно с
    ТЗ coexp_v1.1 §6.2 (подкаталоги equilibrium_wy<XXX>/)."""
    return f"{int(round(float(wy) * 100)):03d}"


def run_equilibrium_for_wy(
        *, candidates, Wy_share, tol_accept, max_iter, grid,
        out_dir, run_id, from_confirm, phase_tag, schema_version,
        ps_solver, relief_fn, constants,
) -> Dict[str, Any]:
    """Выполнить equilibrium solve по списку candidates для ОДНОГО Wy.

    Возвращает полный summary dict. Записывает manifest.json,
    equilibrium_summary.json, equilibrium_results.csv под `out_dir`.
    """
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = grid
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Wy={Wy_share}] out_dir={out_dir}")
    print(f"  tol_accept={tol_accept}, max_iter_nr={max_iter}, "
          f"grid={Phi.shape[1]}x{Phi.shape[0]}")
    print(f"  NR seed FIXED: (X0={EQUILIBRIUM_NR_SEED_X0}, "
          f"Y0={EQUILIBRIUM_NR_SEED_Y0})")

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
            constants["F0"], Wy_share,
            ps_solver=ps_solver, texture_relief_fn=relief_fn,
            tol_accept=tol_accept,
            step_cap=DEFAULT_STEP_CAP, eps_max=DEFAULT_EPS_MAX,
            max_iter=max_iter,
        )
        ratios = _ratios_from_pair(result)
        result["ratios"] = ratios
        if ratios:
            result["J_eq"] = compute_J_eq(ratios)
            result["useful"] = bool(equilibrium_useful(ratios))
        else:
            result["J_eq"] = None
            result["useful"] = False
        result["pair_hash_matches"] = (
            result["profile_hash"] == exp.profile_id)
        pairs.append(result)
        dt = time.time() - t0
        useful_tag = "USEFUL" if result["useful"] else "—"
        accepted_tag = (
            f"sm:{'✓' if result['smooth_accepted'] else '✗'}"
            f"/tx:{'✓' if result['textured_accepted'] else '✗'}")
        if ratios:
            print(f"  [{i+1}/{len(candidates)}] "
                  f"hash={result['profile_hash'][:8]} "
                  f"{accepted_tag} h_r={ratios['h_r']:.4f} "
                  f"p_r={ratios['p_r']:.4f} J_eq={result['J_eq']:.4f} "
                  f"{useful_tag} {dt:.1f}s")
        else:
            print(f"  [{i+1}/{len(candidates)}] "
                  f"hash={result['profile_hash'][:8]} "
                  f"{accepted_tag} (ratios n/a — accept failed) {dt:.1f}s")

    n_useful = sum(1 for p in pairs if p["useful"])
    summary = dict(
        schema_version=schema_version,
        phase=phase_tag,
        run_id=run_id,
        from_confirm=from_confirm,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        git_sha=git_sha(),
        oil="mineral", pv=False,
        reference_texture=REFERENCE_TEXTURE,
        grid=dict(N_phi=Phi.shape[1], N_Z=Phi.shape[0]),
        Wy_share=float(Wy_share),
        nr_seed=dict(X0=EQUILIBRIUM_NR_SEED_X0, Y0=EQUILIBRIUM_NR_SEED_Y0),
        tol_accept=float(tol_accept),
        tol_accept_effective=float(tol_accept),
        max_iter_nr=int(max_iter),
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
            schema_version=schema_version,
            phase=phase_tag,
            run_id=run_id,
            from_confirm=from_confirm,
            created_utc=summary["created_utc"],
            git_sha=summary["git_sha"],
            grid=summary["grid"],
            Wy_share=summary["Wy_share"],
            nr_seed=summary["nr_seed"],
            tol_accept=summary["tol_accept"],
            tol_accept_effective=summary["tol_accept_effective"],
            max_iter_nr=summary["max_iter_nr"],
        ), f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "equilibrium_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → equilibrium_summary.json ({len(pairs)} pairs, "
          f"{n_useful} useful)")

    # CSV
    csv_path = os.path.join(out_dir, "equilibrium_results.csv")
    rows = []
    for p in pairs:
        row = dict(
            profile_id=p["profile_hash"],
            experiment_id=p["experiment_id"],
            family=p["profile_spec"]["family"],
            Wy_share=float(Wy_share),
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

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--from-confirm", type=str, required=True)
    parser.add_argument("--Wy-share", type=float, default=DEFAULT_WY_SHARE)
    parser.add_argument("--Wy-share-list", type=str, default=None,
                        help="comma-separated Wy_share list (диагностика). "
                             "Если задан — игнорирует --Wy-share, "
                             "итерирует и пишет в equilibrium_wy<XXX>/.")
    parser.add_argument("--grid", type=str, default="800x200")
    parser.add_argument("--phase", choices=["E1", "E2"], default="E1")
    parser.add_argument("--tol-accept", type=float, default=DEFAULT_TOL_ACCEPT)
    parser.add_argument("--tol-accept-override", type=float, default=None,
                        help="override tol_accept для equilibrium phase. "
                             "Значение записывается в manifest как "
                             "tol_accept_effective.")
    parser.add_argument("--max-iter-nr", type=int, default=80,
                        help="max NR итераций (default 80). Диагностика "
                             "может потребовать 200+.")
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
        # E1 pair dicts use `profile_hash`; top6 candidates use
        # `profile_id`. Обе величины это один и тот же hash (см.
        # make_experiment_spec → profile_id = profile.hash()).
        useful_ids = {p["profile_hash"] for p in e1["pairs"]
                      if p.get("useful")}
        if not useful_ids:
            print("E2 skipped: нет useful candidates в E1.")
            sys.exit(0)
        sorted_e1 = sorted(
            [p for p in e1["pairs"] if p.get("useful")],
            key=lambda p: (p.get("J_eq") or 0.0), reverse=True)
        top3_ids = {p["profile_hash"]
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

    # Diagnostic mode iff any off-spec override is set. ТЗ coexp_v1.1:
    # в diagnostic mode artifacts пишутся в equilibrium_wy<XXX>/ и
    # помечаются schema coexp_v1.1 — чтобы не смешать с baseline.
    wy_list = None
    if args.Wy_share_list:
        try:
            wy_list = [float(x) for x in args.Wy_share_list.split(",")
                       if x.strip()]
        except ValueError as e:
            print(f"FAIL: не могу распарсить --Wy-share-list: {e}")
            sys.exit(1)
        if not wy_list:
            print("FAIL: --Wy-share-list пуст")
            sys.exit(1)

    tol_accept_effective = (float(args.tol_accept_override)
                             if args.tol_accept_override is not None
                             else float(args.tol_accept))
    max_iter_nr = int(args.max_iter_nr)

    is_diagnostic = (wy_list is not None
                      or args.tol_accept_override is not None
                      or max_iter_nr != 80)

    if is_diagnostic and args.phase == "E2":
        print("FAIL: diagnostic mode (--Wy-share-list / "
              "--tol-accept-override / --max-iter-nr != 80) не совместим "
              "с --phase E2. E2 read'ает baseline equilibrium/ summary; "
              "запускай E2 отдельно после non-diagnostic E1.")
        sys.exit(1)

    schema_version = SCHEMA_DIAGNOSTIC if is_diagnostic else SCHEMA_VERSION
    phase_tag = f"equilibrium_{args.phase}"
    if is_diagnostic:
        phase_tag += "_diagnostic"

    ps_solver = load_ps_solver()
    relief_fn = load_relief_fn()
    constants = mineral_constants()
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    grid = (phi_1D, Z_1D, Phi, Zm, d_phi, d_Z)

    print(f"\nEquilibrium phase={args.phase}, grid={N_phi}x{N_Z}, "
          f"mode={'diagnostic' if is_diagnostic else 'baseline'}, "
          f"schema={schema_version}")

    # Iterate over Wy values (single or list).
    wy_values = wy_list if wy_list is not None else [float(args.Wy_share)]
    summaries_by_wy: Dict[str, Dict[str, Any]] = {}
    for wy in wy_values:
        if is_diagnostic:
            subdir = f"equilibrium_wy{wy_tag(wy)}"
        else:
            subdir = "equilibrium"
        out_dir = os.path.join(base, args.run_id, subdir)
        summary = run_equilibrium_for_wy(
            candidates=candidates,
            Wy_share=wy,
            tol_accept=tol_accept_effective,
            max_iter=max_iter_nr,
            grid=grid,
            out_dir=out_dir,
            run_id=args.run_id,
            from_confirm=args.from_confirm,
            phase_tag=phase_tag,
            schema_version=schema_version,
            ps_solver=ps_solver,
            relief_fn=relief_fn,
            constants=constants,
        )
        summaries_by_wy[f"{wy:.3f}"] = summary

    # Print final roll-up across all Wy
    print("\n" + "=" * 60)
    print("Summary across Wy values:")
    for key, s in summaries_by_wy.items():
        print(f"  Wy={key}: {s['n_useful']}/{s['n_pairs']} useful")
    any_useful_total = sum(s["n_useful"] for s in summaries_by_wy.values())
    if any_useful_total == 0:
        print("Ни одного useful — honest negative, infrastructure PASS.")


if __name__ == "__main__":
    main()
