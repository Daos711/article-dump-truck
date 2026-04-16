#!/usr/bin/env python3
"""Phase S2 — confirm top-12 на fine grid 1600×400 (ТЗ §6).

Не пересчитывает screening DOE. Читает `top12_candidates.json` и
прогоняет ИХ JE же ε ∈ {0.30, 0.40, 0.50}, и обе пары smooth+textured
на той же reference texture.
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
    SCHEMA_VERSION, REFERENCE_TEXTURE, SCREENING_EPS,
    EQUILIBRIUM_TOP_N,
)
from models.coexp_screening import select_top_global

from _coexp_runtime import (
    load_ps_solver, load_relief_fn, mineral_constants,
    make_grid, git_sha,
)
from run_coexp_screening import solve_profile, cylindrical_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True,
                        help="confirm run_id (also used as output dir)")
    parser.add_argument("--from-screening", type=str, required=True,
                        help="run_id of upstream screening phase")
    parser.add_argument("--grid", type=str, default="1600x400")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..", "results", "coexp")
    src_top12 = os.path.join(base, args.from_screening, "screening",
                              "top12_candidates.json")
    if not os.path.exists(src_top12):
        print(f"FAIL: нет {src_top12}")
        sys.exit(1)
    with open(src_top12, "r", encoding="utf-8") as f:
        top12 = json.load(f)
    if top12.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: top12 schema={top12.get('schema_version')!r}, "
              f"expected {SCHEMA_VERSION!r}")
        sys.exit(1)
    candidates = top12["candidates"]
    print(f"Loaded {len(candidates)} top-12 candidates from "
          f"{args.from_screening}")

    try:
        N_phi_s, N_Z_s = args.grid.split("x")
        N_phi = int(N_phi_s)
        N_Z = int(N_Z_s)
    except Exception:
        print(f"FAIL: --grid format must be WxH, got {args.grid!r}")
        sys.exit(1)

    out_dir = os.path.join(base, args.run_id, "confirm")
    os.makedirs(out_dir, exist_ok=True)

    ps_solver = load_ps_solver()
    relief_fn = load_relief_fn()
    constants = mineral_constants()
    phi_1D, Z_1D, Phi, Zm, d_phi, d_Z = make_grid(N_phi, N_Z)
    grid = (phi_1D, Z_1D, Phi, Zm, d_phi, d_Z)

    print("\nCylindrical reference on fine grid (once, clarification 2)...")
    cyl_ref = cylindrical_reference(
        SCREENING_EPS, Phi, phi_1D, Z_1D, d_phi, d_Z,
        constants, ps_solver,
        sigma_over_c=constants["sigma"] / constants["c"])
    for eps, m in cyl_ref.items():
        print(f"    eps={eps}: h_min={m['h_min']*1e6:.2f}μm, "
              f"p_max={m['p_max']/1e6:.2f}MPa")

    confirmed: List[Dict[str, Any]] = []
    diffs: List[Dict[str, Any]] = []
    print(f"\nConfirming {len(candidates)} candidates on {N_phi}x{N_Z}...")
    t_start = time.time()
    for i, c in enumerate(candidates):
        spec = c["profile_spec"]
        t0 = time.time()
        r = solve_profile(spec, SCREENING_EPS, grid, constants,
                          ps_solver, relief_fn, cyl_ref)
        r["from_screening_run"] = args.from_screening
        r["screening_J"] = float(c.get("J_screen", float("nan")))
        confirmed.append(r)
        # coarse-vs-fine diff (ТЗ §13.10/11)
        d = dict(profile_id=r["profile_id"],
                 J_coarse=float(c.get("J_screen", float("nan"))),
                 J_fine=float(r.get("J_screen", float("nan"))))
        # average ratio diff across eps
        rs_c = c.get("ratios_per_eps", {}) or {}
        rs_f = r.get("ratios_per_eps", {}) or {}
        if rs_c and rs_f:
            keys = sorted(set(rs_c.keys()) & set(rs_f.keys()))
            diffs_h = []
            diffs_p = []
            for k in keys:
                if isinstance(rs_c[k], dict) and isinstance(rs_f[k], dict):
                    diffs_h.append(abs(rs_f[k]["h_r"] - rs_c[k]["h_r"]))
                    diffs_p.append(abs(rs_f[k]["p_r"] - rs_c[k]["p_r"]))
            d["mean_abs_dh_r"] = (
                float(sum(diffs_h) / len(diffs_h)) if diffs_h else None)
            d["mean_abs_dp_r"] = (
                float(sum(diffs_p) / len(diffs_p)) if diffs_p else None)
        diffs.append(d)
        dt = time.time() - t0
        tag = "screen_fail" if r.get("screen_fail") else "ok"
        print(f"  [{i+1}/{len(candidates)}] hash={r['profile_id'][:8]} "
              f"{tag}  J_coarse={c.get('J_screen', float('nan')):.4f} "
              f"→ J_fine={r.get('J_screen', float('nan')):.4f}  "
              f"{dt:.1f}s")

    # Top-6 confirmed
    top6 = select_top_global(confirmed, n_top=EQUILIBRIUM_TOP_N,
                              score_key="J_screen")
    print(f"\nTop-{EQUILIBRIUM_TOP_N} confirmed:")
    for i, r in enumerate(top6):
        print(f"  [{i+1}] hash={r['profile_id'][:8]} "
              f"J_fine={r['J_screen']:.4f}")

    total = time.time() - t_start
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        phase="confirm",
        run_id=args.run_id,
        from_screening=args.from_screening,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        git_sha=git_sha(),
        oil="mineral", pv=False,
        reference_texture=REFERENCE_TEXTURE,
        screening_eps=list(SCREENING_EPS),
        grid=dict(N_phi=N_phi, N_Z=N_Z),
        cylindrical_reference={str(k): v for k, v in cyl_ref.items()},
        n_candidates=len(candidates),
        n_confirmed=len(confirmed),
        coarse_fine_diff=diffs,
        total_time_sec=total,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = os.path.join(out_dir, "confirm_results.csv")
    rows = []
    for r in confirmed:
        row = dict(
            profile_id=r["profile_id"], family=r["family"],
            screen_fail=bool(r.get("screen_fail")),
            J_fine=float(r.get("J_screen", float("nan"))),
            J_coarse=float(r.get("screening_J", float("nan"))),
        )
        for k, v in r["profile_spec"]["params"].items():
            row[f"param_{k}"] = v
        for eps_key, ratios in r.get("ratios_per_eps", {}).items():
            for rk, rv in ratios.items():
                row[f"{rk}@{eps_key}"] = rv
        rows.append(row)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"confirm_results.csv: {csv_path}")

    top6_path = os.path.join(out_dir, "top6_confirmed.json")
    with open(top6_path, "w", encoding="utf-8") as f:
        json.dump(dict(
            schema_version=SCHEMA_VERSION,
            from_confirm_run=args.run_id,
            top_n=EQUILIBRIUM_TOP_N,
            score_key="J_screen",
            candidates=top6,
        ), f, indent=2, ensure_ascii=False)
    print(f"top6_confirmed.json: {top6_path}")

    # Coarse-vs-fine flag
    big_jumps = [d for d in diffs
                  if (d.get("mean_abs_dh_r") or 0.0) > 0.02
                  or (d.get("mean_abs_dp_r") or 0.0) > 0.02]
    if big_jumps:
        print(f"\n[WARN] {len(big_jumps)} confirmed profiles have "
              f"|Δ ratio| > 2% between coarse and fine grids — "
              f"flagged in manifest.coarse_fine_diff (do NOT hide).")


if __name__ == "__main__":
    main()
