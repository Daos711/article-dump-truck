#!/usr/bin/env python3
"""Select top-6 candidates from geometry rescue screen."""
import argparse, csv, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from cases.gu_loaded_side_v4.schema import SCHEMA

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--pairs-csv", required=True)
    pa.add_argument("--n-top", type=int, default=6)
    pa.add_argument("--out", required=True)
    args = pa.parse_args()

    with open(args.pairs_csv) as f:
        rows = list(csv.DictReader(f))

    filtered = []
    for r in rows:
        st = r.get("status", "")
        if st not in ("hard_converged", "soft_converged"):
            continue
        dCOF = float(r["dCOF_pct"])
        dh = float(r["dh_pct"])
        dg_h = float(r["dg_over_hmin"])
        dp = float(r["dp_pct"])
        rr = float(r["rel_residual"])
        if dCOF > -2.0 or dh < -3.0 or dg_h > 3.0 or dp > 35.0 or rr > 0.02:
            continue
        score = (dCOF
                 + 2.0 * max(0, -dh - 2.0)
                 + 0.10 * max(0, dp - 25.0)
                 + 1.0 * max(0, dg_h - 2.5))
        filtered.append(dict(**r, score=score))

    filtered.sort(key=lambda x: x["score"])
    top = filtered[:args.n_top]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(
            schema_version=SCHEMA,
            n_filtered=len(filtered),
            n_selected=len(top),
            candidates=top,
        ), f, indent=2, ensure_ascii=False)
    print(f"Selected {len(top)}/{len(filtered)} filtered/{len(rows)} total")
    for i, c in enumerate(top):
        print(f"  [{i+1}] {c['variant']} N={c['N_branch']} dg={c['d_g_um']} "
              f"t={c['taper_ratio']} k={c['curvature_k']} "
              f"eps={c['eps_ref']} ΔCOF={float(c['dCOF_pct']):+.1f}% "
              f"score={c['score']:.2f}")

if __name__ == "__main__":
    main()
