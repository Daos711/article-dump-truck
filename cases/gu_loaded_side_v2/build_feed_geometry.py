#!/usr/bin/env python3
"""Stage A_v2 — build and visualize feed-consistent geometry.

Plots unfolded H(phi,Z) for given variant/depth/belt.
Smoke checks: d_g=0 reproduces smooth, no double-depth, symmetry.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.feed_geometry import (
    create_H_with_central_feed_branches,
    feed_geometry_params,
    feed_window_metadata,
)
from cases.gu_loaded_side_v2.schema import SCHEMA
from cases.gu_loaded_side_v2.common import D, R, L, c


def make_grid(N_phi=800, N_Z=200):
    phi = np.linspace(0, 2 * math.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["straight", "half_herringbone"],
                        default="straight")
    parser.add_argument("--dg", type=float, default=10,
                        help="groove depth in μm")
    parser.add_argument("--belt", type=float, default=0.15,
                        help="belt_half_frac (fraction of L/2)")
    parser.add_argument("--beta", type=float, default=20.0,
                        help="branch angle (only for half_herringbone)")
    parser.add_argument("--N-branch", type=int, default=3)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d_g_m = args.dg * 1e-6
    params = feed_geometry_params(
        d_g_m, c, N_branch=args.N_branch, w_g_m=0.004, R_m=R,
        belt_half_frac=args.belt,
        beta_deg=args.beta if args.variant == "half_herringbone" else 0.0,
        variant=args.variant)
    feed_meta = feed_window_metadata(z_belt_half=args.belt)

    phi, Z, Phi, Zm = make_grid()
    eps = 0.5
    H0 = 1.0 + eps * np.cos(Phi)
    H = create_H_with_central_feed_branches(H0, **{
        k: params[k] for k in ("depth_nondim", "N_branch_per_side",
                                 "w_branch_nondim", "belt_half_nondim",
                                 "beta_deg", "variant")},
        Phi=Phi, Z=Zm)

    # ── Smoke checks ─────────────────────────────────────────────
    print(f"Stage A_v2: feed geometry builder")
    print(f"  variant={args.variant}, d_g={args.dg}μm, "
          f"belt={args.belt}, N_branch={args.N_branch}")

    # 1. d_g=0 → identical to H0
    H_zero = create_H_with_central_feed_branches(
        H0, 0.0, Phi, Zm,
        params["N_branch_per_side"], params["w_branch_nondim"],
        params["belt_half_nondim"], params["beta_deg"],
        params["variant"])
    check1 = np.array_equal(H_zero, H0)
    print(f"  [{'✓' if check1 else '✗'}] d_g=0 reproduces smooth")

    # 2. No double-depth
    relief = H - H0
    max_relief = float(np.max(relief))
    check2 = max_relief <= params["depth_nondim"] + 1e-12
    print(f"  [{'✓' if check2 else '✗'}] max relief = "
          f"{max_relief:.6f} ≤ {params['depth_nondim']:.6f}")

    # 3. Symmetry Z ↔ -Z
    relief_flip = relief[::-1, :]
    corr = np.corrcoef(relief.ravel(), relief_flip.ravel())[0, 1]
    check3 = corr > 0.99
    print(f"  [{'✓' if check3 else '✗'}] Z-symmetry correlation = "
          f"{corr:.4f}")

    # 4. Belt zone groove-free
    belt_mask = np.abs(Zm) <= params["belt_half_nondim"]
    belt_relief = relief[belt_mask]
    check4 = float(np.max(np.abs(belt_relief))) < 1e-12
    print(f"  [{'✓' if check4 else '✗'}] belt zone groove-free")

    overall = check1 and check2 and check3 and check4
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(np.degrees(Phi), Zm, H * c * 1e6,
                        shading="auto", cmap="viridis")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("h (μm)")
    ax.set_xlabel("φ (deg)")
    ax.set_ylabel("Z (nondim)")
    ax.axhline(params["belt_half_nondim"], color="white", ls="--",
               lw=0.8, alpha=0.7)
    ax.axhline(-params["belt_half_nondim"], color="white", ls="--",
               lw=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "feed_geometry_H.png"), dpi=150)
    plt.close(fig)

    # ── Manifest ──────────────────────────────────────────────────
    manifest = dict(
        schema_version=SCHEMA,
        stage="A_v2_feed_geometry",
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        geometry_params=params,
        feed_window=feed_meta,
        smoke_checks=dict(
            dg_zero_smooth=bool(check1),
            no_double_depth=bool(check2),
            z_symmetry=float(corr),
            belt_groove_free=bool(check4),
        ),
        overall_pass=bool(overall),
    )
    with open(os.path.join(args.out, "stageA_v2_geometry_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nArtifacts: {args.out}/")


if __name__ == "__main__":
    main()
