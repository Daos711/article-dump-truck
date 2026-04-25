#!/usr/bin/env python3
"""Plot Stage I-A cycle results: eps, hmin, Ploss overlays."""
from __future__ import annotations
import argparse, csv, math, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-dir", required=True)
    args = pa.parse_args()

    cp = os.path.join(args.data_dir, "cycle_history.csv")
    if not os.path.exists(cp):
        print(f"FAIL: нет {cp}")
        sys.exit(1)
    rows = load_csv(cp)

    sm = [r for r in rows if r["geometry"] == "smooth"]
    tx = [r for r in rows if r["geometry"] == "textured"]

    def arr(rs, key):
        return np.array([float(r[key]) for r in rs])

    def status_ok(rs):
        return [r["status"] != "failed" for r in rs]

    phi_sm = arr(sm, "phi_crank_deg")
    phi_tx = arr(tx, "phi_crank_deg")
    ok_sm = status_ok(sm)
    ok_tx = status_ok(tx)

    # Matched mask
    sm_ok_set = {float(r["phi_crank_deg"]) for r in sm if r["status"] != "failed"}
    tx_ok_set = {float(r["phi_crank_deg"]) for r in tx if r["status"] != "failed"}
    matched_set = sm_ok_set & tx_ok_set

    def plot_overlay(key, ylabel, fname, scale=1.0):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ys = arr(sm, key) * scale
        yt = arr(tx, key) * scale
        ax.plot(phi_sm, ys, "b-", lw=1, alpha=0.3, label="smooth (all)")
        ax.plot(phi_tx, yt, "r-", lw=1, alpha=0.3, label="textured (all)")
        # Accepted only
        ax.plot(phi_sm[ok_sm], ys[ok_sm], "bo", ms=4, label="smooth (acc)")
        ax.plot(phi_tx[ok_tx], yt[ok_tx], "r^", ms=4, label="textured (acc)")
        # Shade matched
        for phi in sorted(matched_set):
            ax.axvspan(phi - 5, phi + 5, alpha=0.08, color="green")
        ax.set_xlabel("φ_crank (deg)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 720)
        fig.tight_layout()
        fig.savefig(os.path.join(args.data_dir, fname), dpi=150)
        plt.close(fig)
        print(f"→ {fname}")

    plot_overlay("eps", "ε", "eps_vs_phi.png")
    plot_overlay("h_min_um", "h_min (μm)", "hmin_vs_phi.png")
    plot_overlay("Ploss_W", "P_loss (W)", "ploss_vs_phi.png")

    # Load hodograph
    fig, ax = plt.subplots(figsize=(6, 6))
    Wx = arr(sm, "Wx_N")
    Wy = arr(sm, "Wy_N")
    ax.plot(Wx, Wy, "k-", lw=1)
    ax.plot(Wx[0], Wy[0], "go", ms=8, label="φ=0°")
    ax.set_xlabel("Wx (N)")
    ax.set_ylabel("Wy (N)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.data_dir, "load_hodograph.png"), dpi=150)
    plt.close(fig)
    print("→ load_hodograph.png")

    # Wx, Wy vs phi
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(phi_sm, Wx, "b-", lw=1.5, label="Wx")
    ax.plot(phi_sm, Wy, "r-", lw=1.5, label="Wy")
    ax.set_xlabel("φ_crank (deg)")
    ax.set_ylabel("W (N)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 720)
    fig.tight_layout()
    fig.savefig(os.path.join(args.data_dir, "wx_wy_vs_phi.png"), dpi=150)
    plt.close(fig)
    print("→ wx_wy_vs_phi.png")

    # Compare CSV
    compare = []
    for s, t in zip(sm, tx):
        phi = float(s["phi_crank_deg"])
        matched = phi in matched_set
        compare.append(dict(
            phi_crank_deg=phi,
            matched=matched,
            smooth_status=s["status"],
            textured_status=t["status"],
            smooth_eps=float(s["eps"]),
            textured_eps=float(t["eps"]),
            smooth_Ploss=float(s["Ploss_W"]),
            textured_Ploss=float(t["Ploss_W"]),
            smooth_hmin=float(s["h_min_um"]),
            textured_hmin=float(t["h_min_um"]),
            dPloss_pct=((float(t["Ploss_W"]) - float(s["Ploss_W"]))
                         / max(float(s["Ploss_W"]), 1e-20) * 100
                         if matched else None),
        ))
    ccp = os.path.join(args.data_dir, "compare_smooth_vs_textured.csv")
    if compare:
        flds = sorted(compare[0].keys())
        with open(ccp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            for r in compare:
                w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                            for k, v in r.items()})
    print(f"→ compare_smooth_vs_textured.csv")


if __name__ == "__main__":
    main()
