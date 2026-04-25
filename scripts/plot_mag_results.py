#!/usr/bin/env python3
"""Графики + MD report для magnetic unloading exploratory case.

Читает ТОЛЬКО согласованную пару файлов одного run_id (ТЗ §3.7, §4.4):
  results/magnetic_pump/<run_id>/manifest.json          — required
  results/magnetic_pump/<run_id>/textured_compare.json  — optional

Schema: magnetic_v4.
Legacy keys (`mag_share_target`, `hydro_load_share`, `mag_load_share`,
`sector_angles_deg`) и legacy файлы (`mag_smooth_summary.json`,
`mag_textured_summary.json`, `pmax_vs_mag_share.png`,
`ratios_tex_vs_smooth.png` в flat-directory) НЕ читаются.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEMA_VERSION = "magnetic_v4"

LEGACY_KEYS = (
    "mag_share_target",
    "hydro_load_share",
    "mag_load_share",
    "sector_angles_deg",
)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_schema(doc, path):
    """Schema mismatch → FAIL (ТЗ §4.4.3). Никакого silent degrade."""
    if doc is None:
        return
    if doc.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: {path} schema_version="
              f"{doc.get('schema_version')!r}, expected "
              f"{SCHEMA_VERSION!r}. Regenerate via "
              f"run_mag_smooth_continuation.py")
        sys.exit(1)
    # Явная защита от legacy keys
    legacy_found = [k for k in LEGACY_KEYS if k in doc]
    if legacy_found:
        print(f"FAIL: {path} contains legacy keys {legacy_found}. "
              f"Use fresh magnetic_v4 run.")
        sys.exit(1)


def resolve_run_dir(args):
    base = os.path.join(os.path.dirname(__file__), "..",
                        "results", "magnetic_pump")
    if args.data_dir:
        return os.path.abspath(args.data_dir)
    if args.run_id:
        return os.path.join(base, args.run_id)
    latest = os.path.join(base, "latest_run.txt")
    if os.path.exists(latest):
        with open(latest, "r", encoding="utf-8") as f:
            rid = f.read().strip()
        return os.path.join(base, rid)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                        help="explicit run directory")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args)
    if run_dir is None or not os.path.isdir(run_dir):
        print("Нет run директории. Запусти smooth continuation сначала.")
        sys.exit(1)
    print(f"Run dir: {run_dir}")

    manifest_path = os.path.join(run_dir, "manifest.json")
    sm = load_json(manifest_path)
    if sm is None:
        print(f"FAIL: отсутствует {manifest_path}")
        sys.exit(1)
    assert_schema(sm, manifest_path)

    tx_path = os.path.join(run_dir, "textured_compare.json")
    tx = load_json(tx_path)
    assert_schema(tx, tx_path)

    smooth_runs = sm.get("smooth_accepted", [])
    if not smooth_runs:
        print("В manifest нет smooth_accepted")
        sys.exit(1)

    out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    targets = np.array([r["unload_share_target"] for r in smooth_runs]) * 100

    def col(rs, key):
        return np.array([r[key] for r in rs])

    eps_s = col(smooth_runs, "eps")
    hmin_s = col(smooth_runs, "h_min") * 1e6
    pmax_s = col(smooth_runs, "p_max") / 1e6
    hydro_s = col(smooth_runs, "hydro_share_actual")
    unload_s = col(smooth_runs, "unload_share_actual")

    # accepted textured pairs (may be empty — тогда tex-графики не строим)
    tex_pairs = []
    if tx is not None:
        tex_pairs = [p for p in tx.get("pairs", [])
                     if p.get("accepted") and p.get("textured")]
    tex_available = len(tex_pairs) > 0

    # 1. eps_eq vs unload_share_target
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, eps_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tex_available:
        tex_eps = [p["textured"]["eps"] for p in tex_pairs]
        tex_t = [p["unload_share_target"] * 100 for p in tex_pairs]
        ax.plot(tex_t, tex_eps, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("ε_eq")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eps_vs_unload_share.png"), dpi=150)
    plt.close(fig)

    # 2. h_min
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, hmin_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tex_available:
        tex_h = [p["textured"]["h_min"] * 1e6 for p in tex_pairs]
        tex_t = [p["unload_share_target"] * 100 for p in tex_pairs]
        ax.plot(tex_t, tex_h, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("h_min (μm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hmin_vs_unload_share.png"), dpi=150)
    plt.close(fig)

    # 3. p_max
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, pmax_s, "bo-", lw=2, markersize=7, label="Smooth")
    if tex_available:
        tex_p = [p["textured"]["p_max"] / 1e6 for p in tex_pairs]
        tex_t = [p["unload_share_target"] * 100 for p in tex_pairs]
        ax.plot(tex_t, tex_p, "rs-", lw=2, markersize=7, label="Textured")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("p_max (MPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pmax_vs_unload_share.png"), dpi=150)
    plt.close(fig)

    # 4. Force sharing
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, hydro_s, "bo-", lw=2, markersize=7, label="hydro_share")
    ax.plot(targets, unload_s, "rs-", lw=2, markersize=7, label="unload_share")
    ax.plot(targets, hydro_s + unload_s, "g^--", lw=1.5, markersize=5,
            alpha=0.7, label="sum")
    ax.plot(targets, targets / 100, "k:", lw=1, label="target")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("load share (on ê_resist)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "force_sharing.png"), dpi=150)
    plt.close(fig)

    # 5. textured/smooth ratios — строим ТОЛЬКО если есть accepted pairs
    # (ТЗ §4.4.5: no ratios plot when 0 accepted).
    if tex_available:
        t_arr = [p["unload_share_target"] * 100 for p in tex_pairs]
        hr = [p["ratios"]["h_ratio"] for p in tex_pairs]
        pr = [p["ratios"]["p_ratio"] for p in tex_pairs]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(t_arr, hr, "bo-", lw=2, markersize=7)
        axes[0].axhline(1.0, color="gray", ls=":")
        axes[0].set_xlabel("unload_share_target (%)")
        axes[0].set_ylabel("h_min_tex / h_min_smooth")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t_arr, pr, "rs-", lw=2, markersize=7)
        axes[1].axhline(1.0, color="gray", ls=":")
        axes[1].set_xlabel("unload_share_target (%)")
        axes[1].set_ylabel("p_max_tex / p_max_smooth")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ratios_tex_vs_smooth.png"),
                     dpi=150)
        plt.close(fig)
    else:
        print("[plot] 0 accepted textured pairs — ratios plot пропущен")

    # MD report
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        cfg = sm["config"]
        f.write("# Pump bearing + magnetic unloading (radial surrogate)\n\n")
        f.write(f"- schema: `{sm.get('schema_version')}`\n")
        f.write(f"- run_id: `{sm.get('run_id')}`\n")
        f.write(f"- Model: {sm.get('model', 'radial')}\n")
        f.write(f"- Grid: {cfg['N_phi']}×{cfg['N_Z']}\n")
        f.write(f"- F₀ = {cfg['F0_N']:.1f} Н\n")
        f.write(f"- W_applied = ({cfg['W_applied_N'][0]:.1f}, "
                f"{cfg['W_applied_N'][1]:.1f}) Н\n")
        f.write(f"- tol_accept = {cfg['tol_accept']:.1e}, "
                f"step_cap = {cfg['step_cap']:.2f}, "
                f"eps_max = {cfg['eps_max']:.2f}\n\n")

        f.write("## Baseline raw (initial NR seed)\n\n")
        br = sm["baseline_raw"]
        f.write(f"- ε = {br['eps']:.4f}, X = {br['X']:.4f}, "
                f"Y = {br['Y']:.4f}\n")
        f.write(f"- h_min = {br['h_min']*1e6:.2f} μm, "
                f"p_max = {br['p_max']/1e6:.2f} MPa\n")
        f.write(f"- residual = {br['rel_residual']:.2e}, "
                f"status = `{br.get('status', 'n/a')}`\n\n")

        f.write("## Baseline canonical (accepted target=0.0)\n\n")
        bc = sm["baseline_canonical"]
        f.write(f"- ε = {bc['eps']:.4f}, X = {bc['X']:.4f}, "
                f"Y = {bc['Y']:.4f}\n")
        f.write(f"- h_min = {bc['h_min']*1e6:.2f} μm, "
                f"p_max = {bc['p_max']/1e6:.2f} MPa\n")
        f.write(f"- residual = {bc['rel_residual']:.2e}, "
                f"status = `{bc.get('status', 'n/a')}`\n\n")

        f.write("## Acceptance\n\n")
        acc = sm["acceptance"]
        for key in ["baseline_reproduced", "unload_positive",
                    "eps_monotonic", "sum_shares_ok"]:
            v = acc.get(key)
            f.write(f"- {key}: **{'PASS' if v else 'FAIL'}**\n")
        f.write(f"- max residual = {acc['max_residual']:.2e}\n")
        f.write(f"- accepted targets = {acc.get('accepted_targets', [])}\n\n")

        f.write("## Smooth continuation (accepted targets)\n\n")
        f.write("| target | ε | h_min, μm | p_max, MPa "
                "| hydro_share | unload_share |\n")
        f.write("|--------|---|-----------|-----------"
                "|-------------|--------------|\n")
        for r in smooth_runs:
            f.write(f"| {r['unload_share_target']*100:.2f}% "
                    f"| {r['eps']:.4f} "
                    f"| {r['h_min']*1e6:.2f} "
                    f"| {r['p_max']/1e6:.2f} "
                    f"| {r['hydro_share_actual']:+.4f} "
                    f"| {r['unload_share_actual']:+.4f} |\n")
        f.write("\n")

        if tx is not None:
            if tex_available:
                f.write("## Textured comparison (accepted only)\n\n")
                f.write("| target | smooth ε | tex ε | Δε "
                        "| h_t/h_s | p_t/p_s | Δcav |\n")
                f.write("|--------|----------|-------|-----"
                        "|---------|---------|------|\n")
                for p in tex_pairs:
                    rs = p["smooth_ref"]
                    rt = p["textured"]
                    rr = p["ratios"]
                    f.write(f"| {p['unload_share_target']*100:.2f}% "
                            f"| {rs['eps']:.4f} | {rt['eps']:.4f} "
                            f"| {rr['delta_eps']:+.4f} "
                            f"| {rr['h_ratio']:.4f} "
                            f"| {rr['p_ratio']:.4f} "
                            f"| {rr['delta_cav']:+.4f} |\n")
                f.write("\n")
            else:
                f.write("## Textured comparison\n\n")
                f.write("_No accepted textured pairs — "
                        "ratios not plotted._\n\n")

        f.write("## Figures\n\n")
        for name in ["eps_vs_unload_share", "hmin_vs_unload_share",
                     "pmax_vs_unload_share", "force_sharing"]:
            f.write(f"![{name}]({name}.png)\n\n")
        if tex_available:
            f.write("![ratios_tex_vs_smooth](ratios_tex_vs_smooth.png)\n")

    print(f"→ {out_dir}/report.md + figures")


if __name__ == "__main__":
    main()
