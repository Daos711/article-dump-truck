#!/usr/bin/env python3
"""Aggregate multiple magnetic_v4 runs into overlay plots + summary.

Читает `manifest.json` (required) и `textured_compare.json` (optional)
под каждым run_id, строит overlay-графики и сводный summary.md/json.

Вход:
  --run-ids r1,r2,r3       явный список
  --pattern '2026-04-16*'  glob-pattern по именам run-каталогов
  --base <path>            корневая директория (default: results/magnetic_pump)

Выход (под `<base>/_aggregate/<timestamp>/`):
  overlay_eps_vs_unload.png    ε_eq vs unload_share_target
  overlay_hratio_vs_eps.png    h_t/h_s vs ε_eq_smooth  (главный график)
  overlay_dEps_vs_eps.png      Δε vs ε_eq_smooth
  overlay_fratio_vs_eps.png    f_t/f_s vs ε_eq_smooth
  summary.md, summary.json

Никакой интерпретации в коде. Только метрики + констатация факта
«есть ли хоть одна точка с h_t/h_s > 1.0».
"""
import argparse
import datetime
import fnmatch
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEMA_VERSION = "magnetic_v4"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_schema(doc, path):
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema_version={doc.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}")


def discover_runs(base, run_ids=None, pattern=None):
    """Вернуть список run_id, которые содержат manifest.json + magnetic_v4."""
    if not os.path.isdir(base):
        return []
    if run_ids:
        candidates = [r.strip() for r in run_ids if r.strip()]
    else:
        candidates = sorted(
            name for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name))
            and not name.startswith("_")
        )
        if pattern:
            candidates = [c for c in candidates if fnmatch.fnmatch(c, pattern)]
    valid = []
    for rid in candidates:
        mp = os.path.join(base, rid, "manifest.json")
        if not os.path.exists(mp):
            print(f"  skip {rid}: no manifest.json")
            continue
        try:
            m = load_json(mp)
            assert_schema(m, mp)
        except Exception as e:
            print(f"  skip {rid}: {e}")
            continue
        valid.append(rid)
    return valid


def wy_share_from_manifest(m):
    """W_y_share = |W_applied.y| / F0."""
    F0 = float(m["config"]["F0_N"])
    Wy = float(m["config"]["W_applied_N"][1])
    if F0 <= 0:
        return None
    return abs(Wy) / F0


def load_run(base, rid):
    """Вернуть dict с aggregated данными по одному run."""
    run_dir = os.path.join(base, rid)
    manifest = load_json(os.path.join(run_dir, "manifest.json"))
    assert_schema(manifest, rid + "/manifest.json")

    tx_path = os.path.join(run_dir, "textured_compare.json")
    tx = load_json(tx_path)
    if tx is not None:
        assert_schema(tx, rid + "/textured_compare.json")

    smooth_accepted = manifest.get("smooth_accepted", [])
    pairs = (tx.get("pairs", []) if tx is not None else [])

    wy = wy_share_from_manifest(manifest)
    baseline_eps = (manifest.get("baseline_canonical") or {}).get("eps")

    # Извлечь точки для overlay
    smooth_targets = [s["unload_share_target"] for s in smooth_accepted]
    smooth_eps = [s["eps"] for s in smooth_accepted]

    tex_targets = []
    tex_eps = []
    tex_eps_smooth_ref = []
    tex_h_ratio = []
    tex_p_ratio = []
    tex_f_ratio = []
    tex_delta_eps = []
    for p in pairs:
        if not p.get("accepted") or not p.get("textured"):
            continue
        tex_targets.append(p["unload_share_target"])
        tex_eps.append(p["textured"]["eps"])
        tex_eps_smooth_ref.append(p["smooth_ref"]["eps"])
        rr = p.get("ratios", {})
        tex_h_ratio.append(rr.get("h_ratio"))
        tex_p_ratio.append(rr.get("p_ratio"))
        tex_f_ratio.append(rr.get("f_ratio"))
        tex_delta_eps.append(rr.get("delta_eps"))

    # Best h-ratio point (среди accepted textured)
    best_h_ratio = None
    best_eps_at_best_h_ratio = None
    if tex_h_ratio:
        idx = int(np.argmax(tex_h_ratio))
        best_h_ratio = float(tex_h_ratio[idx])
        best_eps_at_best_h_ratio = float(tex_eps_smooth_ref[idx])

    min_eps_reached = (min(smooth_eps) if smooth_eps else None)
    max_unload_actual = max(
        (s.get("unload_share_actual", 0.0) for s in smooth_accepted),
        default=0.0)

    return dict(
        run_id=rid,
        W_y_share=wy,
        baseline_eps=baseline_eps,
        min_eps_reached=min_eps_reached,
        max_unload_actual=float(max_unload_actual),
        n_accepted_smooth=len(smooth_accepted),
        n_accepted_tex=len(tex_targets),
        best_h_ratio=best_h_ratio,
        best_eps_at_best_h_ratio=best_eps_at_best_h_ratio,
        smooth_targets=smooth_targets,
        smooth_eps=smooth_eps,
        tex_targets=tex_targets,
        tex_eps=tex_eps,
        tex_eps_smooth_ref=tex_eps_smooth_ref,
        tex_h_ratio=tex_h_ratio,
        tex_p_ratio=tex_p_ratio,
        tex_f_ratio=tex_f_ratio,
        tex_delta_eps=tex_delta_eps,
    )


def color_for(wy, wy_list):
    """Цвет по W_y_share — интерполяция через viridis."""
    cmap = plt.get_cmap("viridis")
    if not wy_list:
        return cmap(0.5)
    lo, hi = min(wy_list), max(wy_list)
    if hi - lo < 1e-9:
        return cmap(0.5)
    return cmap((wy - lo) / (hi - lo))


def label_for(run):
    wy = run["W_y_share"]
    if wy is None:
        return run["run_id"]
    return f"wy={wy:.2f}"


def plot_eps_vs_unload(runs, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    wy_list = [r["W_y_share"] for r in runs if r["W_y_share"] is not None]
    for run in runs:
        c = color_for(run["W_y_share"] or 0.0, wy_list)
        lbl = label_for(run)
        if run["smooth_targets"]:
            t = np.array(run["smooth_targets"]) * 100
            ax.plot(t, run["smooth_eps"], "o-", color=c, lw=2,
                    markersize=6, label=f"{lbl} smooth")
        if run["tex_targets"]:
            t = np.array(run["tex_targets"]) * 100
            ax.plot(t, run["tex_eps"], "s--", color=c, lw=1.5,
                    markersize=6, alpha=0.9, label=f"{lbl} tex")
    ax.set_xlabel("unload_share_target (%)")
    ax.set_ylabel("ε_eq")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_vs_eps(runs, key, ylabel, axhline, out_path):
    """Generic scatter: [ε_eq_smooth, key] c цветом по W_y_share."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    wy_list = [r["W_y_share"] for r in runs if r["W_y_share"] is not None]
    any_data = False
    for run in runs:
        if not run["tex_eps_smooth_ref"]:
            continue
        any_data = True
        c = color_for(run["W_y_share"] or 0.0, wy_list)
        xs = run["tex_eps_smooth_ref"]
        ys = run[key]
        # Filter None
        xs_f, ys_f = [], []
        for x, y in zip(xs, ys):
            if x is None or y is None:
                continue
            xs_f.append(x)
            ys_f.append(y)
        if not xs_f:
            continue
        ax.plot(xs_f, ys_f, "o-", color=c, lw=1.5, markersize=7,
                label=label_for(run))
    if axhline is not None:
        ax.axhline(axhline, color="gray", ls=":", lw=1)
    ax.set_xlabel("ε_eq (smooth)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no accepted textured pairs",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=None,
                        help="magnetic_pump root (default: "
                             "results/magnetic_pump)")
    parser.add_argument("--run-ids", type=str, default=None,
                        help="comma-separated run_ids")
    parser.add_argument("--pattern", type=str, default=None,
                        help="glob pattern over run dirnames, e.g. "
                             "'2026-04-16*radial*'")
    parser.add_argument("--out", type=str, default=None,
                        help="explicit output dir (default: "
                             "<base>/_aggregate/<timestamp>)")
    args = parser.parse_args()

    base = args.base or os.path.join(
        os.path.dirname(__file__), "..", "results", "magnetic_pump")
    base = os.path.abspath(base)

    run_ids = None
    if args.run_ids:
        run_ids = [x for x in args.run_ids.split(",") if x.strip()]

    print(f"Base: {base}")
    ids = discover_runs(base, run_ids=run_ids, pattern=args.pattern)
    if not ids:
        print("FAIL: не найдено ни одного magnetic_v4 run")
        sys.exit(1)
    print(f"Found {len(ids)} runs: {ids}")

    runs = [load_run(base, rid) for rid in ids]

    # Output dir
    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = os.path.join(base, "_aggregate", stamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Aggregate out: {out_dir}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_eps_vs_unload(runs,
                       os.path.join(out_dir, "overlay_eps_vs_unload.png"))
    plot_scatter_vs_eps(runs, "tex_h_ratio",
                         "h_min_tex / h_min_smooth",
                         axhline=1.0,
                         out_path=os.path.join(
                             out_dir, "overlay_hratio_vs_eps.png"))
    plot_scatter_vs_eps(runs, "tex_delta_eps",
                         "Δε (tex − smooth)",
                         axhline=0.0,
                         out_path=os.path.join(
                             out_dir, "overlay_dEps_vs_eps.png"))
    plot_scatter_vs_eps(runs, "tex_f_ratio",
                         "friction_tex / friction_smooth",
                         axhline=1.0,
                         out_path=os.path.join(
                             out_dir, "overlay_fratio_vs_eps.png"))

    # ── summary.json ──────────────────────────────────────────────
    def serialize_run(r):
        return dict(
            run_id=r["run_id"],
            W_y_share=r["W_y_share"],
            baseline_eps=r["baseline_eps"],
            min_eps_reached=r["min_eps_reached"],
            max_unload_actual=r["max_unload_actual"],
            n_accepted_smooth=r["n_accepted_smooth"],
            n_accepted_tex=r["n_accepted_tex"],
            best_h_ratio=r["best_h_ratio"],
            best_eps_at_best_h_ratio=r["best_eps_at_best_h_ratio"],
        )

    # Констатация факта h_t/h_s > 1.0
    gain_points = []  # (run_id, target, eps_smooth, h_ratio)
    for r in runs:
        for i, hr in enumerate(r["tex_h_ratio"]):
            if hr is None:
                continue
            if hr > 1.0:
                gain_points.append(dict(
                    run_id=r["run_id"],
                    W_y_share=r["W_y_share"],
                    unload_share_target=r["tex_targets"][i],
                    eps_smooth=r["tex_eps_smooth_ref"][i],
                    h_ratio=float(hr),
                    delta_eps=(r["tex_delta_eps"][i]
                               if r["tex_delta_eps"][i] is not None else None),
                    f_ratio=(r["tex_f_ratio"][i]
                             if r["tex_f_ratio"][i] is not None else None),
                ))

    summary = dict(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        base=base,
        runs=[serialize_run(r) for r in runs],
        has_texture_gain=bool(gain_points),
        gain_points=gain_points,
    )
    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary.json: {out_dir}/summary.json")

    # ── summary.md ────────────────────────────────────────────────
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Magnetic W-load sweep summary\n\n")
        f.write(f"- schema: `{SCHEMA_VERSION}`\n")
        f.write(f"- aggregated at: {summary['created_utc']}\n")
        f.write(f"- base: `{base}`\n")
        f.write(f"- runs: {len(runs)}\n\n")

        f.write("## Per-run summary\n\n")
        f.write("| run_id | W_y_share | baseline_eps | min_eps_reached "
                "| max_unload_actual | n_acc_smooth | n_acc_tex "
                "| best_h_ratio | ε @ best h_ratio |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in runs:
            wy = "—" if r["W_y_share"] is None else f"{r['W_y_share']:.3f}"
            be = ("—" if r["baseline_eps"] is None
                  else f"{r['baseline_eps']:.4f}")
            me = ("—" if r["min_eps_reached"] is None
                  else f"{r['min_eps_reached']:.4f}")
            bh = ("—" if r["best_h_ratio"] is None
                  else f"{r['best_h_ratio']:.4f}")
            be_h = ("—" if r["best_eps_at_best_h_ratio"] is None
                    else f"{r['best_eps_at_best_h_ratio']:.4f}")
            f.write(f"| `{r['run_id']}` | {wy} | {be} | {me} "
                    f"| {r['max_unload_actual']:.4f} "
                    f"| {r['n_accepted_smooth']} | {r['n_accepted_tex']} "
                    f"| {bh} | {be_h} |\n")
        f.write("\n")

        eps_values = [r["min_eps_reached"] for r in runs
                      if r["min_eps_reached"] is not None]
        min_eps_all = min(eps_values) if eps_values else float("nan")

        f.write("## Texture gain (h_t/h_s > 1.0)\n\n")
        if not gain_points:
            f.write("**Ни одной точки с `h_t/h_s > 1.0` не найдено.**\n\n")
            f.write(f"Минимальный достигнутый ε среди всех runs: "
                    f"{min_eps_all:.4f}.\n\n")
        else:
            f.write(f"Найдено **{len(gain_points)}** точек с "
                    f"`h_t/h_s > 1.0`. Минимальный достигнутый ε "
                    f"по всем runs: {min_eps_all:.4f}.\n\n")
            f.write("| run_id | W_y_share | target | ε_smooth "
                    "| h_t/h_s | Δε | f_t/f_s |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for g in gain_points:
                wy = "—" if g["W_y_share"] is None else f"{g['W_y_share']:.3f}"
                de = ("—" if g["delta_eps"] is None
                      else f"{g['delta_eps']:+.4f}")
                fr = ("—" if g["f_ratio"] is None
                      else f"{g['f_ratio']:.4f}")
                f.write(f"| `{g['run_id']}` | {wy} "
                        f"| {g['unload_share_target']*100:.2f}% "
                        f"| {g['eps_smooth']:.4f} "
                        f"| {g['h_ratio']:.4f} | {de} | {fr} |\n")
            f.write("\n")

        f.write("## Figures\n\n")
        for name in ["overlay_eps_vs_unload",
                     "overlay_hratio_vs_eps",
                     "overlay_dEps_vs_eps",
                     "overlay_fratio_vs_eps"]:
            f.write(f"![{name}]({name}.png)\n\n")

    print(f"summary.md: {md_path}")
    print(f"\nRuns aggregated: {len(runs)}")
    print(f"Texture gain points (h_t/h_s > 1.0): {len(gain_points)}")


if __name__ == "__main__":
    main()
