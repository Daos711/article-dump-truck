#!/usr/bin/env python3
"""Plot + report для coexp_v1 (ТЗ §10.4).

Schema-strict (T7): любой phase manifest без `schema_version=coexp_v1`
→ FAIL. Никаких legacy-fallbacks.

Читает в одном run_id:
  results/coexp/<run_id>/screening/manifest.json + jsonl
  results/coexp/<run_id>/confirm/manifest.json (optional)
  results/coexp/<run_id>/equilibrium/equilibrium_summary.json (optional)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.coexp_schema import SCHEMA_VERSION


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_schema(doc, path):
    if doc is None:
        return
    if doc.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: {path} schema_version="
              f"{doc.get('schema_version')!r}, expected "
              f"{SCHEMA_VERSION!r}")
        sys.exit(1)


def load_jsonl(path) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="results/coexp/<run_id> directory")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(run_dir):
        print(f"FAIL: нет {run_dir}")
        sys.exit(1)

    screening_dir = os.path.join(run_dir, "screening")
    confirm_dir = os.path.join(run_dir, "confirm")
    eq_dir = os.path.join(run_dir, "equilibrium")

    sm = load_json(os.path.join(screening_dir, "manifest.json"))
    if sm is None:
        print(f"FAIL: нет screening manifest")
        sys.exit(1)
    assert_schema(sm, "screening/manifest.json")

    cf = load_json(os.path.join(confirm_dir, "manifest.json"))
    assert_schema(cf, "confirm/manifest.json")

    eq = load_json(os.path.join(eq_dir, "equilibrium_summary.json"))
    assert_schema(eq, "equilibrium/equilibrium_summary.json")

    out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    screening_records = load_jsonl(
        os.path.join(screening_dir, "screening_results.jsonl"))
    rejects = load_jsonl(
        os.path.join(screening_dir, "screening_rejects.jsonl"))

    # ── Figure 1: J_screen distribution per family ─────────────────
    by_family: Dict[str, List[float]] = {}
    for r in screening_records:
        if r.get("screen_fail"):
            continue
        j = r.get("J_screen")
        if j is None or (isinstance(j, float) and np.isnan(j)):
            continue
        by_family.setdefault(r["family"], []).append(float(j))
    if by_family:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = sorted(by_family.keys())
        ax.boxplot([by_family[k] for k in labels], labels=labels)
        ax.axhline(0.0, color="gray", ls=":")
        ax.set_ylabel("J_screen")
        ax.set_title(f"J_screen per family (run {sm['run_id']})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "screening_J_per_family.png"),
                     dpi=150)
        plt.close(fig)

    # ── Figure 2: top12 J_screen vs profile (bar) ──────────────────
    top12_doc = load_json(
        os.path.join(screening_dir, "top12_candidates.json"))
    assert_schema(top12_doc, "screening/top12_candidates.json")
    if top12_doc is not None and top12_doc["candidates"]:
        candidates = top12_doc["candidates"]
        labels = [f"{c['family'][:4]}_{c['profile_id'][:6]}"
                  for c in candidates]
        Js = [float(c["J_screen"]) for c in candidates]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(Js)), Js)
        ax.set_xticks(range(len(Js)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("J_screen")
        ax.set_title(f"Top-12 candidates (run {sm['run_id']})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top12_J_screen.png"), dpi=150)
        plt.close(fig)

    # ── Figure 3: confirm coarse vs fine J ─────────────────────────
    if cf is not None:
        diffs = cf.get("coarse_fine_diff", []) or []
        if diffs:
            xs = list(range(len(diffs)))
            j_c = [d.get("J_coarse", float("nan")) for d in diffs]
            j_f = [d.get("J_fine", float("nan")) for d in diffs]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(xs, j_c, "o-", label="J_coarse")
            ax.plot(xs, j_f, "s-", label="J_fine")
            ax.set_xticks(xs)
            ax.set_xticklabels([d["profile_id"][:6] for d in diffs],
                                rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("J_screen")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Confirm: coarse vs fine grid")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "confirm_coarse_vs_fine.png"),
                         dpi=150)
            plt.close(fig)

    # ── Figure 4: equilibrium ratios ───────────────────────────────
    if eq is not None and eq.get("pairs"):
        pairs = eq["pairs"]
        labels = [p["profile_hash"][:8] for p in pairs]
        h_r = [p["ratios"].get("h_r", float("nan")) if p.get("ratios")
               else float("nan") for p in pairs]
        p_r = [p["ratios"].get("p_r", float("nan")) if p.get("ratios")
               else float("nan") for p in pairs]
        f_r = [p["ratios"].get("f_r", float("nan")) if p.get("ratios")
               else float("nan") for p in pairs]
        useful = [bool(p.get("useful")) for p in pairs]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(labels))
        w = 0.27
        bars1 = ax.bar(x - w, h_r, w, label="h_r", color="tab:blue")
        bars2 = ax.bar(x,     p_r, w, label="p_r", color="tab:orange")
        bars3 = ax.bar(x + w, f_r, w, label="f_r", color="tab:green")
        for i, u in enumerate(useful):
            if u:
                ax.text(i, max(h_r[i], p_r[i], f_r[i]) * 1.02,
                         "USEFUL", ha="center", fontsize=8,
                         color="darkred")
        ax.axhline(1.0, color="gray", ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("ratio (textured / smooth)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(f"Equilibrium ratios (Wy_share={eq['Wy_share']})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "equilibrium_ratios.png"),
                     dpi=150)
        plt.close(fig)

    # ── Report markdown ────────────────────────────────────────────
    md = []
    md.append(f"# Co-design report — run `{sm['run_id']}`\n\n")
    md.append(f"- schema: `{SCHEMA_VERSION}`\n")
    md.append(f"- created (screening): {sm.get('created_utc')}\n")
    md.append(f"- git sha: `{sm.get('git_sha') or 'n/a'}`\n")
    md.append(f"- oil: {sm.get('oil')}, pv: {sm.get('pv')}\n")
    md.append(f"- families: {sm.get('families')}\n")
    md.append(f"- screening grid: {sm['grid']}\n")
    md.append(f"- screening eps: {sm.get('screening_eps')}\n")
    pr = sm.get("pass_rate", {})
    md.append(f"- pass rate: {pr.get('n_pass')}/{pr.get('n_total')} "
              f"→ **{pr.get('status')}**\n\n")

    md.append("## Cylindrical reference (per ε)\n\n")
    md.append("| ε | h_min, μm | p_max, MPa | friction |\n")
    md.append("|---|---|---|---|\n")
    for k, v in sm.get("cylindrical_reference", {}).items():
        md.append(f"| {k} | {v['h_min']*1e6:.2f} | "
                  f"{v['p_max']/1e6:.2f} | {v['friction']:.2f} |\n")
    md.append("\n")

    if top12_doc is not None:
        md.append("## Top-12 (screening)\n\n")
        md.append("| # | family | hash | J_screen | params |\n")
        md.append("|---|---|---|---|---|\n")
        for i, c in enumerate(top12_doc["candidates"]):
            md.append(f"| {i+1} | {c['family']} | "
                      f"`{c['profile_id'][:8]}` | "
                      f"{c['J_screen']:.4f} | "
                      f"{c['profile_spec']['params']} |\n")
        md.append("\n")

    if cf is not None:
        md.append(f"## Confirm (run `{cf['run_id']}`, "
                  f"grid {cf['grid']})\n\n")
        md.append(f"- candidates: {cf['n_candidates']}, "
                  f"confirmed: {cf['n_confirmed']}\n")
        diffs = cf.get("coarse_fine_diff", []) or []
        big = [d for d in diffs
                if (d.get("mean_abs_dh_r") or 0.0) > 0.02
                or (d.get("mean_abs_dp_r") or 0.0) > 0.02]
        md.append(f"- large coarse-vs-fine diffs (>2%): {len(big)}\n\n")

    if eq is not None:
        md.append(f"## Equilibrium ({eq['phase']}, "
                  f"grid {eq['grid']}, Wy_share={eq['Wy_share']})\n\n")
        md.append(f"- NR seed: X0={eq['nr_seed']['X0']}, "
                  f"Y0={eq['nr_seed']['Y0']}\n")
        md.append(f"- pairs: {eq['n_pairs']}, useful: "
                  f"**{eq['n_useful']}**\n\n")
        md.append("| hash | smooth_acc | tex_acc | h_r | p_r | f_r | "
                  "c_d | useful |\n")
        md.append("|---|---|---|---|---|---|---|---|\n")
        for p in eq["pairs"]:
            r = p.get("ratios") or {}
            md.append(
                f"| `{p['profile_hash'][:8]}` "
                f"| {'✓' if p['smooth_accepted'] else '✗'} "
                f"| {'✓' if p['textured_accepted'] else '✗'} "
                f"| {r.get('h_r', float('nan')):.4f} "
                f"| {r.get('p_r', float('nan')):.4f} "
                f"| {r.get('f_r', float('nan')):.4f} "
                f"| {r.get('c_d', float('nan')):+.4f} "
                f"| {'YES' if p.get('useful') else '—'} |\n")
        md.append("\n")
        if eq.get("any_useful"):
            md.append("**Result: at least one candidate is physically "
                      "useful.** See equilibrium_results.csv for full "
                      "metrics.\n\n")
        else:
            md.append("**Result: NO candidate satisfies all four "
                      "useful gates** (h_r>1.005, p_r≤1.000, f_r≤1.02, "
                      "c_d≤0.02). Infrastructure PASS, physics NEGATIVE "
                      "(ТЗ §15 scenario C).\n\n")

    md.append("## Figures\n\n")
    for name in ("screening_J_per_family", "top12_J_screen",
                 "confirm_coarse_vs_fine", "equilibrium_ratios"):
        path = os.path.join(out_dir, name + ".png")
        if os.path.exists(path):
            md.append(f"![{name}]({name}.png)\n\n")

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"report.md: {md_path}")


if __name__ == "__main__":
    main()
