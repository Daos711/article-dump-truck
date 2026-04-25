#!/usr/bin/env python3
"""Textured vs smooth сравнение с магнитной разгрузкой.

Работает ИСКЛЮЧИТЕЛЬНО с accepted smooth-точками из manifest.json
(schema magnetic_v4). Smooth solve повторно НЕ запускается — это было
главной ошибкой прошлой версии (разные smooth states при сравнении).

Pipeline (см. ТЗ §3.3, §3.6):
    load manifest.json
    accepted = manifest["smooth_accepted"]
    for sref in accepted:
        K_mag := sref["K_mag"]
        X0_tex, Y0_tex := sref["X"], sref["Y"]
        rt := find_equilibrium(textured, ... tol_accept from manifest)
        store pair = {smooth_ref: sref (byte-identical copy),
                      textured: rt,
                      accepted: is_accepted(rt, tol_accept)}
"""
import sys
import os
import csv
import json
import time
import argparse
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

from models.magnetic_force import (
    RadialUnloadForceModel, sanity_checks,
)
from models.magnetic_equilibrium import (
    find_equilibrium, is_accepted, result_to_dict,
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Config ──────────────────────────────────────────────────────
SCHEMA_VERSION = "magnetic_v4"
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * params.n / 60.0
P_SCALE = 6 * ETA * OMEGA * (params.R / params.c) ** 2
F0 = P_SCALE * params.R * params.L

# Reference texture
TEX_PHI_START = 0.0
TEX_PHI_END = 90.0
TEX_A_MM = 1.5
TEX_B_MM = 1.2
TEX_HP_UM = 30
TEX_SF = 1.5
TEX_PROFILE = "smoothcap"


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_smooth_H0(X, Y, Phi):
    H0 = 1.0 + X * np.cos(Phi) + Y * np.sin(Phi)
    return np.sqrt(H0**2 + (params.sigma / params.c) ** 2)


def setup_texture(Phi, Zm):
    a_phi = (TEX_B_MM * 1e-3) / params.R
    a_Z = 2 * (TEX_A_MM * 1e-3) / params.L
    depth = (TEX_HP_UM * 1e-6) / params.c

    phi_s = np.deg2rad(TEX_PHI_START)
    phi_e = np.deg2rad(TEX_PHI_END)
    phi_span = phi_e - phi_s
    N_phi_tex = max(1, int(phi_span / (TEX_SF * 2 * a_phi)))
    N_Z_tex = max(1, int(2.0 / (TEX_SF * 2 * a_Z)))

    margin = a_phi * 1.1
    usable = phi_span - 2 * margin
    if N_phi_tex == 1:
        phi_c = np.array([phi_s + phi_span / 2])
    else:
        phi_c = phi_s + margin + np.linspace(0, usable, N_phi_tex)

    margin_Z = a_Z * 1.1
    usable_Z = 2.0 - 2 * margin_Z
    if N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_c = -1.0 + margin_Z + np.linspace(0, usable_Z, N_Z_tex)

    pg, zg = np.meshgrid(phi_c, Z_c)
    return pg.ravel(), zg.ravel(), a_phi, a_Z, depth


def make_H_and_force_textured(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z, tex):
    """Замыкание: build_H(X,Y) + texture + PS solve + metrics.

    Namespaced "textured only" — чтобы compare точно не умел решать
    smooth случай. Это инвариант (см. ТЗ §3.3).
    """

    def H_and_force(X, Y):
        H0 = make_smooth_H0(X, Y, Phi)
        H = create_H_with_ellipsoidal_depressions(
            H0, tex["depth"], Phi, Zm,
            tex["phi_c"], tex["Z_c"],
            tex["a_Z"], tex["a_phi"], profile=TEX_PROFILE)
        P, theta, _, _ = _ps_solver(
            H, d_phi, d_Z, params.R, params.L,
            tol=1e-6, max_iter=10_000_000)
        P_dim = P * P_SCALE
        Fx = -np.trapezoid(
            np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * params.R * params.L / 2
        Fy = -np.trapezoid(
            np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * params.R * params.L / 2
        h_dim = H * params.c
        h_min = float(np.min(h_dim))
        p_max = float(np.max(P_dim))
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
        tau_c = ETA * OMEGA * params.R / h_dim
        friction = float(
            np.sum(tau_c) * params.R * (2 * np.pi / H.shape[1])
            * params.L * (2 / H.shape[0]) / 2)
        return (float(Fx), float(Fy), h_min, p_max, cav_frac,
                friction, P, theta)

    return H_and_force


def resolve_manifest(args):
    """Локализация manifest.json: explicit --manifest > --run-id > latest."""
    base = os.path.join(os.path.dirname(__file__), "..",
                        "results", "magnetic_pump")
    if args.manifest:
        return os.path.abspath(args.manifest)
    if args.run_id:
        return os.path.join(base, args.run_id, "manifest.json")
    latest = os.path.join(base, "latest_run.txt")
    if os.path.exists(latest):
        with open(latest, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        return os.path.join(base, run_id, "manifest.json")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None,
                        help="path to manifest.json")
    parser.add_argument("--run-id", type=str, default=None,
                        help="run_id under results/magnetic_pump/")
    args = parser.parse_args()

    manifest_path = resolve_manifest(args)
    if manifest_path is None or not os.path.exists(manifest_path):
        print("Нет manifest.json. Запусти run_mag_smooth_continuation.py "
              "или передай --manifest / --run-id.")
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Schema check (ТЗ §3.4): несовпадение — FAIL, а не silent downgrade.
    if manifest.get("schema_version") != SCHEMA_VERSION:
        print(f"FAIL: manifest schema_version="
              f"{manifest.get('schema_version')!r}, "
              f"expected {SCHEMA_VERSION!r}. Перегенерируй smooth run.")
        sys.exit(1)

    run_dir = os.path.dirname(manifest_path)
    run_id = manifest["run_id"]
    cfg = manifest["config"]
    tol_accept = float(cfg["tol_accept"])
    step_cap = float(cfg["step_cap"])
    eps_max = float(cfg["eps_max"])
    N_phi = int(cfg["N_phi"])
    N_Z = int(cfg["N_Z"])
    W_applied = np.array(cfg["W_applied_N"], dtype=float)

    accepted_smooth = manifest["smooth_accepted"]
    if not accepted_smooth:
        print("В manifest нет smooth_accepted точек")
        sys.exit(1)

    print(f"Manifest: {manifest_path}")
    print(f"Run ID: {run_id}")
    print(f"tol_accept={tol_accept}, step_cap={step_cap}, eps_max={eps_max}")

    ok, _ = sanity_checks(verbose=True)
    if not ok:
        print("FAIL sanity")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("TEXTURED vs SMOOTH + MAGNETIC UNLOAD (accepted smooth only)")
    acc_str = ", ".join(f"{s['unload_share_target']*100:.1f}%"
                         for s in accepted_smooth)
    print(f"Accepted targets: [{acc_str}]")
    print("=" * 72)

    phi, Z, Phi, Zm, dp, dz = make_grid(N_phi, N_Z)
    phi_c, Z_c, a_phi, a_Z, depth = setup_texture(Phi, Zm)
    n_dimples = int(len(phi_c))
    tex = dict(phi_c=phi_c, Z_c=Z_c, a_phi=a_phi, a_Z=a_Z, depth=depth)
    print(f"Texture: {n_dimples} лунок, zone "
          f"{TEX_PHI_START}-{TEX_PHI_END}°")

    H_and_force_tex = make_H_and_force_textured(
        Phi, Zm, phi, Z, dp, dz, tex)

    # Одна textured-template модель, которую мы клонируем с K_mag из smooth.
    mag_template = RadialUnloadForceModel(
        n_mag=3, H_reg=0.05, H_floor=0.02)

    # Fallback seed: предыдущий accepted textured (см. ТЗ §3.6 optional).
    prev_tex_X, prev_tex_Y = None, None

    pairs = []
    for sref in accepted_smooth:
        target = float(sref["unload_share_target"])
        K_mag = float(sref["K_mag"])
        X0_tex = float(sref["X"])
        Y0_tex = float(sref["Y"])

        m_tex = copy.copy(mag_template)
        m_tex.scale = K_mag

        # First try: seed = smooth accepted point (ТЗ §3.6 required).
        t0 = time.time()
        rt = find_equilibrium(
            H_and_force_tex, m_tex, W_applied,
            X0=X0_tex, Y0=Y0_tex,
            tol=tol_accept, step_cap=step_cap, eps_max=eps_max,
            tol_accept=tol_accept)
        rt.unload_share_target = target

        # Optional fallback: previous accepted textured seed.
        if not is_accepted(rt, tol_accept) and prev_tex_X is not None:
            m_tex2 = copy.copy(mag_template)
            m_tex2.scale = K_mag
            rt_fb = find_equilibrium(
                H_and_force_tex, m_tex2, W_applied,
                X0=prev_tex_X, Y0=prev_tex_Y,
                tol=tol_accept, step_cap=step_cap, eps_max=eps_max,
                tol_accept=tol_accept)
            rt_fb.unload_share_target = target
            if is_accepted(rt_fb, tol_accept):
                rt = rt_fb

        dt_t = time.time() - t0
        accepted_flag = is_accepted(rt, tol_accept)
        if accepted_flag:
            prev_tex_X, prev_tex_Y = rt.X, rt.Y

        # Ratios (smooth_ref берётся как reference — НЕ пересчитывается).
        hr = rt.h_min / max(sref["h_min"], 1e-12)
        pr = rt.p_max / max(sref["p_max"], 1e-12)
        fr_r = rt.friction / max(sref["friction"], 1e-12)
        dcav = rt.cav_frac - sref["cav_frac"]
        deps = rt.eps - sref["eps"]

        print(f"  target={target*100:5.2f}%: "
              f"smooth_ref ε={sref['eps']:.4f}, "
              f"tex ε={rt.eps:.4f} (res={rt.rel_residual:.1e}), "
              f"K_mag={K_mag:.3e}, "
              f"{dt_t:.1f}с "
              f"{'✓' if accepted_flag else '✗'}")
        print(f"    h_t/h_s={hr:.4f}, p_t/p_s={pr:.4f}, "
              f"fr_t/fr_s={fr_r:.4f}, Δcav={dcav:+.4f}, "
              f"Δε={deps:+.4f}")

        # smooth_ref — бит-в-бит копия accepted-entry из manifest.
        pairs.append(dict(
            unload_share_target=target,
            smooth_ref=copy.deepcopy(sref),
            textured=result_to_dict(rt),
            accepted=bool(accepted_flag),
            ratios=dict(
                h_ratio=float(hr),
                p_ratio=float(pr),
                f_ratio=float(fr_r),
                delta_cav=float(dcav),
                delta_eps=float(deps),
            ),
        ))

    # ── CSV dump ──────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, "mag_textured_equilibrium.csv")
    rows = []
    for p in pairs:
        row = dict(unload_share_target=p["unload_share_target"],
                   accepted=p["accepted"])
        for k, v in p["smooth_ref"].items():
            row[f"s_{k}"] = v
        if p["textured"]:
            for k, v in p["textured"].items():
                row[f"t_{k}"] = v
        for k, v in p["ratios"].items():
            row[k] = v if v is not None else ""
        rows.append(row)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\nCSV: {csv_path}")

    # ── textured_compare.json ─────────────────────────────────────
    out = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source_manifest": os.path.abspath(manifest_path),
        "config_ref": {
            "tol_accept": tol_accept,
            "step_cap": step_cap,
            "eps_max": eps_max,
            "N_phi": N_phi,
            "N_Z": N_Z,
            "W_applied_N": [float(w_) for w_ in W_applied],
        },
        "texture": dict(
            zone_deg=[TEX_PHI_START, TEX_PHI_END],
            a_mm=TEX_A_MM, b_mm=TEX_B_MM,
            hp_um=TEX_HP_UM, sf=TEX_SF, profile=TEX_PROFILE,
            n_dimples=n_dimples),
        "pairs": pairs,
    }
    out_json = os.path.join(run_dir, "textured_compare.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"JSON: {out_json}")

    accepted_count = sum(1 for p in pairs if p["accepted"])
    print(f"\nAccepted textured pairs: {accepted_count}/{len(pairs)}")


if __name__ == "__main__":
    main()
