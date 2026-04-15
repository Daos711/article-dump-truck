#!/usr/bin/env python3
"""Textured case: дизельный main bearing с текстурой.

Читает smooth results → data-driven texture placement → прогон textured
до cycle convergence. Те же циклы, та же нагрузка, та же сетка — только
другой H через texture_relief.
"""
import sys
import os
import json
import time
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(CASE_DIR))
sys.path.insert(0, CASE_DIR)
sys.path.insert(0, THIS)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reynolds_solver import solve_ausas_journal_dynamic_gpu

import case_config as cfg
from scaling import (
    omega_from_rpm, pressure_scale, force_scale, mass_nondim,
    make_load_fn_from_crank, CYCLE_TAU, tau_to_crank_deg,
    dt_from_crank_step,
)
from texture_builder import (
    suggest_texture_zones, build_diesel_texture,
    compute_phi_hmin, compute_phi_pmax_est,
)

M_EFF_KG = 10.0

# ─── Texture config ───────────────────────────────────────────────
TEXTURE_ZONE_NAME = "upstream"       # upstream / centered / slightly_downstream
DIMPLE_DIAM_UM = 200
DIMPLE_DEPTH_UM = 5
AREA_DENSITY = 0.10


def load_smooth_results(case_dir, n_rpm, load_pct):
    """Загрузить smooth npz последнего цикла для texture placement."""
    smooth_dir = os.path.join(
        case_dir, "results", f"smooth_{n_rpm}rpm_{load_pct}pct")
    # Найти последний cycle_N.npz
    files = sorted([f for f in os.listdir(smooth_dir)
                     if f.startswith("cycle_") and f.endswith(".npz")])
    if not files:
        raise FileNotFoundError(
            f"Нет cycle_*.npz в {smooth_dir}. Сначала запусти "
            f"run_smooth_cycle.py")
    last = os.path.join(smooth_dir, files[-1])
    d = np.load(last)
    return dict(X=d["X"], Y=d["Y"], t=d["t"],
                 h_min=d["h_min"], p_max=d["p_max"],
                 smooth_dir=smooth_dir)


def load_dim_load_data(case_dir, n_rpm, load_pct):
    base = f"load_main_bearing_{n_rpm}rpm_{load_pct}pct"
    nd = np.genfromtxt(
        os.path.join(case_dir, "derived", base + "_nd.csv"),
        delimiter=",", skip_header=1)
    return make_load_fn_from_crank(
        tau_to_crank_deg(nd[:, 0]), nd[:, 1], nd[:, 2])


def run_cycles(load_fn, texture_relief, out_dir):
    """Цикл за циклом textured до сходимости."""
    omega = omega_from_rpm(cfg.n_rpm)
    F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
    p_sc = pressure_scale(cfg.eta, omega, cfg.R, cfg.c)
    M_nd = mass_nondim(M_EFF_KG, cfg.eta, omega, cfg.R, cfg.c)
    B_ausas = cfg.L_bearing / (2 * cfg.R)
    dt_val, NT_cycle = dt_from_crank_step(cfg.crank_step_deg)

    print("=" * 72)
    print("TEXTURED — diesel main bearing (Sun-geometry surrogate)")
    print(f"Сетка {cfg.N1}×{cfg.N2}, NT_cycle={NT_cycle}, "
          f"dt={dt_val:.5f}, crank_step={cfg.crank_step_deg}°")
    print(f"Texture: relief shape={texture_relief.shape}, "
          f"max depth nondim={texture_relief.max():.4f}")
    print("=" * 72)

    all_results = []
    X_prev, Y_prev = cfg.X0, cfg.Y0

    t_all0 = time.time()
    for cycle_i in range(cfg.max_cycles):
        print(f"\nCycle {cycle_i + 1}/{cfg.max_cycles}")
        t0 = time.time()
        result = solve_ausas_journal_dynamic_gpu(
            NT=NT_cycle, dt=dt_val,
            N_Z=cfg.N2 + 2, N_phi=cfg.N1 + 2,
            d_phi=1.0 / cfg.N1, d_Z=B_ausas / cfg.N2,
            R=0.5, L=1.0,
            mass_M=M_nd,
            load_fn=load_fn,
            X0=X_prev, Y0=Y_prev,
            texture_relief=texture_relief,
            omega_p=cfg.omega_p, omega_theta=cfg.omega_theta,
            tol_inner=cfg.tol_inner, max_inner=cfg.max_inner,
            scheme=cfg.scheme, verbose=False,
        )
        dt_cycle = time.time() - t0
        all_results.append(result)
        X_prev = float(result.X[-1])
        Y_prev = float(result.Y[-1])
        eps = np.sqrt(X_prev**2 + Y_prev**2)
        print(f"  X={X_prev:+.4f}, Y={Y_prev:+.4f}, ε={eps:.4f}, "
              f"h_min [{np.min(result.h_min):.4f}..{np.max(result.h_min):.4f}], "
              f"p_max [{np.min(result.p_max):.4f}..{np.max(result.p_max):.4f}], "
              f"{dt_cycle:.1f}с")

        np.savez(os.path.join(out_dir, f"cycle_{cycle_i}.npz"),
                  t=result.t, X=result.X, Y=result.Y,
                  WX=result.WX, WY=result.WY,
                  p_max=result.p_max, h_min=result.h_min,
                  cav_frac=result.cav_frac)

        # Convergence (как в smooth)
        if cycle_i >= cfg.warmup_cycles and cycle_i >= 1:
            r_prev = all_results[-2]
            n = min(len(r_prev.X), len(result.X))
            rms_X = np.sqrt(np.mean((np.asarray(r_prev.X[-n:])
                                      - np.asarray(result.X[-n:]))**2))
            rms_Y = np.sqrt(np.mean((np.asarray(r_prev.Y[-n:])
                                      - np.asarray(result.Y[-n:]))**2))
            scale_xy = max(np.max(np.abs(result.X[-n:])),
                            np.max(np.abs(result.Y[-n:])), 0.01)
            rel_XY = (rms_X + rms_Y) / scale_xy
            print(f"    convergence XY={rel_XY:.4f}")
            if rel_XY < 0.02:
                print(f"  CONVERGED")
                break

    t_all = time.time() - t_all0
    print(f"\nВсего циклов: {len(all_results)} за {t_all:.1f} с")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rpm", type=int, default=cfg.n_rpm)
    parser.add_argument("--load-pct", type=int, default=cfg.load_pct)
    parser.add_argument("--zone", type=str, default=TEXTURE_ZONE_NAME,
                        choices=["upstream", "centered", "slightly_downstream"])
    parser.add_argument("--dimple-um", type=int, default=DIMPLE_DIAM_UM)
    parser.add_argument("--depth-um", type=int, default=DIMPLE_DEPTH_UM)
    parser.add_argument("--density", type=float, default=AREA_DENSITY)
    args = parser.parse_args()

    out_dir = os.path.join(
        CASE_DIR, "results",
        f"textured_{args.n_rpm}rpm_{args.load_pct}pct_{args.zone}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Загрузить smooth результат
    smooth = load_smooth_results(CASE_DIR, args.n_rpm, args.load_pct)
    print(f"  Smooth загружен из {smooth['smooth_dir']}")

    # 2. Анализ loaded arc + выбор зоны
    zones = suggest_texture_zones(smooth["X"], smooth["Y"],
                                    zone_width_deg=90.0)
    selected = zones[args.zone]
    print(f"\n  Loaded arc center: {zones['loaded_arc_center_deg']:.1f}°, "
          f"FWHM: {zones['loaded_arc_fwhm_deg']:.1f}°")
    print(f"  Выбранная зона ({args.zone}): "
          f"{selected['phi_start_deg']:.1f}° – {selected['phi_end_deg']:.1f}°")

    # 3. Построить texture_relief
    B_ausas = cfg.L_bearing / (2 * cfg.R)
    d_phi = 1.0 / cfg.N1
    d_Z = B_ausas / cfg.N2
    relief, tex_info = build_diesel_texture(
        N_phi=cfg.N1 + 2, N_Z=cfg.N2 + 2,
        B_ausas=B_ausas, d_phi=d_phi, d_Z=d_Z,
        phi_start_deg=selected["phi_start_deg"],
        phi_end_deg=selected["phi_end_deg"],
        dimple_diameter_um=args.dimple_um,
        dimple_depth_um=args.depth_um,
        area_density=args.density,
        c_clearance_m=cfg.c, R_bearing_m=cfg.R, L_bearing_m=cfg.L_bearing,
    )
    print(f"  Лунок: {tex_info['N_tex']} "
          f"({tex_info['N_phi_tex']}×{tex_info['N_Z_tex']})")
    print(f"  H_p = {tex_info['H_p']:.4f}, "
          f"spacing = {tex_info['spacing_m']*1e3:.2f} мм")

    np.savez(os.path.join(out_dir, "texture_relief.npz"),
              relief=relief, **{k: v for k, v in tex_info.items()
                                 if not isinstance(v, dict)})

    # Квик-плот relief
    fig, ax = plt.subplots(figsize=(10, 4))
    phi_arr = np.linspace(0, 1, relief.shape[1])
    Z_arr = np.linspace(0, B_ausas, relief.shape[0])
    c = ax.contourf(phi_arr, Z_arr, relief, levels=30, cmap="viridis")
    fig.colorbar(c, ax=ax, label="relief (nondim)")
    ax.set_xlabel("φ (Ausas, [0,1])")
    ax.set_ylabel("Z (Ausas)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "texture_relief.png"), dpi=150)
    plt.close(fig)
    print(f"  texture_relief.png saved")

    # 4. Загрузить нагрузку
    load_fn = load_dim_load_data(CASE_DIR, args.n_rpm, args.load_pct)

    # 5. Run cycles
    all_results = run_cycles(load_fn, relief, out_dir)

    # 6. Save config
    with open(os.path.join(out_dir, "texture_config.json"),
               "w", encoding="utf-8") as f:
        json.dump({
            "zone": args.zone,
            "phi_start_deg": selected["phi_start_deg"],
            "phi_end_deg": selected["phi_end_deg"],
            "dimple_diameter_um": args.dimple_um,
            "dimple_depth_um": args.depth_um,
            "area_density": args.density,
            "N_tex": tex_info["N_tex"],
            "loaded_arc_center_deg": zones["loaded_arc_center_deg"],
            "loaded_arc_fwhm_deg": zones["loaded_arc_fwhm_deg"],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nВыход → {out_dir}")


if __name__ == "__main__":
    main()
