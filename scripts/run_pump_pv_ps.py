#!/usr/bin/env python3
"""Пересчёт подшипника насоса с PV+PS и без PV.

Этап 1: baseline текстура из pump_params.py (hp=15мкм, зона 90-270, 8×9).
Этап 2: кандидатные текстуры (если baseline не дал gain_W > 1).

Для каждого режима (с PV / без PV):
  4 конфигурации × 15 точек по ε.
"""
import sys
import os
import time
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
)
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from config import pump_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"
N_PHI = 800
N_Z = 200
EPSILON_VALUES = np.linspace(0.1, 0.8, 15)

OILS = [
    ("mineral", "Минеральное", MINERAL_OIL),
    ("rapeseed", "Рапсовое", RAPESEED_OIL),
]


def run_sweep(p, phi_c, Z_c, use_pv, tag=""):
    """Sweep по ε для 4 конфигов: smooth/tex × mineral/rapeseed.

    Returns dict[oil_key] -> dict с массивами.
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)
    omega = 2 * np.pi * p.n / 60.0
    U = omega * p.R
    n_eps = len(EPSILON_VALUES)

    all_results = {}
    for oil_key, oil_name, oil in OILS:
        eta = oil["eta_pump"]
        alpha_pv = oil.get("alpha_pv") if use_pv else None

        W = np.zeros((2, n_eps))    # [smooth, tex]
        f = np.zeros((2, n_eps))
        pmax = np.zeros((2, n_eps))
        cav = np.zeros((2, n_eps))
        F_tr = np.zeros((2, n_eps))

        for vi, (textured, vname) in enumerate([(False, "гладкий"),
                                                  (True, "текстура")]):
            P_prev = None
            pv_tag = "+PV" if use_pv else ""
            print(f"    {oil_name} / {vname} {pv_tag}:")

            for ie, eps in enumerate(EPSILON_VALUES):
                if textured:
                    H = make_H(eps, Phi_mesh, Z_mesh, p, textured=True,
                               phi_c_flat=phi_c, Z_c_flat=Z_c,
                               profile=PROFILE)
                else:
                    H = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)

                _, F, mu, _, _, pm, Ffr, _, _, cf = solve_and_compute(
                    H, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                    phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
                    cavitation=CAVITATION, alpha_pv=alpha_pv)
                P_prev = None  # не переиспользовать P_init между PV-расчётами

                W[vi, ie] = F
                f[vi, ie] = mu
                pmax[vi, ie] = pm
                cav[vi, ie] = cf
                F_tr[vi, ie] = Ffr

                print(f"      eps={eps:.2f}: W={F:8.0f}, f={mu:.4f}, "
                      f"pmax={pm/1e6:.1f}МПа, cav={cf:.1%}")

        gain_W = np.where(W[0] > 0, W[1] / W[0], np.nan)
        gain_f = np.where(f[0] > 0, f[1] / f[0], np.nan)
        gain_pmax = np.where(pmax[0] > 0, pmax[1] / pmax[0], np.nan)

        all_results[oil_key] = {
            "W": W, "f": f, "pmax": pmax, "cav": cav, "F_tr": F_tr,
            "gain_W": gain_W, "gain_f": gain_f, "gain_pmax": gain_pmax,
        }

    return all_results


def save_csv(results, label, out_dir):
    for oil_key, oil_name, _ in OILS:
        r = results[oil_key]
        path = os.path.join(out_dir, f"{label}_{oil_key}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["eps", "W_smooth", "W_tex", "gain_W",
                         "f_smooth", "f_tex", "gain_f",
                         "pmax_smooth_MPa", "pmax_tex_MPa", "gain_pmax",
                         "cav_smooth", "cav_tex"])
            for ie, eps in enumerate(EPSILON_VALUES):
                w.writerow([
                    f"{eps:.3f}",
                    f"{r['W'][0,ie]:.1f}", f"{r['W'][1,ie]:.1f}",
                    f"{r['gain_W'][ie]:.4f}",
                    f"{r['f'][0,ie]:.5f}", f"{r['f'][1,ie]:.5f}",
                    f"{r['gain_f'][ie]:.4f}",
                    f"{r['pmax'][0,ie]/1e6:.2f}", f"{r['pmax'][1,ie]/1e6:.2f}",
                    f"{r['gain_pmax'][ie]:.4f}",
                    f"{r['cav'][0,ie]:.3f}", f"{r['cav'][1,ie]:.3f}",
                ])
        print(f"  CSV: {path}")


def print_gain_table(results_nopv, results_pv, label):
    ie_ref = np.argmin(np.abs(EPSILON_VALUES - 0.6))
    eps_ref = EPSILON_VALUES[ie_ref]

    print(f"\n{'=' * 80}")
    print(f"GAIN при ε={eps_ref:.2f} — {label}")
    print(f"{'=' * 80}")
    print(f"{'масло':>12} {'gain_W noPV':>12} {'gain_W PV':>11} "
          f"{'Δgain_W':>9} {'gain_f noPV':>12} {'gain_f PV':>11}")
    print("-" * 80)

    for oil_key, oil_name, _ in OILS:
        gw_np = results_nopv[oil_key]["gain_W"][ie_ref]
        gw_pv = results_pv[oil_key]["gain_W"][ie_ref]
        gf_np = results_nopv[oil_key]["gain_f"][ie_ref]
        gf_pv = results_pv[oil_key]["gain_f"][ie_ref]
        delta = gw_pv - gw_np
        print(f"{oil_name:>12} {gw_np:12.4f} {gw_pv:11.4f} "
              f"{delta:+9.4f} {gf_np:12.4f} {gf_pv:11.4f}")


def plot_gain_comparison(results_nopv, results_pv, label, out_dir):
    """Графики gain_W(ε) с PV и без PV."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [(axes[0], "gain_W", "gain_W"),
                                 (axes[1], "gain_f", "gain_f")]:
        for oil_key, oil_name, _ in OILS:
            ls = "-" if oil_key == "mineral" else "--"
            ax.plot(EPSILON_VALUES, results_nopv[oil_key][metric],
                    color="blue", linestyle=ls, linewidth=2,
                    label=f"{oil_name}, без PV")
            ax.plot(EPSILON_VALUES, results_pv[oil_key][metric],
                    color="red", linestyle=ls, linewidth=2,
                    label=f"{oil_name}, с PV")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("ε")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Влияние PV на эффект текстуры — {label}", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"gain_pv_comparison_{label}.{ext}"),
                    dpi=300)
    plt.close(fig)
    print(f"  График: gain_pv_comparison_{label}.png/pdf")


def load_from_csv(out_dir, label):
    """Загрузить results dict из CSV, сохранённых save_csv()."""
    results = {}
    for oil_key, oil_name, _ in OILS:
        path = os.path.join(out_dir, f"{label}_{oil_key}.csv")
        if not os.path.exists(path):
            return None
        rows = []
        with open(path, "r", encoding="utf-8") as fcsv:
            reader = csv.DictReader(fcsv)
            rows = list(reader)
        n = len(rows)
        W = np.zeros((2, n))
        f_arr = np.zeros((2, n))
        pmax = np.zeros((2, n))
        cav = np.zeros((2, n))
        for i, row in enumerate(rows):
            W[0, i] = float(row["W_smooth"])
            W[1, i] = float(row["W_tex"])
            f_arr[0, i] = float(row["f_smooth"])
            f_arr[1, i] = float(row["f_tex"])
            pmax[0, i] = float(row["pmax_smooth_MPa"]) * 1e6
            pmax[1, i] = float(row["pmax_tex_MPa"]) * 1e6
            cav[0, i] = float(row["cav_smooth"])
            cav[1, i] = float(row["cav_tex"])
        gain_W = np.where(W[0] > 0, W[1] / W[0], np.nan)
        gain_f = np.where(f_arr[0] > 0, f_arr[1] / f_arr[0], np.nan)
        gain_pmax = np.where(pmax[0] > 0, pmax[1] / pmax[0], np.nan)
        results[oil_key] = {
            "W": W, "f": f_arr, "pmax": pmax, "cav": cav,
            "gain_W": gain_W, "gain_f": gain_f, "gain_pmax": gain_pmax,
        }
    return results


def run_baseline(out_dir):
    """Этап 1: baseline текстура из pump_params.py."""
    print(f"\n{'=' * 80}")
    print("ЭТАП 1: BASELINE (текстура из pump_params.py)")
    print(f"hp={params.h_p*1e6:.0f}мкм, зона {params.phi_start_deg}-"
          f"{params.phi_end_deg}°, N_tex={params.N_phi_tex}×{params.N_Z_tex}")
    print(f"Сетка: {N_PHI}×{N_Z}")
    print(f"{'=' * 80}")

    phi_c, Z_c = setup_texture(params)

    # Без PV
    print("\n  --- Без PV ---")
    t0 = time.time()
    results_nopv = run_sweep(params, phi_c, Z_c, use_pv=False)
    dt_nopv = time.time() - t0
    print(f"  Время без PV: {dt_nopv:.1f} с")
    save_csv(results_nopv, "baseline_nopv", out_dir)

    # С PV
    print("\n  --- С PV ---")
    t0 = time.time()
    results_pv = run_sweep(params, phi_c, Z_c, use_pv=True)
    dt_pv = time.time() - t0
    print(f"  Время с PV: {dt_pv:.1f} с")
    save_csv(results_pv, "baseline_pv", out_dir)

    print_gain_table(results_nopv, results_pv, "baseline")
    plot_gain_comparison(results_nopv, results_pv, "baseline", out_dir)

    return results_nopv, results_pv


def run_candidates(out_dir):
    """Этап 2: кандидатные текстуры с PV."""
    print(f"\n{'=' * 80}")
    print("ЭТАП 2: КАНДИДАТНЫЕ ТЕКСТУРЫ")
    print(f"{'=' * 80}")

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)

    candidates = [
        {"label": "cand1: 90-180 2.0/1.5 hp5",
         "phi_start": 90, "phi_end": 180,
         "a_dim": 2.0e-3, "b_dim": 1.5e-3, "h_p": 5e-6},
        {"label": "cand2: 180-360 2.0/1.5 hp5",
         "phi_start": 180, "phi_end": 360,
         "a_dim": 2.0e-3, "b_dim": 1.5e-3, "h_p": 5e-6},
    ]

    for cand in candidates:
        p = types.SimpleNamespace(**{k: getattr(params, k)
                                     for k in dir(params)
                                     if not k.startswith('_')})
        p.a_dim = cand["a_dim"]
        p.b_dim = cand["b_dim"]
        p.h_p = cand["h_p"]
        p.phi_start_deg = cand["phi_start"]
        p.phi_end_deg = cand["phi_end"]

        # Автоматический N_tex
        B = p.b_dim / p.R
        A = 2 * p.a_dim / p.L
        phi_span = np.deg2rad(cand["phi_end"] - cand["phi_start"])
        p.N_phi_tex = max(1, int(phi_span / (2.5 * 2 * B)))
        p.N_Z_tex = max(1, int(2.0 / (2.5 * 2 * A)))

        phi_c, Z_c = setup_texture(p)

        print(f"\n  {cand['label']} (N={len(phi_c)} лунок)")

        for use_pv in [False, True]:
            pv_tag = "PV" if use_pv else "noPV"
            for oil_key, oil_name, oil in OILS:
                eta = oil["eta_pump"]
                alpha_pv = oil.get("alpha_pv") if use_pv else None

                H_s = make_H(EPS_REF, Phi_mesh, Z_mesh, p, textured=False)
                _, W_s, f_s, _, _, _, _, _, _, _ = solve_and_compute(
                    H_s, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                    phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION,
                    alpha_pv=alpha_pv)

                H_t = make_H(EPS_REF, Phi_mesh, Z_mesh, p, textured=True,
                             phi_c_flat=phi_c, Z_c_flat=Z_c, profile=PROFILE)
                _, W_t, f_t, _, _, _, _, _, _, _ = solve_and_compute(
                    H_t, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                    phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION,
                    alpha_pv=alpha_pv)

                gw = W_t / W_s if W_s > 0 else 0
                gf = f_t / f_s if f_s > 0 else 0
                print(f"    {oil_name:>12s} {pv_tag}: "
                      f"W_s={W_s:.0f}, W_t={W_t:.0f}, "
                      f"gain_W={gw:.4f}, gain_f={gf:.4f}")


EPS_REF = 0.6


def main():
    parser = argparse.ArgumentParser(
        description="Пересчёт насоса с PV+PS")
    parser.add_argument("--stage", type=int, default=0,
                        help="0=оба, 1=baseline, 2=candidates")
    parser.add_argument("--plot-only", action="store_true",
                        help="Загрузить CSV и перестроить только графики")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    default_dir = os.path.join(os.path.dirname(__file__), "..",
                                "results", "pump_pv_ps")
    out_dir = args.data_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ПЕРЕСЧЁТ ПОДШИПНИКА НАСОСА — PV + PAYVAR-SALANT")
    print(f"Результаты → {out_dir}")
    print("=" * 80)

    if args.plot_only:
        print("--plot-only: загрузка CSV baseline")
        results_nopv = load_from_csv(out_dir, "baseline_nopv")
        results_pv = load_from_csv(out_dir, "baseline_pv")
        if results_nopv is None or results_pv is None:
            print("CSV не найдены")
            sys.exit(1)
        print_gain_table(results_nopv, results_pv, "baseline")
        plot_gain_comparison(results_nopv, results_pv, "baseline", out_dir)
        return

    if args.stage in (0, 1):
        results_nopv, results_pv = run_baseline(out_dir)

        # Проверка: есть ли gain_W > 1?
        ie_ref = np.argmin(np.abs(EPSILON_VALUES - 0.6))
        any_gain = False
        for oil_key in ["mineral", "rapeseed"]:
            if np.any(results_pv[oil_key]["gain_W"] > 1.0):
                any_gain = True

        if not any_gain and args.stage == 0:
            print("\nBaseline не дал gain_W > 1 с PV → запускаю этап 2")
            run_candidates(out_dir)
        elif any_gain:
            print("\nBaseline дал gain_W > 1 с PV — этап 2 не обязателен")

    elif args.stage == 2:
        run_candidates(out_dir)


if __name__ == "__main__":
    main()
