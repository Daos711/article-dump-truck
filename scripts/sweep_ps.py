#!/usr/bin/env python3
"""Полный sweep подшипника насоса с Payvar-Salant: конфиги A/B/C.

Для каждого конфига и каждого масла:
  - sweep ε = 0.1..0.8 (15 точек)
  - 4 конфигурации: гладкий/текстура × минеральное/рапсовое
  - gain-коэффициенты, CSV, сводные таблицы

Без пьезовязкости. Профиль smoothcap.
"""
import sys
import os
import time
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
)
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL
from config.pump_params_micro import CONFIG_A, CONFIG_B, CONFIG_C, ALL_CONFIGS

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"
EPSILON_VALUES = np.linspace(0.1, 0.8, 15)

OILS = [
    ("mineral", "Минеральное", MINERAL_OIL),
    ("rapeseed", "Рапсовое", RAPESEED_OIL),
]


def make_params(cfg):
    """Создать namespace из base_params + micro-config overrides."""
    p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                 for k in dir(base_params)
                                 if not k.startswith('_')})
    for k, v in cfg.items():
        if k not in ("label", "N_phi", "N_Z"):
            setattr(p, k, v)
    return p


def sweep_one_config(cfg, out_dir):
    """Полный sweep одного конфига: 4 варианта × 15 eps.

    Returns dict с результатами.
    """
    label = cfg["label"]
    N_phi = cfg["N_phi"]
    N_Z = cfg["N_Z"]
    p = make_params(cfg)
    n_eps = len(EPSILON_VALUES)

    print(f"\n{'=' * 70}")
    print(f"КОНФИГ {label}: a={p.a_dim*1e3:.2f} мм, b={p.b_dim*1e3:.2f} мм, "
          f"hp={p.h_p*1e6:.0f} мкм")
    print(f"Сетка: {N_phi}×{N_Z}, зона {p.phi_start_deg}°–{p.phi_end_deg}°, "
          f"N_tex={p.N_phi_tex}×{p.N_Z_tex}")
    print(f"{'=' * 70}")

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)
    phi_c, Z_c = setup_texture(p)

    omega = 2 * np.pi * p.n / 60.0
    U = omega * p.R

    all_results = {}

    for oil_key, oil_name, oil in OILS:
        eta = oil["eta_pump"]

        # Массивы: [smooth, textured] × n_eps
        W = np.zeros((2, n_eps))
        f = np.zeros((2, n_eps))
        hmin = np.zeros((2, n_eps))
        Q = np.zeros((2, n_eps))
        F_tr = np.zeros((2, n_eps))
        N_loss = np.zeros((2, n_eps))
        pmax = np.zeros((2, n_eps))
        cav_frac = np.zeros((2, n_eps))

        for variant_idx, (textured, var_name) in enumerate(
                [(False, "гладкий"), (True, "текстура")]):

            P_prev = None
            print(f"\n  {oil_name} / {var_name}:")

            for ie, eps in enumerate(EPSILON_VALUES):
                if textured:
                    H = make_H(eps, Phi_mesh, Z_mesh, p, textured=True,
                               phi_c_flat=phi_c, Z_c_flat=Z_c,
                               profile=PROFILE)
                else:
                    H = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)

                P, F, mu, Qv, h_m, p_m, F_friction, n_out, theta, cf = \
                    solve_and_compute(
                        H, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                        phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
                        cavitation=CAVITATION,
                    )
                P_prev = P

                W[variant_idx, ie] = F
                f[variant_idx, ie] = mu
                hmin[variant_idx, ie] = h_m
                Q[variant_idx, ie] = Qv
                F_tr[variant_idx, ie] = F_friction
                N_loss[variant_idx, ie] = F_friction * U
                pmax[variant_idx, ie] = p_m
                cav_frac[variant_idx, ie] = cf

                print(f"    eps={eps:.2f}: W={F:8.0f} Н, f={mu:.4f}, "
                      f"cav={cf:.1%}")

        # Gain
        gain_W = np.where(W[0] > 0, W[1] / W[0], np.nan)
        gain_f = np.where(f[0] > 0, f[1] / f[0], np.nan)
        gain_Ftr = np.where(F_tr[0] > 0, F_tr[1] / F_tr[0], np.nan)
        gain_Nloss = np.where(N_loss[0] > 0, N_loss[1] / N_loss[0], np.nan)
        gain_pmax = np.where(pmax[0] > 0, pmax[1] / pmax[0], np.nan)

        r = {
            "W": W, "f": f, "hmin": hmin, "Q": Q,
            "F_tr": F_tr, "N_loss": N_loss, "pmax": pmax, "cav_frac": cav_frac,
            "gain_W": gain_W, "gain_f": gain_f, "gain_Ftr": gain_Ftr,
            "gain_Nloss": gain_Nloss, "gain_pmax": gain_pmax,
        }
        all_results[oil_key] = r

        # CSV
        csv_path = os.path.join(out_dir,
                                f"sweep_config_{label}_{oil_key}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "eps",
                "W_smooth", "W_tex", "gain_W",
                "f_smooth", "f_tex", "gain_f",
                "Ftr_smooth", "Ftr_tex", "gain_Ftr",
                "Nloss_smooth", "Nloss_tex", "gain_Nloss",
                "pmax_smooth_MPa", "pmax_tex_MPa", "gain_pmax",
                "hmin_um", "Q_cm3s",
                "cav_smooth", "cav_tex",
            ])
            for ie, eps in enumerate(EPSILON_VALUES):
                writer.writerow([
                    f"{eps:.3f}",
                    f"{W[0,ie]:.1f}", f"{W[1,ie]:.1f}",
                    f"{gain_W[ie]:.4f}",
                    f"{f[0,ie]:.5f}", f"{f[1,ie]:.5f}",
                    f"{gain_f[ie]:.4f}",
                    f"{F_tr[0,ie]:.2f}", f"{F_tr[1,ie]:.2f}",
                    f"{gain_Ftr[ie]:.4f}",
                    f"{N_loss[0,ie]:.1f}", f"{N_loss[1,ie]:.1f}",
                    f"{gain_Nloss[ie]:.4f}",
                    f"{pmax[0,ie]/1e6:.2f}", f"{pmax[1,ie]/1e6:.2f}",
                    f"{gain_pmax[ie]:.4f}",
                    f"{hmin[1,ie]*1e6:.2f}",
                    f"{Q[1,ie]*1e6:.4f}",
                    f"{cav_frac[0,ie]:.3f}", f"{cav_frac[1,ie]:.3f}",
                ])
        print(f"\n  CSV: {csv_path}")

    return all_results


def print_gain_summary(label, all_results, eps_ref=0.6):
    """Печатает сводную таблицу gain при опорном ε."""
    ie_ref = np.argmin(np.abs(EPSILON_VALUES - eps_ref))
    eps_actual = EPSILON_VALUES[ie_ref]

    print(f"\n{'=' * 70}")
    print(f"GAIN при ε = {eps_actual:.2f} — конфиг {label}")
    print(f"{'=' * 70}")
    print(f"{'масло':>12} {'gain_W':>8} {'gain_f':>8} {'gain_Ftr':>9} "
          f"{'gain_Nloss':>10} {'gain_pmax':>10}")
    print("-" * 60)

    for oil_key, oil_name, _ in OILS:
        r = all_results[oil_key]
        gw = r["gain_W"][ie_ref]
        gf = r["gain_f"][ie_ref]
        gft = r["gain_Ftr"][ie_ref]
        gn = r["gain_Nloss"][ie_ref]
        gp = r["gain_pmax"][ie_ref]
        print(f"{oil_name:>12} {gw:8.4f} {gf:8.4f} {gft:9.4f} "
              f"{gn:10.4f} {gp:10.4f}")

        if gw > 2.0:
            print(f"  !!! gain_W > 2 — подозрительно!")
        elif gw < 1.0:
            print(f"  !!! gain_W < 1 — текстура ухудшает W")


def main():
    parser = argparse.ArgumentParser(
        description="Полный sweep с Payvar-Salant")
    parser.add_argument("--configs", type=str, default="B",
                        help="Конфиги для расчёта: A, B, C или AB, ABC "
                             "(default: B)")
    args = parser.parse_args()

    config_map = {"A": CONFIG_A, "B": CONFIG_B, "C": CONFIG_C}
    configs_to_run = []
    for ch in args.configs.upper():
        if ch in config_map:
            configs_to_run.append(config_map[ch])
        else:
            print(f"Неизвестный конфиг: {ch}")
            sys.exit(1)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pump_ps")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("ПОЛНЫЙ SWEEP ПОДШИПНИКА НАСОСА — PAYVAR-SALANT JFO")
    print(f"Конфиги: {args.configs.upper()}")
    print(f"ε = {EPSILON_VALUES[0]:.2f}..{EPSILON_VALUES[-1]:.2f} "
          f"({len(EPSILON_VALUES)} точек)")
    print(f"Масла: минеральное, рапсовое")
    print(f"Профиль: {PROFILE}, без пьезовязкости")
    print(f"Результаты → {out_dir}")
    print("=" * 70)

    t0_total = time.time()

    for cfg in configs_to_run:
        t0 = time.time()
        results = sweep_one_config(cfg, out_dir)
        dt = time.time() - t0
        print(f"\nВремя конфиг {cfg['label']}: {dt:.1f} с")
        print_gain_summary(cfg["label"], results)

    dt_total = time.time() - t0_total
    print(f"\n{'=' * 70}")
    print(f"ОБЩЕЕ ВРЕМЯ: {dt_total:.1f} с")
    print(f"{'=' * 70}")

    # Сохранить npz со всеми данными
    save_dict = {"epsilon": EPSILON_VALUES}
    for cfg in configs_to_run:
        # Перепрогнать не нужно — CSV уже есть.
        # Для npz загружаем из CSV (или можно кэшировать в main).
        pass
    print(f"\nCSV файлы в {out_dir}/")


if __name__ == "__main__":
    main()
