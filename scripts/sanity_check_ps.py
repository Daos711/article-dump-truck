#!/usr/bin/env python3
"""Sanity-check: конфиг A с Payvar-Salant кавитацией.

Plumbing test — проверяем что текстура подключена и считается.
Сетка 2800×570 (конфиг A), ε=0.6, минеральное масло.

Выводит:
- gain_W, gain_f
- Контурные карты H, P, θ
- Midplane P(φ) — гладкий vs текстурированный

Usage:
  python scripts/sanity_check_ps.py                       # полный расчёт
  python scripts/sanity_check_ps.py --plot-only           # только графики из npz
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL
from config.pump_params_micro import CONFIG_A

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"
EPS = 0.6
ETA = MINERAL_OIL["eta_pump"]


def make_params(cfg):
    p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                 for k in dir(base_params)
                                 if not k.startswith('_')})
    for k, v in cfg.items():
        if k not in ("label", "N_phi", "N_Z"):
            setattr(p, k, v)
    return p


def compute(out_dir):
    """Полный расчёт + сохранение в npz."""
    from models.bearing_model import (
        setup_grid, setup_texture, make_H, solve_and_compute,
    )

    cfg = CONFIG_A
    N_phi = cfg["N_phi"]
    N_Z = cfg["N_Z"]
    p = make_params(cfg)

    print("=" * 65)
    print(f"SANITY-CHECK: конфиг {cfg['label']}, PS, eps={EPS}")
    print(f"Сетка: {N_phi}x{N_Z}, профиль={PROFILE}")
    print("=" * 65)

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)
    phi_c, Z_c = setup_texture(p)

    print("\n  Гладкий...")
    t0 = time.time()
    H_smooth = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
    P_s, W_s, f_s, Q_s, hmin_s, pmax_s, Ftr_s, _, theta_s, cf_s = \
        solve_and_compute(
            H_smooth, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
            phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)
    dt_s = time.time() - t0
    print(f"    W={W_s:.0f} Н, f={f_s:.4f}, pmax={pmax_s/1e6:.1f} МПа, "
          f"cav={cf_s:.1%}, time={dt_s:.1f} с")

    print("\n  Текстурированный...")
    t0 = time.time()
    H_tex = make_H(EPS, Phi_mesh, Z_mesh, p, textured=True,
                   phi_c_flat=phi_c, Z_c_flat=Z_c, profile=PROFILE)
    P_t, W_t, f_t, Q_t, hmin_t, pmax_t, Ftr_t, _, theta_t, cf_t = \
        solve_and_compute(
            H_tex, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
            phi_1D, Z_1D, Phi_mesh, cavitation=CAVITATION)
    dt_t = time.time() - t0
    print(f"    W={W_t:.0f} Н, f={f_t:.4f}, pmax={pmax_t/1e6:.1f} МПа, "
          f"cav={cf_t:.1%}, time={dt_t:.1f} с")

    gain_W = W_t / W_s if W_s > 0 else 0
    gain_f = f_t / f_s if f_s > 0 else 0
    print(f"\n  gain_W = {gain_W:.4f}")
    print(f"  gain_f = {gain_f:.4f}")
    if gain_W > 2.0:
        print("  !!! gain_W > 2 — подозрительно")
    elif gain_W < 1.0:
        print("  !!! gain_W < 1 — текстура не помогает")
    else:
        print("  OK")

    omega = 2 * np.pi * p.n / 60.0
    p_scale = 6.0 * ETA * omega * (p.R / p.c) ** 2

    npz_path = os.path.join(out_dir, "data.npz")
    np.savez(npz_path,
             phi_1D=phi_1D, Z_1D=Z_1D,
             H_tex=H_tex, H_smooth=H_smooth,
             P_s=P_s, P_t=P_t,
             theta_s=theta_s if theta_s is not None else np.array([]),
             theta_t=theta_t if theta_t is not None else np.array([]),
             p_scale=p_scale, W_s=W_s, W_t=W_t, f_s=f_s, f_t=f_t,
             pmax_s=pmax_s, pmax_t=pmax_t, cf_s=cf_s, cf_t=cf_t,
             N_Z=N_Z, label=cfg["label"])
    print(f"\n  Данные: {npz_path}")

    return dict(phi_1D=phi_1D, Z_1D=Z_1D,
                H_tex=H_tex, P_s=P_s, P_t=P_t,
                theta_s=theta_s, theta_t=theta_t,
                p_scale=p_scale, N_Z=N_Z, label=cfg["label"])


def load_npz(data_dir):
    npz_path = os.path.join(data_dir, "data.npz")
    if not os.path.exists(npz_path):
        print(f"Не найден: {npz_path}")
        sys.exit(1)
    d = np.load(npz_path, allow_pickle=True)
    theta_s = d["theta_s"] if d["theta_s"].size > 0 else None
    theta_t = d["theta_t"] if d["theta_t"].size > 0 else None
    return dict(phi_1D=d["phi_1D"], Z_1D=d["Z_1D"],
                H_tex=d["H_tex"],
                P_s=d["P_s"], P_t=d["P_t"],
                theta_s=theta_s, theta_t=theta_t,
                p_scale=float(d["p_scale"]),
                N_Z=int(d["N_Z"]), label=str(d["label"]))


def plot(data, out_dir):
    phi_1D = data["phi_1D"]
    Z_1D = data["Z_1D"]
    H_tex = data["H_tex"]
    P_s = data["P_s"]
    P_t = data["P_t"]
    theta_s = data["theta_s"]
    theta_t = data["theta_t"]
    p_scale = data["p_scale"]
    N_Z = data["N_Z"]
    label = data["label"]
    phi_deg = np.rad2deg(phi_1D)

    fig, ax = plt.subplots(figsize=(10, 4))
    c1 = ax.contourf(phi_deg, Z_1D, H_tex, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=ax, label="H (безразмерный)")
    ax.set_xlabel("φ (°)")
    ax.set_ylabel("Z")
    ax.set_title(f"H(φ,Z) — конфиг {label}, ε={EPS}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "H_contour.png"), dpi=150)
    plt.close(fig)
    print(f"  H_contour.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    P_s_dim = P_s * p_scale / 1e6
    P_t_dim = P_t * p_scale / 1e6
    vmax = max(P_s_dim.max(), P_t_dim.max())
    for ax, Pd, title in [(axes[0], P_s_dim, "Гладкий"),
                           (axes[1], P_t_dim, f"Текстура {label}")]:
        c2 = ax.contourf(phi_deg, Z_1D, Pd, levels=50, cmap="hot_r",
                         vmin=0, vmax=vmax)
        fig.colorbar(c2, ax=ax, label="P (МПа)")
        ax.set_xlabel("φ (°)")
        ax.set_title(f"{title}, ε={EPS}")
    axes[0].set_ylabel("Z")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "P_contour.png"), dpi=150)
    plt.close(fig)
    print(f"  P_contour.png")

    if theta_t is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        for ax, th, title in [(axes[0], theta_s, "Гладкий"),
                               (axes[1], theta_t, f"Текстура {label}")]:
            if th is not None:
                c3 = ax.contourf(phi_deg, Z_1D, th, levels=50, cmap="RdYlBu")
                fig.colorbar(c3, ax=ax, label="θ")
            else:
                ax.text(0.5, 0.5, "θ = None", transform=ax.transAxes,
                        ha="center")
            ax.set_xlabel("φ (°)")
            ax.set_title(f"{title}, ε={EPS}")
        axes[0].set_ylabel("Z")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "theta_contour.png"), dpi=150)
        plt.close(fig)
        print(f"  theta_contour.png")

    iz_mid = N_Z // 2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(phi_deg, P_s_dim[iz_mid, :], "b-", linewidth=2, label="Гладкий")
    ax.plot(phi_deg, P_t_dim[iz_mid, :], "r-", linewidth=1.5,
            label=f"Текстура {label}")
    ax.axvspan(180, 360, alpha=0.08, color="green", label="Дивергентная зона")
    ax.set_xlabel("φ (°)")
    ax.set_ylabel("P (МПа)")
    ax.set_title(f"Midplane P(φ) при Z=0, ε={EPS}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "P_midplane.png"), dpi=150)
    plt.close(fig)
    print(f"  P_midplane.png")

    print(f"\n  Все графики → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Загрузить data.npz и перестроить графики")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Папка с data.npz для --plot-only")
    args = parser.parse_args()

    default_dir = os.path.join(os.path.dirname(__file__), "..",
                                "results", "pump", "sanity_check")
    out_dir = args.data_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        print(f"--plot-only: загрузка из {out_dir}")
        data = load_npz(out_dir)
    else:
        data = compute(out_dir)

    print("\nПостроение графиков:")
    plot(data, out_dir)


if __name__ == "__main__":
    main()
