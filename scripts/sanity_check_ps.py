#!/usr/bin/env python3
"""Sanity-check: конфиг A с Payvar-Salant кавитацией.

Plumbing test — проверяем что текстура подключена и считается.
Сетка 2800×570 (конфиг A), ε=0.6, минеральное масло.

Выводит:
- gain_W, gain_f
- Контурные карты H, P, θ
- Midplane P(φ) — гладкий vs текстурированный
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
)
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL
from config.pump_params_micro import CONFIG_A

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"
EPS = 0.6
ETA = MINERAL_OIL["eta_pump"]


def make_params(cfg):
    """Создать namespace из base_params + micro-config overrides."""
    p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                 for k in dir(base_params)
                                 if not k.startswith('_')})
    for k, v in cfg.items():
        if k not in ("label", "N_phi", "N_Z"):
            setattr(p, k, v)
    return p


def run_sanity():
    cfg = CONFIG_A
    N_phi = cfg["N_phi"]
    N_Z = cfg["N_Z"]
    p = make_params(cfg)

    print("=" * 65)
    print(f"SANITY-CHECK: конфиг {cfg['label']}, PS, eps={EPS}")
    print(f"Сетка: {N_phi}x{N_Z}, профиль={PROFILE}")
    print(f"Лунка: a={p.a_dim*1e3:.2f} мм, b={p.b_dim*1e3:.2f} мм, "
          f"hp={p.h_p*1e6:.0f} мкм")
    print(f"Зона: {p.phi_start_deg}°–{p.phi_end_deg}°, "
          f"N_tex={p.N_phi_tex}x{p.N_Z_tex}")
    print("=" * 65)

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)
    phi_c, Z_c = setup_texture(p)

    # === Гладкий ===
    print("\n  Гладкий...")
    t0 = time.time()
    H_smooth = make_H(EPS, Phi_mesh, Z_mesh, p, textured=False)
    P_s, W_s, f_s, Q_s, hmin_s, pmax_s, Ftr_s, _, theta_s, cf_s = \
        solve_and_compute(
            H_smooth, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
            phi_1D, Z_1D, Phi_mesh,
            cavitation=CAVITATION,
        )
    dt_s = time.time() - t0
    print(f"    W={W_s:.0f} Н, f={f_s:.4f}, pmax={pmax_s/1e6:.1f} МПа, "
          f"cav={cf_s:.1%}, time={dt_s:.1f} с")

    # === Текстурированный ===
    print("\n  Текстурированный...")
    t0 = time.time()
    H_tex = make_H(EPS, Phi_mesh, Z_mesh, p, textured=True,
                   phi_c_flat=phi_c, Z_c_flat=Z_c, profile=PROFILE)
    P_t, W_t, f_t, Q_t, hmin_t, pmax_t, Ftr_t, _, theta_t, cf_t = \
        solve_and_compute(
            H_tex, d_phi, d_Z, p.R, p.L, ETA, p.n, p.c,
            phi_1D, Z_1D, Phi_mesh,
            cavitation=CAVITATION,
        )
    dt_t = time.time() - t0
    print(f"    W={W_t:.0f} Н, f={f_t:.4f}, pmax={pmax_t/1e6:.1f} МПа, "
          f"cav={cf_t:.1%}, time={dt_t:.1f} с")

    # === Gain ===
    gain_W = W_t / W_s if W_s > 0 else 0
    gain_f = f_t / f_s if f_s > 0 else 0
    print(f"\n  gain_W = {gain_W:.4f}")
    print(f"  gain_f = {gain_f:.4f}")

    if gain_W > 2.0:
        print("  !!! gain_W > 2 — подозрительно, проверить сетку")
    elif gain_W < 1.0:
        print("  !!! gain_W < 1 — текстура не помогает при данных параметрах")
    else:
        print("  OK — gain в разумном диапазоне")

    # === Визуализации ===
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "pump", "sanity_check")
    os.makedirs(out_dir, exist_ok=True)

    omega = 2 * np.pi * p.n / 60.0
    p_scale = 6.0 * ETA * omega * (p.R / p.c) ** 2
    phi_deg = np.rad2deg(phi_1D)

    # --- 1. H(φ,Z) контурная карта текстурированного ---
    fig, ax = plt.subplots(figsize=(10, 4))
    c1 = ax.contourf(phi_deg, Z_1D, H_tex, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=ax, label="H (безразмерный)")
    ax.set_xlabel("φ (°)")
    ax.set_ylabel("Z")
    ax.set_title(f"H(φ,Z) — конфиг {cfg['label']}, ε={EPS}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "H_contour.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Сохранён: H_contour.png")

    # --- 2. P(φ,Z) — гладкий vs текстурированный ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    P_s_dim = P_s * p_scale / 1e6
    P_t_dim = P_t * p_scale / 1e6
    vmax = max(P_s_dim.max(), P_t_dim.max())
    for ax, Pd, title in [(axes[0], P_s_dim, "Гладкий"),
                           (axes[1], P_t_dim, f"Текстура {cfg['label']}")]:
        c2 = ax.contourf(phi_deg, Z_1D, Pd, levels=50, cmap="hot_r",
                         vmin=0, vmax=vmax)
        fig.colorbar(c2, ax=ax, label="P (МПа)")
        ax.set_xlabel("φ (°)")
        ax.set_title(f"{title}, ε={EPS}")
    axes[0].set_ylabel("Z")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "P_contour.png"), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: P_contour.png")

    # --- 3. θ(φ,Z) — кавитационная зона (если есть) ---
    if theta_t is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        for ax, th, title in [(axes[0], theta_s, "Гладкий"),
                               (axes[1], theta_t, f"Текстура {cfg['label']}")]:
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
        print(f"  Сохранён: theta_contour.png")

    # --- 4. Midplane P(φ) ---
    iz_mid = N_Z // 2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(phi_deg, P_s_dim[iz_mid, :], "b-", linewidth=2, label="Гладкий")
    ax.plot(phi_deg, P_t_dim[iz_mid, :], "r-", linewidth=1.5,
            label=f"Текстура {cfg['label']}")
    ax.set_xlabel("φ (°)", fontsize=12)
    ax.set_ylabel("P (МПа)", fontsize=12)
    ax.set_title(f"Midplane P(φ) при Z=0, ε={EPS}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Отметить дивергентную зону
    ax.axvspan(180, 360, alpha=0.08, color="green", label="Дивергентная зона")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "P_midplane.png"), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: P_midplane.png")

    print(f"\n  Все графики → {out_dir}")


if __name__ == "__main__":
    run_sanity()
