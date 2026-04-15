#!/usr/bin/env python3
"""Графики и таблицы для статьи — Payvar-Salant.

Режимы:
  --from-csv DIR   Построить из готовых CSV (после sweep_ps.py)
  --live           Запустить sweep + графики в одном прогоне

Графики:
  1. W(ε) — 4 кривые
  2. f(ε) — 4 кривые
  3. gain_W(ε) — 2 кривые
  4. gain_f(ε) — 2 кривые
  5. P(φ,Z) контурная карта — гладкий vs текстура
  6. θ(φ,Z) контурная карта — кавитационная зона
  7. Midplane P(φ) — наложенные
  8. Midplane θ(φ)

Таблица для статьи: сводка при ε=0.6.
"""
import sys
import os
import csv
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL
from config.pump_params_micro import CONFIG_A, CONFIG_B, CONFIG_C

# ─── Стиль для ГИАБ ──────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# Times New Roman может быть недоступен — fallback на serif
try:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
except Exception:
    pass

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"

OILS_META = {
    "mineral": {"name": "Минеральное", "oil": MINERAL_OIL,
                "color_s": "royalblue", "color_t": "firebrick",
                "ls_s": "-", "ls_t": "-"},
    "rapeseed": {"name": "Рапсовое", "oil": RAPESEED_OIL,
                 "color_s": "royalblue", "color_t": "firebrick",
                 "ls_s": "--", "ls_t": "--"},
}


# ===================================================================
#  CSV loading
# ===================================================================

def load_sweep_csv(csv_path):
    """Загрузить CSV из sweep_ps.py.

    Returns dict с numpy-массивами.
    """
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    eps = np.array([float(r["eps"]) for r in rows])
    data["eps"] = eps

    for key in ["W_smooth", "W_tex", "gain_W",
                "f_smooth", "f_tex", "gain_f",
                "Ftr_smooth", "Ftr_tex", "gain_Ftr",
                "Nloss_smooth", "Nloss_tex", "gain_Nloss",
                "pmax_smooth_MPa", "pmax_tex_MPa", "gain_pmax",
                "hmin_um", "Q_cm3s", "cav_smooth", "cav_tex"]:
        data[key] = np.array([float(r[key]) for r in rows])

    return data


def load_config_data(data_dir, label):
    """Загрузить данные обоих масел для конфига label."""
    result = {}
    for oil_key in ["mineral", "rapeseed"]:
        path = os.path.join(data_dir, f"sweep_config_{label}_{oil_key}.csv")
        if os.path.exists(path):
            result[oil_key] = load_sweep_csv(path)
    return result


# ===================================================================
#  Live computation (fallback, когда CSV нет)
# ===================================================================

def make_params(cfg):
    p = types.SimpleNamespace(**{k: getattr(base_params, k)
                                 for k in dir(base_params)
                                 if not k.startswith('_')})
    for k, v in cfg.items():
        if k not in ("label", "N_phi", "N_Z"):
            setattr(p, k, v)
    return p


def compute_live(cfg, out_dir):
    """Полный sweep одного конфига — возвращает данные + сохраняет CSV.

    Дублирует логику sweep_ps.py, но также возвращает P, theta для контуров.
    """
    from models.bearing_model import (
        setup_grid, setup_texture, make_H, solve_and_compute,
    )

    label = cfg["label"]
    N_phi, N_Z = cfg["N_phi"], cfg["N_Z"]
    p = make_params(cfg)
    epsilon_values = np.linspace(0.1, 0.8, 15)
    n_eps = len(epsilon_values)

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_phi, N_Z)
    phi_c, Z_c = setup_texture(p)
    omega = 2 * np.pi * p.n / 60.0
    U = omega * p.R

    all_data = {}
    # Для контуров при ε=0.6
    contour_data = {}

    ie_ref = np.argmin(np.abs(epsilon_values - 0.6))

    for oil_key in ["mineral", "rapeseed"]:
        meta = OILS_META[oil_key]
        eta = meta["oil"]["eta_pump"]

        d = {"eps": epsilon_values}
        W_s, W_t = np.zeros(n_eps), np.zeros(n_eps)
        f_s, f_t = np.zeros(n_eps), np.zeros(n_eps)
        Ftr_s, Ftr_t = np.zeros(n_eps), np.zeros(n_eps)
        Nloss_s, Nloss_t = np.zeros(n_eps), np.zeros(n_eps)
        pmax_s, pmax_t = np.zeros(n_eps), np.zeros(n_eps)
        hmin_arr = np.zeros(n_eps)
        Q_arr = np.zeros(n_eps)
        cav_s, cav_t = np.zeros(n_eps), np.zeros(n_eps)

        P_prev_s, P_prev_t = None, None
        print(f"\n  {meta['name']} — конфиг {label}...")

        for ie, eps in enumerate(epsilon_values):
            # Smooth
            H_smooth = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
            P, F, mu, Qv, h_m, p_m, Ffr, _, theta, cf = \
                solve_and_compute(
                    H_smooth, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                    phi_1D, Z_1D, Phi_mesh, P_init=P_prev_s,
                    cavitation=CAVITATION)
            P_prev_s = P
            W_s[ie], f_s[ie] = F, mu
            Ftr_s[ie], Nloss_s[ie] = Ffr, Ffr * U
            pmax_s[ie], cav_s[ie] = p_m, cf

            if ie == ie_ref and oil_key == "mineral":
                p_scale = 6.0 * eta * omega * (p.R / p.c) ** 2
                contour_data["P_smooth"] = P * p_scale
                contour_data["theta_smooth"] = theta
                contour_data["phi_1D"] = phi_1D
                contour_data["Z_1D"] = Z_1D

            # Textured
            H_tex = make_H(eps, Phi_mesh, Z_mesh, p, textured=True,
                           phi_c_flat=phi_c, Z_c_flat=Z_c, profile=PROFILE)
            P, F, mu, Qv, h_m, p_m, Ffr, _, theta, cf = \
                solve_and_compute(
                    H_tex, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                    phi_1D, Z_1D, Phi_mesh, P_init=P_prev_t,
                    cavitation=CAVITATION)
            P_prev_t = P
            W_t[ie], f_t[ie] = F, mu
            Ftr_t[ie], Nloss_t[ie] = Ffr, Ffr * U
            pmax_t[ie], cav_t[ie] = p_m, cf
            hmin_arr[ie], Q_arr[ie] = h_m, Qv

            if ie == ie_ref and oil_key == "mineral":
                contour_data["P_tex"] = P * p_scale
                contour_data["theta_tex"] = theta

            print(f"    eps={eps:.2f}: W_s={W_s[ie]:.0f}, W_t={W_t[ie]:.0f}, "
                  f"gain_W={W_t[ie]/max(W_s[ie],1):.4f}")

        gain_W = np.where(W_s > 0, W_t / W_s, np.nan)
        gain_f = np.where(f_s > 0, f_t / f_s, np.nan)

        d.update({
            "W_smooth": W_s, "W_tex": W_t, "gain_W": gain_W,
            "f_smooth": f_s, "f_tex": f_t, "gain_f": gain_f,
            "Ftr_smooth": Ftr_s, "Ftr_tex": Ftr_t,
            "gain_Ftr": np.where(Ftr_s > 0, Ftr_t / Ftr_s, np.nan),
            "Nloss_smooth": Nloss_s, "Nloss_tex": Nloss_t,
            "gain_Nloss": np.where(Nloss_s > 0, Nloss_t / Nloss_s, np.nan),
            "pmax_smooth_MPa": pmax_s / 1e6, "pmax_tex_MPa": pmax_t / 1e6,
            "gain_pmax": np.where(pmax_s > 0, pmax_t / pmax_s, np.nan),
            "hmin_um": hmin_arr * 1e6, "Q_cm3s": Q_arr * 1e6,
            "cav_smooth": cav_s, "cav_tex": cav_t,
        })
        all_data[oil_key] = d

        # Сохранить CSV
        csv_path = os.path.join(out_dir,
                                f"sweep_config_{label}_{oil_key}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "eps", "W_smooth", "W_tex", "gain_W",
                "f_smooth", "f_tex", "gain_f",
                "Ftr_smooth", "Ftr_tex", "gain_Ftr",
                "Nloss_smooth", "Nloss_tex", "gain_Nloss",
                "pmax_smooth_MPa", "pmax_tex_MPa", "gain_pmax",
                "hmin_um", "Q_cm3s", "cav_smooth", "cav_tex",
            ])
            for ie in range(n_eps):
                writer.writerow([
                    f"{epsilon_values[ie]:.3f}",
                    f"{W_s[ie]:.1f}", f"{W_t[ie]:.1f}",
                    f"{gain_W[ie]:.4f}",
                    f"{f_s[ie]:.5f}", f"{f_t[ie]:.5f}",
                    f"{gain_f[ie]:.4f}",
                    f"{Ftr_s[ie]:.2f}", f"{Ftr_t[ie]:.2f}",
                    f"{d['gain_Ftr'][ie]:.4f}",
                    f"{Nloss_s[ie]:.1f}", f"{Nloss_t[ie]:.1f}",
                    f"{d['gain_Nloss'][ie]:.4f}",
                    f"{pmax_s[ie]/1e6:.2f}", f"{pmax_t[ie]/1e6:.2f}",
                    f"{d['gain_pmax'][ie]:.4f}",
                    f"{hmin_arr[ie]*1e6:.2f}", f"{Q_arr[ie]*1e6:.4f}",
                    f"{cav_s[ie]:.3f}", f"{cav_t[ie]:.3f}",
                ])

    return all_data, contour_data


# ===================================================================
#  Plotting functions
# ===================================================================

def plot_W_eps(data, label, out_dir):
    """Рис. 1: W(ε) — 4 кривые."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for oil_key, meta in OILS_META.items():
        d = data[oil_key]
        ax.plot(d["eps"], d["W_smooth"], color=meta["color_s"],
                linestyle=meta["ls_s"], linewidth=2,
                label=f"Гладкий, {meta['name']}")
        ax.plot(d["eps"], d["W_tex"], color=meta["color_t"],
                linestyle=meta["ls_t"], linewidth=2,
                label=f"Текстура {label}, {meta['name']}")
    ax.set_xlabel("Эксцентриситет ε")
    ax.set_ylabel("Несущая способность W, Н")
    ax.set_title(f"W(ε) — конфиг {label}, Payvar-Salant")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"W_eps_{label}.{ext}"))
    plt.close(fig)
    print(f"  W_eps_{label}.png/pdf")


def plot_f_eps(data, label, out_dir):
    """Рис. 2: f(ε) — 4 кривые."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for oil_key, meta in OILS_META.items():
        d = data[oil_key]
        ax.plot(d["eps"], d["f_smooth"], color=meta["color_s"],
                linestyle=meta["ls_s"], linewidth=2,
                label=f"Гладкий, {meta['name']}")
        ax.plot(d["eps"], d["f_tex"], color=meta["color_t"],
                linestyle=meta["ls_t"], linewidth=2,
                label=f"Текстура {label}, {meta['name']}")
    ax.set_xlabel("Эксцентриситет ε")
    ax.set_ylabel("Коэффициент трения f")
    ax.set_title(f"f(ε) — конфиг {label}, Payvar-Salant")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"f_eps_{label}.{ext}"))
    plt.close(fig)
    print(f"  f_eps_{label}.png/pdf")


def plot_gain_W(data, label, out_dir):
    """Рис. 3: gain_W(ε) — 2 кривые."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for oil_key, meta in OILS_META.items():
        d = data[oil_key]
        ax.plot(d["eps"], d["gain_W"], color=meta["color_t"],
                linestyle=meta["ls_t"], linewidth=2,
                marker="o", markersize=4, label=meta["name"])
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Эксцентриситет ε")
    ax.set_ylabel("gain_W = W_текст / W_гладк")
    ax.set_title(f"gain_W(ε) — конфиг {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"gain_W_{label}.{ext}"))
    plt.close(fig)
    print(f"  gain_W_{label}.png/pdf")


def plot_gain_f(data, label, out_dir):
    """Рис. 4: gain_f(ε) — 2 кривые."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for oil_key, meta in OILS_META.items():
        d = data[oil_key]
        ax.plot(d["eps"], d["gain_f"], color=meta["color_t"],
                linestyle=meta["ls_t"], linewidth=2,
                marker="s", markersize=4, label=meta["name"])
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Эксцентриситет ε")
    ax.set_ylabel("gain_f = f_текст / f_гладк")
    ax.set_title(f"gain_f(ε) — конфиг {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"gain_f_{label}.{ext}"))
    plt.close(fig)
    print(f"  gain_f_{label}.png/pdf")


def plot_contours(contour_data, label, out_dir):
    """Рис. 5-6: контурные карты P и θ при ε=0.6."""
    phi_deg = np.rad2deg(contour_data["phi_1D"])
    Z_1D = contour_data["Z_1D"]

    # --- P(φ,Z) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    P_s = contour_data["P_smooth"] / 1e6  # МПа
    P_t = contour_data["P_tex"] / 1e6
    vmax = max(np.nanmax(P_s), np.nanmax(P_t))

    for ax, Pd, title in [(axes[0], P_s, "Гладкий"),
                           (axes[1], P_t, f"Текстура {label}")]:
        c = ax.contourf(phi_deg, Z_1D, Pd, levels=50, cmap="hot_r",
                        vmin=0, vmax=vmax)
        fig.colorbar(c, ax=ax, label="P, МПа")
        ax.set_xlabel("φ, °")
        ax.set_title(title)
    axes[0].set_ylabel("Z")
    fig.suptitle("Поле давления P(φ, Z) при ε = 0.6", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"P_contour_{label}.{ext}"))
    plt.close(fig)
    print(f"  P_contour_{label}.png/pdf")

    # --- θ(φ,Z) ---
    th_s = contour_data.get("theta_smooth")
    th_t = contour_data.get("theta_tex")
    if th_t is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, th, title in [(axes[0], th_s, "Гладкий"),
                               (axes[1], th_t, f"Текстура {label}")]:
            if th is not None:
                c = ax.contourf(phi_deg, Z_1D, th, levels=50, cmap="RdYlBu")
                fig.colorbar(c, ax=ax, label="θ")
            else:
                ax.text(0.5, 0.5, "θ недоступно", transform=ax.transAxes,
                        ha="center", fontsize=14)
            ax.set_xlabel("φ, °")
            ax.set_title(title)
        axes[0].set_ylabel("Z")
        fig.suptitle("Поле заполнения θ(φ, Z) при ε = 0.6", fontsize=13)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"theta_contour_{label}.{ext}"))
        plt.close(fig)
        print(f"  theta_contour_{label}.png/pdf")


def plot_midplane(contour_data, label, out_dir):
    """Рис. 7-8: midplane P(φ) и θ(φ) при Z=0."""
    phi_deg = np.rad2deg(contour_data["phi_1D"])
    Z_1D = contour_data["Z_1D"]
    iz_mid = len(Z_1D) // 2

    P_s = contour_data["P_smooth"] / 1e6
    P_t = contour_data["P_tex"] / 1e6

    # --- P(φ) midplane ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(phi_deg, P_s[iz_mid, :], "b-", linewidth=2, label="Гладкий")
    ax.plot(phi_deg, P_t[iz_mid, :], "r-", linewidth=1.5,
            label=f"Текстура {label}")
    ax.axvspan(180, 360, alpha=0.07, color="green")
    ax.text(270, ax.get_ylim()[1] * 0.9, "Диверг.", ha="center",
            fontsize=10, color="green", alpha=0.7)
    ax.set_xlabel("φ, °")
    ax.set_ylabel("P, МПа")
    ax.set_title("Давление в мидплоскости (Z = 0) при ε = 0.6")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"P_midplane_{label}.{ext}"))
    plt.close(fig)
    print(f"  P_midplane_{label}.png/pdf")

    # --- θ(φ) midplane ---
    th_s = contour_data.get("theta_smooth")
    th_t = contour_data.get("theta_tex")
    if th_t is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        if th_s is not None:
            ax.plot(phi_deg, th_s[iz_mid, :], "b-", linewidth=2,
                    label="Гладкий")
        ax.plot(phi_deg, th_t[iz_mid, :], "r-", linewidth=1.5,
                label=f"Текстура {label}")
        ax.axvspan(180, 360, alpha=0.07, color="green")
        ax.set_xlabel("φ, °")
        ax.set_ylabel("θ")
        ax.set_title("Заполнение θ в мидплоскости (Z = 0) при ε = 0.6")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"theta_midplane_{label}.{ext}"))
        plt.close(fig)
        print(f"  theta_midplane_{label}.png/pdf")


# ===================================================================
#  Summary table
# ===================================================================

def print_article_table(all_configs_data, out_dir):
    """Сводная таблица при ε=0.6 для всех конфигов."""
    eps_ref = 0.6

    print(f"\n{'=' * 90}")
    print(f"СВОДНАЯ ТАБЛИЦА ДЛЯ СТАТЬИ (ε = {eps_ref})")
    print(f"{'=' * 90}")
    header = (f"{'Конфиг':<10} {'Масло':<12} {'W, Н':>8} {'gain_W':>8} "
              f"{'f':>8} {'gain_f':>8} {'p_max МПа':>10} {'cav%':>6}")
    print(header)
    print("-" * 90)

    rows = []

    for label, data in all_configs_data.items():
        for oil_key, meta in OILS_META.items():
            if oil_key not in data:
                continue
            d = data[oil_key]
            eps = d["eps"]
            ie = np.argmin(np.abs(eps - eps_ref))

            # Гладкий
            if label == list(all_configs_data.keys())[0]:
                row_s = {
                    "config": "Гладкий", "oil": meta["name"],
                    "W": d["W_smooth"][ie], "gain_W": 1.0,
                    "f": d["f_smooth"][ie], "gain_f": 1.0,
                    "pmax": d["pmax_smooth_MPa"][ie],
                    "cav": d["cav_smooth"][ie],
                }
                rows.append(row_s)
                print(f"{'Гладкий':<10} {meta['name']:<12} "
                      f"{row_s['W']:8.0f} {1.0:8.4f} "
                      f"{row_s['f']:8.4f} {1.0:8.4f} "
                      f"{row_s['pmax']:10.1f} {row_s['cav']*100:6.1f}")

            row_t = {
                "config": label, "oil": meta["name"],
                "W": d["W_tex"][ie], "gain_W": d["gain_W"][ie],
                "f": d["f_tex"][ie], "gain_f": d["gain_f"][ie],
                "pmax": d["pmax_tex_MPa"][ie],
                "cav": d["cav_tex"][ie],
            }
            rows.append(row_t)
            print(f"{label:<10} {meta['name']:<12} "
                  f"{row_t['W']:8.0f} {row_t['gain_W']:8.4f} "
                  f"{row_t['f']:8.4f} {row_t['gain_f']:8.4f} "
                  f"{row_t['pmax']:10.1f} {row_t['cav']*100:6.1f}")

    # Рекомендация
    tex_rows = [r for r in rows if r["config"] != "Гладкий"]
    if tex_rows:
        best = max(tex_rows, key=lambda r: r["gain_W"])
        print(f"\nРекомендация: конфиг {best['config']}, {best['oil']} "
              f"— gain_W = {best['gain_W']:.4f}")

    # CSV
    csv_path = os.path.join(out_dir, "article_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "oil", "W_N", "gain_W", "f", "gain_f",
                     "pmax_MPa", "cav_frac"])
        for r in rows:
            w.writerow([r["config"], r["oil"],
                        f"{r['W']:.1f}", f"{r['gain_W']:.4f}",
                        f"{r['f']:.5f}", f"{r['gain_f']:.4f}",
                        f"{r['pmax']:.2f}", f"{r['cav']:.3f}"])
    print(f"\n  Таблица: {csv_path}")


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Графики для статьи — Payvar-Salant")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Папка с CSV от sweep_ps.py")
    parser.add_argument("--live", action="store_true",
                        help="Запустить sweep + графики")
    parser.add_argument("--configs", type=str, default="B",
                        help="Конфиги: B, AB, ABC (default: B)")
    args = parser.parse_args()

    config_map = {"A": CONFIG_A, "B": CONFIG_B, "C": CONFIG_C}
    configs = []
    for ch in args.configs.upper():
        if ch in config_map:
            configs.append((ch, config_map[ch]))

    out_dir = args.from_csv or os.path.join(
        os.path.dirname(__file__), "..", "results", "pump_ps")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("ГРАФИКИ ДЛЯ СТАТЬИ — PAYVAR-SALANT")
    print(f"Конфиги: {args.configs.upper()}")
    print(f"Результаты → {out_dir}")
    print("=" * 70)

    all_configs_data = {}
    contour_data = None

    if args.from_csv:
        # Загрузить из CSV
        for label, cfg in configs:
            data = load_config_data(args.from_csv, label)
            if data:
                all_configs_data[label] = data
                print(f"  Загружен конфиг {label}: "
                      f"{list(data.keys())}")
            else:
                print(f"  CSV для конфига {label} не найден, пропускаю")
    elif args.live:
        # Live расчёт
        for label, cfg in configs:
            t0 = time.time()
            data, cd = compute_live(cfg, out_dir)
            dt = time.time() - t0
            all_configs_data[label] = data
            if contour_data is None:
                contour_data = cd
            print(f"\n  Конфиг {label}: {dt:.1f} с")
    else:
        # Попробовать CSV из default пути
        for label, cfg in configs:
            data = load_config_data(out_dir, label)
            if data:
                all_configs_data[label] = data
        if not all_configs_data:
            print("\nCSV не найдены. Используйте --live или --from-csv DIR")
            sys.exit(1)

    if not all_configs_data:
        print("Нет данных для построения графиков.")
        sys.exit(1)

    # --- Графики ---
    print(f"\nПостроение графиков:")
    for label in all_configs_data:
        data = all_configs_data[label]
        plot_W_eps(data, label, out_dir)
        plot_f_eps(data, label, out_dir)
        plot_gain_W(data, label, out_dir)
        plot_gain_f(data, label, out_dir)

    # Контурные карты (только в live-режиме, когда есть 2D-поля)
    if contour_data:
        label_0 = list(all_configs_data.keys())[0]
        plot_contours(contour_data, label_0, out_dir)
        plot_midplane(contour_data, label_0, out_dir)

    # --- Таблица ---
    print_article_table(all_configs_data, out_dir)

    print(f"\nВсе результаты → {out_dir}")


if __name__ == "__main__":
    main()
