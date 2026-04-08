#!/usr/bin/env python3
"""Запуск нестационарного расчёта подшипника ДВС.

Time-stepping: совместное решение Reynolds со squeeze + уравнение движения вала.
Генерирует графики ε(φ), h_min(φ), f(φ), p_max(φ), Fy_ext(φ), орбиту вала,
summary.txt и data.npz для 4 конфигураций.
"""
import sys
import os
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from models.diesel_transient import run_transient, load_diesel, CONFIGS
from config import diesel_params as params


def make_results_dir(debug=False):
    """Создать папку: results/diesel/YYMMDD_HHMM_F850kN_hp30_zone90-225/"""
    F_kN = int((params.F_max_debug if debug else params.F_max) / 1000)
    hp_um = int(params.h_p * 1e6)
    tag = datetime.now().strftime("%y%m%d_%H%M")
    name = f"{tag}_F{F_kN}kN_hp{hp_um}_zone{params.phi_start_deg}-{params.phi_end_deg}"
    d = os.path.join(os.path.dirname(__file__), "..", "results", "diesel", name)
    os.makedirs(d, exist_ok=True)
    return d


SMOOTH_WINDOW = 15


def plot_curves(phi, data, ylabel, filename, title, results_dir):
    """Построить график: тонкая сырая + жирная сглаженная."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_cfg = data.shape[0]
    for ic in range(n_cfg):
        cfg = CONFIGS[ic]
        ax.plot(phi, data[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=0.5, alpha=0.2)
        smooth = uniform_filter1d(data[ic], size=SMOOTH_WINDOW)
        ax.plot(phi, smooth, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1.8, label=cfg["label"])
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(phi[0], phi[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Загрузить data.npz и перестроить графики без расчёта")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Путь к папке с data.npz для --plot-only")
    args = parser.parse_args()

    DEBUG = False  # production 850 кН

    print("=" * 60)
    print("НЕСТАЦИОНАРНЫЙ РАСЧЁТ ПОДШИПНИКА ДВС")
    print("(Reynolds со squeeze + уравнение движения вала)")
    print("=" * 60)

    if args.plot_only:
        results_dir = args.data_dir or os.path.join(
            os.path.dirname(__file__), "..", "results", "diesel")
        data_path = os.path.join(results_dir, "data.npz")
        print(f"Режим --plot-only: загрузка {data_path}...")
        d = np.load(data_path, allow_pickle=True)
        results = {k: d[k] for k in d.files}
        dt_calc = 0.0
        cfg_times = [0.0] * len(CONFIGS)
    else:
        results_dir = make_results_dir(debug=DEBUG)
        print(f"  Результаты → {results_dir}")
        t0 = time.time()
        results = run_transient(debug=DEBUG)
        dt_calc = time.time() - t0
        cfg_times = results.get("cfg_times", [0.0] * len(CONFIGS))
        print(f"\nВремя расчёта: {dt_calc:.1f} с")

    # Индексы последнего цикла
    if "last_start" in results:
        s = int(results["last_start"])
        nc = int(results["n_steps_per_cycle"])
        phi = results["phi_last"]
    else:
        phi_all = results["phi_crank_deg"]
        nc = int(round(720.0 / params.d_phi_crank_deg))
        n_total = len(phi_all)
        n_cyc = max(1, n_total // nc)
        s = nc * (n_cyc - 1)
        phi = phi_all[s:s+nc]

    eps_x = results["eps_x"][:, s:s+nc]
    eps_y = results["eps_y"][:, s:s+nc]
    eps_mag = np.sqrt(eps_x**2 + eps_y**2)
    hmin = results["hmin"][:, s:s+nc]
    pmax = results["pmax"][:, s:s+nc]
    f_arr = results["f"][:, s:s+nc]
    F_tr = results["F_tr"][:, s:s+nc]
    N_loss = results["N_loss"][:, s:s+nc]
    Fy_ext = results["Fy_ext_last"]

    # 1. Внешняя нагрузка
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(phi, -Fy_ext / 1000, "k-", linewidth=2)
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel("Fy_ext (кН, вниз = +)", fontsize=12)
    F_max_val = float(results["F_max"])
    ax.set_title(f"Вертикальная нагрузка Fy_ext(φ) — ДВС "
                 f"(F_max={F_max_val/1e3:.0f} кН, Вибе+КШМ)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(phi[0], phi[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "F_ext_vs_phi.png"), dpi=150)
    plt.close(fig)
    print("  Сохранён: F_ext_vs_phi.png")

    # 2-5. Основные графики
    plot_curves(phi, eps_mag, "Эксцентриситет |ε|",
                "eps_vs_phi.png", "Эксцентриситет |ε|(φ) — ДВС", results_dir)
    plot_curves(phi, hmin * 1e6, "h_min (мкм)",
                "hmin_vs_phi.png", "Минимальный зазор h_min(φ) — ДВС", results_dir)
    plot_curves(phi, f_arr, "Коэффициент трения f",
                "f_vs_phi.png", "Коэффициент трения f(φ) — ДВС", results_dir)
    plot_curves(phi, pmax / 1e6, "p_max (МПа)",
                "pmax_vs_phi.png", "Максимальное давление p_max(φ) — ДВС", results_dir)

    # 6. Орбита
    fig, ax = plt.subplots(figsize=(7, 7))
    n_cfg_actual = eps_x.shape[0]
    kern = max(1, nc // 144)
    for ic in range(n_cfg_actual):
        cfg = CONFIGS[ic]
        ax.plot(eps_x[ic], eps_y[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=0.4, alpha=0.3)
        if kern > 1:
            kernel = np.ones(kern) / kern
            sx = np.convolve(eps_x[ic], kernel, mode="same")
            sy = np.convolve(eps_y[ic], kernel, mode="same")
        else:
            sx, sy = eps_x[ic], eps_y[ic]
        ax.plot(sx, sy, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1.8, label=cfg["label"])
        ax.plot(eps_x[ic, 0], eps_y[ic, 0], "o", color=cfg["color"], markersize=5)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.3,
            label="Граница зазора |ε|=1")
    ax.set_xlabel("εx", fontsize=12)
    ax.set_ylabel("εy", fontsize=12)
    ax.set_title("Орбита вала — последний цикл", fontsize=13)
    ax.set_aspect("equal")
    all_ex = eps_x[:n_cfg_actual].ravel()
    all_ey = eps_y[:n_cfg_actual].ravel()
    valid = np.isfinite(all_ex) & np.isfinite(all_ey)
    if np.any(valid):
        margin = 0.1
        xmin, xmax = all_ex[valid].min(), all_ex[valid].max()
        ymin, ymax = all_ey[valid].min(), all_ey[valid].max()
        dx = max(xmax - xmin, 0.1)
        dy = max(ymax - ymin, 0.1)
        ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
        ax.set_ylim(ymin - margin * dy, ymax + margin * dy)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "orbit.png"), dpi=150)
    plt.close(fig)
    print("  Сохранён: orbit.png")

    # Замкнутость орбиты
    print("\n  Замкнутость орбиты (последний цикл):")
    for ic in range(n_cfg_actual):
        cfg = CONFIGS[ic]
        dx = eps_x[ic, -1] - eps_x[ic, 0]
        dy = eps_y[ic, -1] - eps_y[ic, 0]
        gap = np.sqrt(dx**2 + dy**2)
        status = "OK" if gap < 0.05 else "рассогласование"
        print(f"    {cfg['label']}: Δε = {gap:.4f} ({status})")

    # Summary
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Нестационарный расчёт подшипника ДВС\n")
        f.write(f"F_max = {F_max_val/1e3:.0f} кН (Вибе+КШМ)\n")
        f.write(f"Время расчёта: {dt_calc:.1f} с\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"{'Конфигурация':<30} {'min(h_min)':>10} {'max(p_max)':>10} "
                f"{'mean(f)':>10} {'max(|ε|)':>10} {'mean(N_loss)':>12} "
                f"{'clamp':>6} {'время,с':>8}\n")
        f.write(f"{'':30} {'мкм':>10} {'МПа':>10} {'':>10} {'':>10} "
                f"{'Вт':>12} {'':>6} {'':>8}\n")
        f.write("-" * 110 + "\n")
        for ic in range(n_cfg_actual):
            cfg = CONFIGS[ic]
            hmin_min = np.nanmin(hmin[ic]) * 1e6
            pmax_max = np.nanmax(pmax[ic]) / 1e6
            f_mean = np.nanmean(f_arr[ic])
            eps_max_val = np.nanmax(eps_mag[ic])
            nloss_mean = np.nanmean(N_loss[ic])
            eps_all = np.sqrt(results["eps_x"][ic]**2 + results["eps_y"][ic]**2)
            clamp_count = int(np.sum(eps_all >= params.eps_max - 0.001))
            t_cfg = cfg_times[ic] if ic < len(cfg_times) else 0.0
            f.write(f"{cfg['label']:<30} {hmin_min:>10.1f} {pmax_max:>10.1f} "
                    f"{f_mean:>10.4f} {eps_max_val:>10.3f} {nloss_mean:>12.1f} "
                    f"{clamp_count:>6d} {t_cfg:>8.1f}\n")
        f.write("\n")
        f.write("Замкнутость орбиты (последний цикл):\n")
        for ic in range(n_cfg_actual):
            cfg = CONFIGS[ic]
            dx = eps_x[ic, -1] - eps_x[ic, 0]
            dy = eps_y[ic, -1] - eps_y[ic, 0]
            gap = np.sqrt(dx**2 + dy**2)
            f.write(f"  {cfg['label']}: Δε = {gap:.4f}\n")
    print(f"\n  Сохранён: summary.txt")

    # Сводка
    print(f"\n{'='*60}")
    print(f"СВОДКА (последний цикл)")
    print(f"{'='*60}")
    for ic in range(n_cfg_actual):
        cfg = CONFIGS[ic]
        print(f"  {cfg['label']}:")
        print(f"    min(h_min) = {np.nanmin(hmin[ic])*1e6:.1f} мкм")
        print(f"    max(p_max) = {np.nanmax(pmax[ic])/1e6:.1f} МПа")
        print(f"    mean(f)    = {np.nanmean(f_arr[ic]):.4f}")
        print(f"    max(|ε|)   = {np.nanmax(eps_mag[ic]):.3f}")
        print(f"    mean(N_loss) = {np.nanmean(N_loss[ic]):.1f} Вт")

    # data.npz
    np.savez(
        os.path.join(results_dir, "data.npz"),
        phi_crank_deg=results["phi_crank_deg"],
        phi_last=phi,
        eps_x=results["eps_x"],
        eps_y=results["eps_y"],
        hmin=results["hmin"],
        pmax=results["pmax"],
        f=results["f"],
        F_tr=results["F_tr"],
        N_loss=results["N_loss"],
        Fx_hyd=results["Fx_hyd"],
        Fy_hyd=results["Fy_hyd"],
        Fy_ext_last=Fy_ext,
        F_max=F_max_val,
        labels=[c["label"] for c in CONFIGS],
    )
    print(f"\nДанные сохранены в {results_dir}/data.npz")


if __name__ == "__main__":
    main()
