#!/usr/bin/env python3
"""Запуск нестационарного расчёта подшипника ДВС.

Time-stepping: совместное решение Reynolds со squeeze + уравнение движения вала.
Генерирует графики ε(φ), h_min(φ), f(φ), p_max(φ), Fy_ext(φ), орбиту вала,
summary.txt и data.npz для 4 конфигураций.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.diesel_transient import run_transient, load_diesel, CONFIGS
from config import diesel_params as params

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "diesel")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_curves(phi, data, ylabel, filename, title):
    """Построить график с 4 кривыми по углу коленвала (последний цикл)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for ic, cfg in enumerate(CONFIGS):
        ax.plot(phi, data[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1.5, label=cfg["label"])
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(phi[0], phi[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {filename}")


def main():
    print("=" * 60)
    print("НЕСТАЦИОНАРНЫЙ РАСЧЁТ ПОДШИПНИКА ДВС")
    print("(Reynolds со squeeze + уравнение движения вала)")
    print("=" * 60)

    t0 = time.time()
    results = run_transient(debug=True)
    dt_calc = time.time() - t0
    print(f"\nВремя расчёта: {dt_calc:.1f} с")

    # Индексы последнего цикла
    s = results["last_start"]
    nc = results["n_steps_per_cycle"]
    phi = results["phi_last"]

    eps_x = results["eps_x"][:, s:s+nc]
    eps_y = results["eps_y"][:, s:s+nc]
    eps_mag = np.sqrt(eps_x**2 + eps_y**2)
    hmin = results["hmin"][:, s:s+nc]
    pmax = results["pmax"][:, s:s+nc]
    f_arr = results["f"][:, s:s+nc]
    F_tr = results["F_tr"][:, s:s+nc]
    N_loss = results["N_loss"][:, s:s+nc]
    Fy_ext = results["Fy_ext_last"]

    # 1. Внешняя нагрузка Fy_ext(φ)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(phi, -Fy_ext / 1000, "k-", linewidth=2)
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel("Fy_ext (кН, вниз = +)", fontsize=12)
    ax.set_title(f"Вертикальная нагрузка Fy_ext(φ) — ДВС "
                 f"(F_max={results['F_max']/1e3:.0f} кН, surrogate)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(phi[0], phi[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "F_ext_vs_phi.png"), dpi=150)
    plt.close(fig)
    print("  Сохранён: F_ext_vs_phi.png")

    # 2-5. Основные графики
    plot_curves(phi, eps_mag, "Эксцентриситет |ε|",
                "eps_vs_phi.png", "Эксцентриситет |ε|(φ) — ДВС")
    plot_curves(phi, hmin * 1e6, "h_min (мкм)",
                "hmin_vs_phi.png", "Минимальный зазор h_min(φ) — ДВС")
    plot_curves(phi, f_arr, "Коэффициент трения f",
                "f_vs_phi.png", "Коэффициент трения f(φ) — ДВС")
    plot_curves(phi, pmax / 1e6, "p_max (МПа)",
                "pmax_vs_phi.png", "Максимальное давление p_max(φ) — ДВС")

    # 6. Орбита вала (последний цикл)
    fig, ax = plt.subplots(figsize=(7, 7))
    for ic, cfg in enumerate(CONFIGS):
        ax.plot(eps_x[ic], eps_y[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1.2, label=cfg["label"])
        ax.plot(eps_x[ic, 0], eps_y[ic, 0], "o", color=cfg["color"], markersize=5)
    # Окружность зазора
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.3,
            label="Граница зазора |ε|=1")
    ax.set_xlabel("εx", fontsize=12)
    ax.set_ylabel("εy", fontsize=12)
    ax.set_title("Орбита вала — последний цикл", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "orbit.png"), dpi=150)
    plt.close(fig)
    print("  Сохранён: orbit.png")

    # Проверка замкнутости орбиты
    print("\n  Замкнутость орбиты (последний цикл):")
    for ic, cfg in enumerate(CONFIGS):
        dx = eps_x[ic, -1] - eps_x[ic, 0]
        dy = eps_y[ic, -1] - eps_y[ic, 0]
        gap = np.sqrt(dx**2 + dy**2)
        status = "OK" if gap < 0.05 else "рассогласование"
        print(f"    {cfg['label']}: Δε = {gap:.4f} ({status})")

    # Summary
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Нестационарный расчёт подшипника ДВС\n")
        f.write(f"F_max = {results['F_max']/1e3:.0f} кН (surrogate)\n")
        f.write(f"Время расчёта: {dt_calc:.1f} с\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Конфигурация':<30} {'min(h_min)':>10} {'max(p_max)':>10} "
                f"{'mean(f)':>10} {'max(|ε|)':>10} {'mean(N_loss)':>12} "
                f"{'clamp':>6}\n")
        f.write(f"{'':30} {'мкм':>10} {'МПа':>10} {'':>10} {'':>10} "
                f"{'Вт':>12} {'':>6}\n")
        f.write("-" * 100 + "\n")
        for ic, cfg in enumerate(CONFIGS):
            hmin_min = np.min(hmin[ic]) * 1e6
            pmax_max = np.max(pmax[ic]) / 1e6
            f_mean = np.mean(f_arr[ic])
            eps_max_val = np.max(eps_mag[ic])
            nloss_mean = np.mean(N_loss[ic])
            # Считаем clamp по всем шагам (не только последний цикл)
            eps_all = np.sqrt(results["eps_x"][ic]**2 + results["eps_y"][ic]**2)
            clamp_count = np.sum(eps_all >= params.eps_max - 0.001)
            f.write(f"{cfg['label']:<30} {hmin_min:>10.1f} {pmax_max:>10.1f} "
                    f"{f_mean:>10.4f} {eps_max_val:>10.3f} {nloss_mean:>12.1f} "
                    f"{clamp_count:>6d}\n")
        f.write("\n")
        # Замкнутость
        f.write("Замкнутость орбиты (последний цикл):\n")
        for ic, cfg in enumerate(CONFIGS):
            dx = eps_x[ic, -1] - eps_x[ic, 0]
            dy = eps_y[ic, -1] - eps_y[ic, 0]
            gap = np.sqrt(dx**2 + dy**2)
            f.write(f"  {cfg['label']}: Δε = {gap:.4f}\n")
    print(f"\n  Сохранён: summary.txt")

    # Печать сводки
    print(f"\n{'='*60}")
    print(f"СВОДКА (последний цикл)")
    print(f"{'='*60}")
    for ic, cfg in enumerate(CONFIGS):
        print(f"  {cfg['label']}:")
        print(f"    min(h_min) = {np.min(hmin[ic])*1e6:.1f} мкм")
        print(f"    max(p_max) = {np.max(pmax[ic])/1e6:.1f} МПа")
        print(f"    mean(f)    = {np.mean(f_arr[ic]):.4f}")
        print(f"    max(|ε|)   = {np.max(eps_mag[ic]):.3f}")
        print(f"    mean(N_loss) = {np.mean(N_loss[ic]):.1f} Вт")

    # data.npz
    np.savez(
        os.path.join(RESULTS_DIR, "data.npz"),
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
        F_max=results["F_max"],
        labels=[c["label"] for c in CONFIGS],
    )
    print(f"\nДанные сохранены в results/diesel/data.npz")


if __name__ == "__main__":
    main()
