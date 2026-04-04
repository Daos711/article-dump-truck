#!/usr/bin/env python3
"""Запуск стационарного расчёта подшипника центробежного насоса.

Генерирует графики W(ε), f(ε), h_min(ε), Q(ε) для 4 конфигураций
и сохраняет данные в results/pump/.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pump_steady import run_pump_analysis, CONFIGS, EPSILON_VALUES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pump")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_curves(eps, data, ylabel, filename, title):
    """Построить график с 4 кривыми."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ic, cfg in enumerate(CONFIGS):
        ax.plot(eps, data[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=2, label=cfg["label"])
    ax.set_xlabel("Эксцентриситет ε", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {filename}")


def main():
    print("=" * 60)
    print("РАСЧЁТ ПОДШИПНИКА ЦЕНТРОБЕЖНОГО НАСОСА")
    print("=" * 60)

    t0 = time.time()
    results = run_pump_analysis()
    dt = time.time() - t0
    print(f"\nВремя расчёта: {dt:.1f} с")

    eps = results["epsilon"]

    plot_curves(eps, results["W"], "Несущая способность W (Н)",
                "W_vs_eps.png", "Нагрузочная способность W(ε) — насос")
    plot_curves(eps, results["f"], "Коэффициент трения f",
                "f_vs_eps.png", "Коэффициент трения f(ε) — насос")
    plot_curves(eps, results["hmin"] * 1e6, "Минимальный зазор h_min (мкм)",
                "hmin_vs_eps.png", "Минимальный зазор h_min(ε) — насос")
    plot_curves(eps, results["Q"] * 1e6, "Расход смазки Q (см³/с)",
                "Q_vs_eps.png", "Расход смазки Q(ε) — насос")

    # Сохранить данные
    np.savez(
        os.path.join(RESULTS_DIR, "data.npz"),
        epsilon=eps,
        W=results["W"], f=results["f"],
        hmin=results["hmin"], Q=results["Q"],
        labels=[c["label"] for c in CONFIGS],
    )
    print(f"\nДанные сохранены в results/pump/data.npz")


if __name__ == "__main__":
    main()
