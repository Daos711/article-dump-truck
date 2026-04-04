#!/usr/bin/env python3
"""Запуск квазистационарного расчёта подшипника ДВС.

Генерирует графики ε(φ), h_min(φ), f(φ), p_max(φ), F_ext(φ),
accuracy_check для 4 конфигураций и сохраняет данные в results/diesel/.

Методологическая оговорка: используется квазистационарная аппроксимация.
Squeeze-эффект (∂h/∂t), инерция вала и полная связка Reynolds + equation
of motion НЕ учитываются.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.diesel_quasistatic import (
    run_diesel_analysis, load_diesel, CONFIGS, PHI_CRANK,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "diesel")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_curves(phi, data, ylabel, filename, title):
    """Построить график с 4 кривыми по углу коленвала."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for ic, cfg in enumerate(CONFIGS):
        ax.plot(phi, data[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1.5, label=cfg["label"])
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 720)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {filename}")


def main():
    print("=" * 60)
    print("КВАЗИСТАЦИОНАРНЫЙ РАСЧЁТ ПОДШИПНИКА ДВС")
    print("(квазистационарная аппроксимация, без squeeze-эффекта)")
    print("=" * 60)

    t0 = time.time()
    results = run_diesel_analysis()
    dt = time.time() - t0
    print(f"\nВремя расчёта: {dt:.1f} с")

    phi = results["phi_crank"]
    F_ext = results["F_ext"]

    # 1. Внешняя нагрузка
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(phi, F_ext / 1000, "k-", linewidth=2)
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel("F_ext (кН)", fontsize=12)
    ax.set_title("Внешняя нагрузка F(φ) — ДВС", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 720)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "F_ext_vs_phi.png"), dpi=150)
    plt.close(fig)
    print("  Сохранён: F_ext_vs_phi.png")

    # 2-5. Кривые по конфигурациям
    plot_curves(phi, results["epsilon"], "Эксцентриситет ε",
                "eps_vs_phi.png", "Эксцентриситет ε(φ) — ДВС")
    plot_curves(phi, results["hmin"] * 1e6, "h_min (мкм)",
                "hmin_vs_phi.png", "Минимальный зазор h_min(φ) — ДВС")
    plot_curves(phi, results["f"], "Коэффициент трения f",
                "f_vs_phi.png", "Коэффициент трения f(φ) — ДВС")
    plot_curves(phi, results["pmax"] / 1e6, "p_max (МПа)",
                "pmax_vs_phi.png", "Максимальное давление p_max(φ) — ДВС")

    # 6. Проверка точности бисекции
    rel_err = np.abs(results["F_hyd"] - F_ext[np.newaxis, :]) / F_ext[np.newaxis, :] * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    for ic, cfg in enumerate(CONFIGS):
        ax.plot(phi, rel_err[ic], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=1, label=cfg["label"], alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Порог 1%")
    ax.set_xlabel("Угол коленвала φ (°)", fontsize=12)
    ax.set_ylabel("|W_hyd − F_ext| / F_ext (%)", fontsize=12)
    ax.set_title("Проверка точности бисекции — ДВС", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 720)
    ax.set_ylim(0, max(5, np.max(rel_err) * 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_check.png"), dpi=150)
    plt.close(fig)
    print("  Сохранён: accuracy_check.png")

    max_err = np.max(rel_err)
    print(f"\n  Макс. ошибка бисекции: {max_err:.2f}%"
          f" {'✓ OK' if max_err < 1 else '⚠ >1%!'}")

    # Сохранить данные
    np.savez(
        os.path.join(RESULTS_DIR, "data.npz"),
        phi_crank=phi, F_ext=F_ext,
        epsilon=results["epsilon"],
        hmin=results["hmin"], f=results["f"],
        pmax=results["pmax"], F_hyd=results["F_hyd"],
        labels=[c["label"] for c in CONFIGS],
    )
    print(f"\nДанные сохранены в results/diesel/data.npz")


if __name__ == "__main__":
    main()
