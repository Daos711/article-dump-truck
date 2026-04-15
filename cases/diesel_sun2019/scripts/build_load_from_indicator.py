#!/usr/bin/env python3
"""Построение surrogate bearing load из индикаторной диаграммы.

Создаёт CSV файлы нагрузки для run_smooth_cycle.py:
  derived/load_main_bearing_<n_rpm>rpm_<load_pct>pct.csv     (размерная)
  derived/load_main_bearing_<n_rpm>rpm_<load_pct>pct_nd.csv  (безразмерная)

Surrogate-модель:
  F_gas(α) — gas pressure pulse (Gaussian peak at TDC firing + compression
             bump at 360° CA)
  F_inertia — центробежная (вращ. масса шатунно-поршневой группы)
  Проекция на оси подшипника через кинематику КШМ (первое приближение).

ВНИМАНИЕ: это surrogate, не FE-модель коленвала. Не претендует на точное
совпадение с Sun 2019 Figure X (indicator). Форма нагрузки реалистична
по порядку величины и по фазе (пик при TDC firing).
"""
import sys
import os
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import case_config as cfg
from scaling import (omega_from_rpm, force_scale, crank_deg_to_tau,
                      nondim_force)


def build_surrogate_load(n_rpm, load_pct,
                          n_cyl, bore_m, stroke_m, con_rod_m,
                          m_piston_kg, p_max_MPa, p_motoring_MPa,
                          n_points=720):
    """Surrogate bearing load для 4-тактного дизеля.

    Возвращает crank_deg ∈ [0, 720), WaX_N, WaY_N.
    """
    crank_deg = np.linspace(0, 720, n_points, endpoint=False)
    crank_rad = np.deg2rad(crank_deg)

    # Масштабирование пикового давления по load_pct
    p_peak = p_motoring_MPa + (p_max_MPa - p_motoring_MPa) * (load_pct / 100.0)

    # Gas pressure: Gaussian peak at TDC firing (0° CA)
    p_gas = p_motoring_MPa + (p_peak - p_motoring_MPa) * np.exp(
        -0.5 * ((crank_deg - 5.0) / 15.0) ** 2)
    # Compression secondary bump at ~355° CA
    p_gas += 0.3 * (p_peak - p_motoring_MPa) * np.exp(
        -0.5 * ((crank_deg - 355.0) / 20.0) ** 2)

    A_piston = np.pi * (bore_m / 2) ** 2
    F_gas = p_gas * 1e6 * A_piston  # Н (скалярная сила давления газа)

    # Проекция на оси через кинематику КШМ (упрощённо)
    lambda_ratio = (stroke_m / 2) / con_rod_m
    WaX = -F_gas * np.cos(crank_rad) * (1 + lambda_ratio * np.cos(crank_rad))
    WaY = -F_gas * np.sin(crank_rad) * (1 + 0.5 * lambda_ratio)

    # Центробежная сила (вращающиеся массы шатунно-поршневой группы)
    omega = omega_from_rpm(n_rpm)
    r_crank = stroke_m / 2
    F_inertia = m_piston_kg * r_crank * omega ** 2
    WaX += F_inertia * np.cos(crank_rad)
    WaY += F_inertia * np.sin(crank_rad)

    # Распределение на один main bearing: ~40% нагрузки с цилиндра
    # (остальное на соседние опоры)
    BEARING_SHARE = 0.4
    WaX *= BEARING_SHARE
    WaY *= BEARING_SHARE

    return crank_deg, WaX, WaY


def save_csv_dim(path, crank_deg, WaX_N, WaY_N):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crank_deg", "WaX_dim_N", "WaY_dim_N"])
        for c_deg, fx, fy in zip(crank_deg, WaX_N, WaY_N):
            w.writerow([f"{c_deg:.3f}", f"{fx:.3f}", f"{fy:.3f}"])


def save_csv_nd(path, tau, WaX_nd, WaY_nd):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau", "WaX_nd", "WaY_nd"])
        for t, fx, fy in zip(tau, WaX_nd, WaY_nd):
            w.writerow([f"{t:.6f}", f"{fx:.6e}", f"{fy:.6e}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rpm", type=int, default=cfg.n_rpm)
    parser.add_argument("--load-pct", type=int, default=cfg.load_pct)
    parser.add_argument("--n-points", type=int, default=720,
                        help="точек на цикл 720° (default 720 = 1° resolution)")
    parser.add_argument("--plot", action="store_true",
                        help="построить PNG с кривыми нагрузки")
    args = parser.parse_args()

    print("=" * 65)
    print("SURROGATE BEARING LOAD (Sun 2019 diesel main bearing)")
    print(f"n_rpm={args.n_rpm}, load_pct={args.load_pct}")
    print(f"bore={cfg.bore_m*1e3:.1f}мм, stroke={cfg.stroke_m*1e3:.1f}мм, "
          f"con_rod={cfg.con_rod_m*1e3:.1f}мм")
    print(f"p_max={cfg.p_max_MPa} МПа, m_piston={cfg.m_piston_kg} кг")
    print("=" * 65)

    crank_deg, WaX, WaY = build_surrogate_load(
        n_rpm=args.n_rpm, load_pct=args.load_pct,
        n_cyl=cfg.n_cylinders, bore_m=cfg.bore_m,
        stroke_m=cfg.stroke_m, con_rod_m=cfg.con_rod_m,
        m_piston_kg=cfg.m_piston_kg,
        p_max_MPa=cfg.p_max_MPa, p_motoring_MPa=cfg.p_motoring_MPa,
        n_points=args.n_points,
    )

    F_mag = np.sqrt(WaX**2 + WaY**2)
    print(f"  F_max = {F_mag.max():.0f} Н (at crank_deg={crank_deg[F_mag.argmax()]:.0f}°)")
    print(f"  F_min = {F_mag.min():.0f} Н")
    print(f"  WaX range: [{WaX.min():.0f}, {WaX.max():.0f}] Н")
    print(f"  WaY range: [{WaY.min():.0f}, {WaY.max():.0f}] Н")

    # Размерный CSV
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "derived")
    os.makedirs(out_dir, exist_ok=True)
    base = f"load_main_bearing_{args.n_rpm}rpm_{args.load_pct}pct"
    dim_path = os.path.join(out_dir, base + ".csv")
    save_csv_dim(dim_path, crank_deg, WaX, WaY)
    print(f"\n  Размерный CSV: {dim_path}")

    # Безразмерный CSV
    omega = omega_from_rpm(args.n_rpm)
    F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
    WaX_nd = WaX / F0
    WaY_nd = WaY / F0
    tau = crank_deg_to_tau(crank_deg)
    nd_path = os.path.join(out_dir, base + "_nd.csv")
    save_csv_nd(nd_path, tau, WaX_nd, WaY_nd)
    print(f"  Безразмерный CSV: {nd_path}")
    print(f"  F₀ = {F0:.0f} Н")
    print(f"  |WaX_nd|_max = {np.abs(WaX_nd).max():.4f}")
    print(f"  |WaY_nd|_max = {np.abs(WaY_nd).max():.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(crank_deg, WaX / 1000, "b-", lw=1.2, label="WaX")
        axes[0].plot(crank_deg, WaY / 1000, "r-", lw=1.2, label="WaY")
        axes[0].set_ylabel("F, кН")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(crank_deg, F_mag / 1000, "k-", lw=1.5)
        axes[1].set_ylabel("|F|, кН")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(crank_deg, WaX_nd, "b-", lw=1.2, label="WaX_nd")
        axes[2].plot(crank_deg, WaY_nd, "r-", lw=1.2, label="WaY_nd")
        axes[2].set_xlabel("crank_deg")
        axes[2].set_ylabel("F_nd")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, base + "_plot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  График: {plot_path}")


if __name__ == "__main__":
    main()
