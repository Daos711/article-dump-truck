#!/usr/bin/env python3
"""Нестационарный расчёт подшипника насоса с гармонической нагрузкой.

Используется Ausas transient GPU-solver. Сравнение smooth vs textured
(baseline из pump_params.py). Возбуждение на лопаточной частоте
(6 лопаток × 3000 об/мин = 300 Гц).

Coordinates (Ausas):
  φ ∈ [0, 1], Z ∈ [0, B], B = L/(2R)
  H = 1 + X·cos(2π·φ) + Y·sin(2π·φ) + texture_relief(φ,Z)
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reynolds_solver import solve_ausas_journal_dynamic_gpu
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Фиксированные параметры ─────────────────────────────────────
R = params.R                       # 0.035 м
L = params.L                       # 0.056 м
C = params.c                       # 50 мкм
N_RPM = params.n                   # 3000 об/мин
ETA = MINERAL_OIL["eta_pump"]      # 0.022 Па·с
OMEGA = 2 * np.pi * N_RPM / 60     # 314.16 рад/с
P_SCALE = 6 * ETA * OMEGA * (R / C) ** 2

B_AUSAS = L / (2 * R)              # = 0.8

# ─── Сетка ────────────────────────────────────────────────────────
N1 = 200       # физические ячейки по φ (interior)
N2 = 20        # физические ячейки по Z
N_PHI = N1 + 2  # с ghost
N_Z = N2 + 2
D_PHI = 1.0 / N1
D_Z = B_AUSAS / N2

# Solver-geometry: чтобы alpha_sq = (2R/L · dφ/dZ)² = q²
# API Ausas: alpha_sq = (2R_s/L_s · dφ/dZ)². Подбираем R_s/L_s.
# При B = L/(2R): q² = (dφ/dZ)² × (2R/L)² = (1/N1)² / (B/N2)² × 1
# Мы можем просто выбрать R_s=0.5, L_s=1.0 → 2R_s/L_s=1.0
R_SOLVER = 0.5
L_SOLVER = 1.0

# ─── Вибрация ─────────────────────────────────────────────────────
N_BLADES = 6
F_VIB = N_BLADES * N_RPM / 60.0    # Гц = 300
OMEGA_VIB = 2 * np.pi * F_VIB

# Безразмерный период вибрации
# Ausas-время нормировано ω_shaft: t_ausas = t_phys · ω_shaft
# → T_vib_ausas = 2π/ω_vib · ω_shaft = ω_shaft/ω_vib = 1/N_BLADES
T_VIB = 1.0 / N_BLADES

# ─── Начальные условия и нагрузка ────────────────────────────────
EPS_STATIC = 0.3
X0 = EPS_STATIC
Y0 = 0.0

W_STATIC_X = -0.005
W_STATIC_Y = 0.0
DELTA_W = 0.1 * abs(W_STATIC_X)


def load_fn(t):
    """Безразмерная нагрузка: постоянная + 10% гармоника."""
    WaX = W_STATIC_X + DELTA_W * np.sin(2 * np.pi * t / T_VIB)
    WaY = W_STATIC_Y
    return WaX, WaY


# ─── Параметры расчёта ───────────────────────────────────────────
MASS_M = 1e-4
DT = 1e-3
N_PERIODS = 2
NT = int(N_PERIODS * T_VIB / DT)


def build_texture_relief():
    """Построить массив texture_relief = H_textured - H_smooth.

    Центры лунок из pump_params.py (зона phi_start..phi_end, N×N).
    Координаты Ausas: φ ∈ [0, 1], Z ∈ [0, B].
    Ghost: массив shape (N_Z, N_phi) = (N2+2, N1+2).
    """
    # Безразмерные полуоси
    A_Z = (params.a_dim / R) / B_AUSAS   # вдоль Z (в Ausas Z-координате)
    B_phi = (params.b_dim / R) / (2 * np.pi)  # вдоль φ в [0,1] координате

    H_p = params.h_p / C

    # Сетка Ausas (interior + ghost, точки в центрах ячеек)
    # Используем φ_arr ∈ [0, 1) с endpoint=False эквивалент,
    # но с ghost по краям. Здесь для relief просто центры ячеек.
    phi_arr = np.linspace(-D_PHI / 2, 1 + D_PHI / 2, N_PHI)
    Z_arr = np.linspace(-D_Z / 2, B_AUSAS + D_Z / 2, N_Z)
    Phi, Zm = np.meshgrid(phi_arr, Z_arr)

    # Центры лунок в φ ∈ [0, 1]
    phi_start = params.phi_start_deg / 360.0
    phi_end = params.phi_end_deg / 360.0
    phi_centers = np.linspace(phi_start, phi_end, params.N_phi_tex)
    # Z-центры равномерно в [0.1·B, 0.9·B]
    Z_centers = np.linspace(0.1 * B_AUSAS, 0.9 * B_AUSAS, params.N_Z_tex)
    phi_c, Z_c = np.meshgrid(phi_centers, Z_centers)
    phi_c = phi_c.ravel()
    Z_c = Z_c.ravel()

    # Относительная глубина через эллипсы (convenient: reuse utils)
    # create_H_with_ellipsoidal_depressions добавляет к H0 глубину H_p внутри
    # эллипса с полуосями (B_rad_phi, A_Z_nondim).
    # Здесь H0=0 → получим relief = только текстура.
    H0 = np.zeros_like(Phi)
    # В utils полуоси в тех же координатах что Phi и Zm
    # B_phi_util: полуось по phi (в координатах [0,1]), A_Z_util: полуось Z
    try:
        relief = create_H_with_ellipsoidal_depressions(
            H0, H_p, Phi, Zm, phi_c, Z_c, A_Z, B_phi,
            profile="smoothcap")
    except Exception as e:
        print(f"  WARN: create_H_with_ellipsoidal_depressions: {e}")
        # Fallback: ручное построение
        relief = np.zeros_like(Phi)
        for p_c, z_c in zip(phi_c, Z_c):
            dphi = Phi - p_c
            # periodic wrap
            dphi = (dphi + 0.5) % 1.0 - 0.5
            dz = Zm - z_c
            r2 = (dphi / B_phi)**2 + (dz / A_Z)**2
            mask = r2 < 1
            relief[mask] += H_p * (1 - r2[mask])**2

    return relief


def run_transient(texture_relief, label, out_dir):
    """Один прогон нестационарный."""
    print(f"\n{'=' * 70}")
    print(f"Transient: {label}")
    print(f"NT={NT}, dt={DT}, T_vib={T_VIB:.4f}, {N_PERIODS} периодов")
    print(f"{'=' * 70}")

    t0 = time.time()
    result = solve_ausas_journal_dynamic_gpu(
        NT=NT, dt=DT,
        N_Z=N_Z, N_phi=N_PHI,
        d_phi=D_PHI, d_Z=D_Z,
        R=R_SOLVER, L=L_SOLVER,
        mass_M=MASS_M,
        load_fn=load_fn,
        X0=X0, Y0=Y0,
        texture_relief=texture_relief,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
        scheme="rb", verbose=False,
    )
    dt_run = time.time() - t0
    print(f"  Время: {dt_run:.1f} с")
    print(f"  Итоговые: X={result.X[-1]:.4f}, Y={result.Y[-1]:.4f}")
    print(f"  h_min range: {np.min(result.h_min):.4f}..{np.max(result.h_min):.4f}")
    print(f"  p_max range: {np.min(result.p_max):.4f}..{np.max(result.p_max):.4f}")

    # Сохранить npz
    npz_path = os.path.join(out_dir, f"{label}.npz")
    np.savez(npz_path,
             t=result.t, X=result.X, Y=result.Y,
             WX=result.WX, WY=result.WY,
             p_max=result.p_max, h_min=result.h_min,
             cav_frac=result.cav_frac)
    print(f"  {npz_path}")
    return result


def plot_comparison(r_smooth, r_tex, out_dir):
    """Графики smooth vs textured."""
    t = r_smooth.t
    e_s = np.sqrt(r_smooth.X**2 + r_smooth.Y**2)
    e_t = np.sqrt(r_tex.X**2 + r_tex.Y**2)

    # h_min
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r_smooth.h_min, "b-", lw=1.2, label="Smooth")
    ax.plot(t, r_tex.h_min, "r-", lw=1.2, label="Textured")
    ax.set_xlabel("t (безразмерное)")
    ax.set_ylabel("h_min (безразмерное)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h_min_vs_t.png"), dpi=200)
    plt.close(fig)

    # p_max
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r_smooth.p_max, "b-", lw=1.2, label="Smooth")
    ax.plot(t, r_tex.p_max, "r-", lw=1.2, label="Textured")
    ax.set_xlabel("t (безразмерное)")
    ax.set_ylabel("p_max (безразмерное)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "p_max_vs_t.png"), dpi=200)
    plt.close(fig)

    # эксцентриситет
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, e_s, "b-", lw=1.2, label="Smooth")
    ax.plot(t, e_t, "r-", lw=1.2, label="Textured")
    ax.set_xlabel("t (безразмерное)")
    ax.set_ylabel("ε = √(X²+Y²)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e_vs_t.png"), dpi=200)
    plt.close(fig)

    # cav_frac
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r_smooth.cav_frac, "b-", lw=1.2, label="Smooth")
    ax.plot(t, r_tex.cav_frac, "r-", lw=1.2, label="Textured")
    ax.set_xlabel("t (безразмерное)")
    ax.set_ylabel("cav_frac")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cav_frac_vs_t.png"), dpi=200)
    plt.close(fig)

    print(f"\n  Графики: h_min_vs_t, p_max_vs_t, e_vs_t, cav_frac_vs_t")


def compute_metrics(r_smooth, r_tex):
    """Метрики по последнему периоду."""
    # Индексы последнего периода
    t = r_smooth.t
    t_end = t[-1]
    mask_last = t >= (t_end - T_VIB)

    def stat(arr, mask):
        return {"min": float(np.min(arr[mask])),
                "max": float(np.max(arr[mask])),
                "mean": float(np.mean(arr[mask]))}

    metrics = {
        "smooth": {
            "h_min": stat(r_smooth.h_min, mask_last),
            "p_max": stat(r_smooth.p_max, mask_last),
            "cav_frac": stat(r_smooth.cav_frac, mask_last),
            "X_final": float(r_smooth.X[-1]),
            "Y_final": float(r_smooth.Y[-1]),
        },
        "textured": {
            "h_min": stat(r_tex.h_min, mask_last),
            "p_max": stat(r_tex.p_max, mask_last),
            "cav_frac": stat(r_tex.cav_frac, mask_last),
            "X_final": float(r_tex.X[-1]),
            "Y_final": float(r_tex.Y[-1]),
        },
    }
    h_s = metrics["smooth"]["h_min"]["min"]
    h_t = metrics["textured"]["h_min"]["min"]
    p_s = metrics["smooth"]["p_max"]["max"]
    p_t = metrics["textured"]["p_max"]["max"]
    metrics["gain_hmin"] = h_t / h_s if h_s > 0 else 0
    metrics["gain_pmax"] = p_t / p_s if p_s > 0 else 0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Загрузить npz и построить графики")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    default_dir = os.path.join(os.path.dirname(__file__), "..",
                                "results", "pump_harmonic")
    out_dir = args.data_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("НЕСТАЦИОНАРНЫЙ РАСЧЁТ НАСОСА — ГАРМОНИЧЕСКАЯ НАГРУЗКА")
    print(f"R={R*1e3:.1f}мм, L={L*1e3:.1f}мм, C={C*1e6:.0f}мкм")
    print(f"N={N_RPM}об/мин, f_vib={F_VIB:.0f}Гц ({N_BLADES} лопаток)")
    print(f"Сетка (interior): {N1}×{N2}, B={B_AUSAS}")
    print(f"dt={DT}, NT={NT}, T_vib={T_VIB:.4f}")
    print(f"mass_M={MASS_M}, W_static_x={W_STATIC_X}, ΔW={DELTA_W}")
    print(f"Результаты → {out_dir}")
    print("=" * 70)

    class LoadedResult:
        """Совместимый с AusasTransientResult."""
        pass

    if args.plot_only:
        print("\n--plot-only: загрузка npz")
        r_smooth = LoadedResult()
        r_tex = LoadedResult()
        for name, r in [("smooth", r_smooth), ("textured", r_tex)]:
            path = os.path.join(out_dir, f"{name}.npz")
            if not os.path.exists(path):
                print(f"  Нет {path}")
                sys.exit(1)
            d = np.load(path)
            for key in ["t", "X", "Y", "WX", "WY",
                         "p_max", "h_min", "cav_frac"]:
                setattr(r, key, d[key])
    else:
        print("\nПостроение текстуры...")
        relief = build_texture_relief()
        print(f"  shape={relief.shape}, min={relief.min():.4f}, "
              f"max={relief.max():.4f}")

        r_smooth = run_transient(None, "smooth", out_dir)
        r_tex = run_transient(relief, "textured", out_dir)

    plot_comparison(r_smooth, r_tex, out_dir)
    metrics = compute_metrics(r_smooth, r_tex)

    print("\n" + "=" * 70)
    print("МЕТРИКИ (по последнему периоду)")
    print("=" * 70)
    for key in ["smooth", "textured"]:
        m = metrics[key]
        print(f"  {key}:")
        print(f"    h_min: min={m['h_min']['min']:.4f}, "
              f"max={m['h_min']['max']:.4f}, mean={m['h_min']['mean']:.4f}")
        print(f"    p_max: max={m['p_max']['max']:.4f}")
        print(f"    cav_frac: mean={m['cav_frac']['mean']:.3f}")
    print(f"\n  gain_hmin = {metrics['gain_hmin']:.4f} (>1 = текстура помогает)")
    print(f"  gain_pmax = {metrics['gain_pmax']:.4f}")

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
