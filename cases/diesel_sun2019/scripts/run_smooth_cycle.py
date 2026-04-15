#!/usr/bin/env python3
"""Smooth baseline: дизельный main bearing, cyclic loading.

Прогоняет Ausas dynamic GPU solver цикл за циклом до установления
периодического режима. Сохраняет histories, checkpoints, графики,
JSON с метриками. Готовит данные для Части 3 (texture placement).
"""
import sys
import os
import csv
import json
import time
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(CASE_DIR))
sys.path.insert(0, CASE_DIR)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reynolds_solver import solve_ausas_journal_dynamic_gpu

try:
    from reynolds_solver import save_state, load_state
    _HAS_STATE_IO = True
except ImportError:
    _HAS_STATE_IO = False

import case_config as cfg
from scaling import (
    omega_from_rpm, pressure_scale, force_scale, mass_nondim,
    make_load_fn_from_crank, CYCLE_TAU, tau_to_crank_deg,
)


# ─── Эффективная масса ротора ─────────────────────────────────────
# На один main bearing: часть массы коленвала + распределённая масса
# поршневых групп. Для первого приближения — 10 кг.
M_EFF_KG = 10.0


def load_nondim_load(case_dir, n_rpm, load_pct):
    """Загружает безразмерный CSV и строит load_fn.

    Если _nd.csv нет — строим из размерного.
    """
    base = f"load_main_bearing_{n_rpm}rpm_{load_pct}pct"
    nd_path = os.path.join(case_dir, "derived", base + "_nd.csv")
    dim_path = os.path.join(case_dir, "derived", base + ".csv")

    if not os.path.exists(nd_path):
        print(f"  !! {nd_path} не найден")
        print(f"  Сначала запусти build_load_from_indicator.py")
        sys.exit(1)

    data = np.genfromtxt(nd_path, delimiter=",", skip_header=1)
    tau = data[:, 0]
    WaX_nd = data[:, 1]
    WaY_nd = data[:, 2]

    # Размерная для графиков
    if os.path.exists(dim_path):
        d2 = np.genfromtxt(dim_path, delimiter=",", skip_header=1)
        crank_deg = d2[:, 0]
        WaX_N = d2[:, 1]
        WaY_N = d2[:, 2]
    else:
        crank_deg = tau_to_crank_deg(tau)
        omega = omega_from_rpm(n_rpm)
        F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
        WaX_N = WaX_nd * F0
        WaY_N = WaY_nd * F0

    # Интерполятор через crank_deg
    load_fn = make_load_fn_from_crank(crank_deg, WaX_nd, WaY_nd)

    return load_fn, dict(tau=tau, crank_deg=crank_deg,
                          WaX_nd=WaX_nd, WaY_nd=WaY_nd,
                          WaX_N=WaX_N, WaY_N=WaY_N)


def check_cycle_convergence(r_prev, r_curr, tol=0.02):
    """Сходимость между двумя циклами по X, Y, h_min, p_max, cav."""
    n = min(len(r_prev.X), len(r_curr.X))
    Xp = np.asarray(r_prev.X[-n:])
    Xc = np.asarray(r_curr.X[-n:])
    Yp = np.asarray(r_prev.Y[-n:])
    Yc = np.asarray(r_curr.Y[-n:])

    rms_X = np.sqrt(np.mean((Xp - Xc) ** 2))
    rms_Y = np.sqrt(np.mean((Yp - Yc) ** 2))
    scale = max(np.max(np.abs(Xc)), np.max(np.abs(Yc)), 0.01)
    rel_XY = (rms_X + rms_Y) / scale

    hmin_prev = np.min(r_prev.h_min[-n:])
    hmin_curr = np.min(r_curr.h_min[-n:])
    rel_hmin = abs(hmin_prev - hmin_curr) / max(hmin_curr, 1e-6)

    pmax_prev = np.max(r_prev.p_max[-n:])
    pmax_curr = np.max(r_curr.p_max[-n:])
    rel_pmax = abs(pmax_prev - pmax_curr) / max(pmax_curr, 1e-6)

    cav_prev = np.mean(r_prev.cav_frac[-n:])
    cav_curr = np.mean(r_curr.cav_frac[-n:])
    rel_cav = abs(cav_prev - cav_curr)

    converged = (rel_XY < tol and rel_hmin < tol and rel_pmax < tol
                  and rel_cav < tol)

    print(f"    convergence: XY={rel_XY:.4f}, hmin={rel_hmin:.4f}, "
          f"pmax={rel_pmax:.4f}, cav={rel_cav:.4f} → "
          f"{'OK' if converged else 'not yet'}")
    return converged


def run_cycles(load_fn, out_dir, args):
    """Цикл за циклом до установления или max_cycles."""
    omega = omega_from_rpm(cfg.n_rpm)
    F0 = force_scale(cfg.eta, omega, cfg.R, cfg.c)
    p_sc = pressure_scale(cfg.eta, omega, cfg.R, cfg.c)
    M_nd = mass_nondim(M_EFF_KG, cfg.eta, omega, cfg.R, cfg.c)

    B_ausas = cfg.L_bearing / (2 * cfg.R)
    N_phi = cfg.N1 + 2
    N_Z_ = cfg.N2 + 2
    d_phi = 1.0 / cfg.N1
    d_Z = B_ausas / cfg.N2

    NT_cycle = int(CYCLE_TAU / cfg.dt)

    print("=" * 72)
    print("SMOOTH BASELINE — diesel main bearing (Sun 2019)")
    print(f"R={cfg.R*1e3:.1f}мм, L={cfg.L_bearing*1e3:.1f}мм, "
          f"c={cfg.c*1e6:.0f}мкм")
    print(f"n={cfg.n_rpm} rpm, ω={omega:.1f} рад/с")
    print(f"η={cfg.eta} Па·с, m_eff={M_EFF_KG}кг → M_nd={M_nd:.3e}")
    print(f"F₀={F0:.0f} Н, p_scale={p_sc/1e6:.2f} МПа, B_ausas={B_ausas:.4f}")
    print(f"Сетка {cfg.N1}×{cfg.N2} (interior), NT_cycle={NT_cycle}, "
          f"dt={cfg.dt}")
    print("=" * 72)

    all_results = []
    state = None
    converged_at = None

    t_all0 = time.time()
    for cycle_i in range(cfg.max_cycles):
        print(f"\nCycle {cycle_i + 1}/{cfg.max_cycles}")
        t0 = time.time()

        kwargs = dict(
            NT=NT_cycle, dt=cfg.dt,
            N_Z=N_Z_, N_phi=N_phi,
            d_phi=d_phi, d_Z=d_Z,
            R=0.5, L=1.0,
            mass_M=M_nd,
            load_fn=load_fn,
            texture_relief=None,
            omega_p=cfg.omega_p, omega_theta=cfg.omega_theta,
            tol_inner=cfg.tol_inner, max_inner=cfg.max_inner,
            scheme=cfg.scheme,
            verbose=False,
        )

        if state is None:
            kwargs["X0"] = cfg.X0
            kwargs["Y0"] = cfg.Y0
        else:
            # continuation: новый X0, Y0 из конца предыдущего цикла
            kwargs["X0"] = float(state["X"])
            kwargs["Y0"] = float(state["Y"])
            # Если solver принимает state — можно передать
            if _HAS_STATE_IO and hasattr(state, "_ausas_state"):
                kwargs["state"] = state["_ausas_state"]

        try:
            result = solve_ausas_journal_dynamic_gpu(**kwargs)
        except TypeError as e:
            # state не поддерживается — убрать и повторить
            kwargs.pop("state", None)
            result = solve_ausas_journal_dynamic_gpu(**kwargs)

        dt_cycle = time.time() - t0
        all_results.append(result)

        # Обновить state (минимум X, Y финальные)
        state = {
            "X": float(result.X[-1]),
            "Y": float(result.Y[-1]),
        }
        if hasattr(result, "final_state"):
            state["_ausas_state"] = result.final_state
            if _HAS_STATE_IO:
                try:
                    save_state(result.final_state,
                                os.path.join(out_dir,
                                              f"checkpoint_cycle_{cycle_i}.npz"))
                except Exception as e:
                    print(f"    save_state warn: {e}")

        X_f, Y_f = state["X"], state["Y"]
        eps_f = np.sqrt(X_f**2 + Y_f**2)
        print(f"  X={X_f:+.4f}, Y={Y_f:+.4f}, ε={eps_f:.4f}, "
              f"h_min range [{np.min(result.h_min):.4f}, "
              f"{np.max(result.h_min):.4f}], "
              f"p_max range [{np.min(result.p_max):.4f}, "
              f"{np.max(result.p_max):.4f}], {dt_cycle:.1f}с")

        # Сохранить scalar histories цикла
        np.savez(os.path.join(out_dir, f"cycle_{cycle_i}.npz"),
                  t=result.t, X=result.X, Y=result.Y,
                  WX=result.WX, WY=result.WY,
                  p_max=result.p_max, h_min=result.h_min,
                  cav_frac=result.cav_frac)

        # Convergence
        if cycle_i >= cfg.warmup_cycles and cycle_i >= 1:
            if check_cycle_convergence(all_results[-2], all_results[-1]):
                converged_at = cycle_i
                print(f"  CONVERGED on cycle {cycle_i + 1}")
                break

    t_all = time.time() - t_all0
    print(f"\nВсего: {len(all_results)} циклов за {t_all:.1f} с")
    return all_results, converged_at, dict(omega=omega, F0=F0,
                                             p_scale=p_sc, M_nd=M_nd)


def save_outputs(all_results, converged_at, scales, load_data, out_dir):
    """Графики + JSON + CSV для texture placement."""
    r_last = all_results[-1]
    t = np.asarray(r_last.t)
    X = np.asarray(r_last.X)
    Y = np.asarray(r_last.Y)
    h_min = np.asarray(r_last.h_min)
    p_max = np.asarray(r_last.p_max)
    cav = np.asarray(r_last.cav_frac)
    WX = np.asarray(r_last.WX)
    WY = np.asarray(r_last.WY)

    # Размерные
    h_min_um = h_min * cfg.c * 1e6
    p_max_MPa = p_max * scales["p_scale"] / 1e6
    crank_deg = tau_to_crank_deg(t - t[0])  # от начала цикла

    # --- 1. X(t), Y(t) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(crank_deg, X, "b-", lw=1.2, label="X")
    ax.plot(crank_deg, Y, "r-", lw=1.2, label="Y")
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smooth_XY_vs_ca.png"), dpi=150)
    plt.close(fig)

    # --- 2. Orbit ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(X, Y, "b-", lw=1.0)
    ax.plot(X[0], Y[0], "go", markersize=8, label="start")
    ax.plot(X[-1], Y[-1], "r^", markersize=8, label="end")
    # Единичная окружность (clearance)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smooth_orbit.png"), dpi=150)
    plt.close(fig)

    # --- 3. h_min, p_max, cav ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(crank_deg, h_min_um, "b-", lw=1.2)
    axes[0].set_ylabel("h_min, мкм")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(5, color="r", ls=":", lw=0.8, label="5 мкм")
    axes[0].axhline(10, color="orange", ls=":", lw=0.8, label="10 мкм")
    axes[0].legend()

    axes[1].plot(crank_deg, p_max_MPa, "r-", lw=1.2)
    axes[1].set_ylabel("p_max, МПа")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(50, color="orange", ls=":", lw=0.8, label="50 МПа")
    axes[1].axhline(100, color="r", ls=":", lw=0.8, label="100 МПа")
    axes[1].legend()

    axes[2].plot(crank_deg, cav, "g-", lw=1.2)
    axes[2].set_ylabel("cav_frac")
    axes[2].set_xlabel("crank angle (°)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smooth_metrics_vs_ca.png"), dpi=150)
    plt.close(fig)

    # --- 4. Нагрузка ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(load_data["crank_deg"], load_data["WaX_N"] / 1000, "b-",
             lw=1.2, label="WaX")
    ax.plot(load_data["crank_deg"], load_data["WaY_N"] / 1000, "r-",
             lw=1.2, label="WaY")
    ax.set_xlabel("crank angle (°)")
    ax.set_ylabel("F, кН")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "load_vs_ca.png"), dpi=150)
    plt.close(fig)

    # --- 5. JSON scalar metrics ---
    eps_t = np.sqrt(X**2 + Y**2)
    metrics = {
        "n_cycles_run": len(all_results),
        "converged_at_cycle": converged_at,
        "last_cycle": {
            "h_min_min_nondim": float(np.min(h_min)),
            "h_min_min_um": float(np.min(h_min_um)),
            "p_max_max_nondim": float(np.max(p_max)),
            "p_max_max_MPa": float(np.max(p_max_MPa)),
            "cav_frac_mean": float(np.mean(cav)),
            "cav_frac_max": float(np.max(cav)),
            "e_max": float(np.max(eps_t)),
            "e_min": float(np.min(eps_t)),
            "X_final": float(X[-1]),
            "Y_final": float(Y[-1]),
        },
        "scales": {
            "F0_N": float(scales["F0"]),
            "p_scale_Pa": float(scales["p_scale"]),
            "M_nondim": float(scales["M_nd"]),
            "omega_rad_s": float(scales["omega"]),
        },
        "checks": {
            "e_max_le_1": bool(np.max(eps_t) < 1.0),
            "h_min_pos": bool(np.min(h_min) > 0),
        },
    }
    with open(os.path.join(out_dir, "smooth_metrics.json"),
               "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --- 6. Данные для texture placement (Часть 3) ---
    # Угол максимума давления как функция CA (нужен P_last на каждом шаге,
    # но в AusasTransientResult обычно сохранены только scalar histories).
    # Сохраним то что есть: h_min_phi (индекс угла минимума зазора)
    # из final field, если доступен; иначе только таблица scalar per-step.
    # Это ориентировочные данные.
    tex_csv_path = os.path.join(out_dir, "smooth_scalars_last_cycle.csv")
    with open(tex_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crank_deg", "X", "Y", "e",
                     "h_min_nondim", "p_max_nondim",
                     "cav_frac", "WaX_nd", "WaY_nd"])
        for i in range(len(t)):
            w.writerow([f"{crank_deg[i]:.3f}",
                        f"{X[i]:.6f}", f"{Y[i]:.6f}",
                        f"{eps_t[i]:.6f}",
                        f"{h_min[i]:.6f}", f"{p_max[i]:.6f}",
                        f"{cav[i]:.6f}",
                        f"{WX[i]:.6f}", f"{WY[i]:.6f}"])

    # P-field последнего шага (для texture placement)
    if hasattr(r_last, "P_last") and r_last.P_last is not None:
        np.savez(os.path.join(out_dir, "smooth_P_last.npz"),
                  P_last=r_last.P_last,
                  theta_last=getattr(r_last, "theta_last", None))

    print(f"\nВыход → {out_dir}")
    print(f"  smooth_XY_vs_ca.png, smooth_orbit.png,")
    print(f"  smooth_metrics_vs_ca.png, load_vs_ca.png,")
    print(f"  smooth_metrics.json, smooth_scalars_last_cycle.csv")

    # Check summary
    print("\n" + "=" * 60)
    print("ПРОВЕРКИ")
    print("=" * 60)
    print(f"  [{'✓' if metrics['checks']['e_max_le_1'] else '✗'}] "
          f"e_max = {metrics['last_cycle']['e_max']:.4f} < 1.0")
    print(f"  [{'✓' if metrics['checks']['h_min_pos'] else '✗'}] "
          f"h_min = {metrics['last_cycle']['h_min_min_um']:.3f} мкм > 0")
    print(f"  [{'✓' if converged_at is not None else '~'}] "
          f"converged at cycle {converged_at if converged_at is not None else 'NOT (max_cycles reached)'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rpm", type=int, default=cfg.n_rpm)
    parser.add_argument("--load-pct", type=int, default=cfg.load_pct)
    args = parser.parse_args()

    out_dir = os.path.join(CASE_DIR, "results",
                            f"smooth_{args.n_rpm}rpm_{args.load_pct}pct")
    os.makedirs(out_dir, exist_ok=True)

    load_fn, load_data = load_nondim_load(CASE_DIR, args.n_rpm, args.load_pct)

    all_results, converged_at, scales = run_cycles(load_fn, out_dir, args)
    save_outputs(all_results, converged_at, scales, load_data, out_dir)


if __name__ == "__main__":
    main()
