#!/usr/bin/env python3
"""Smooth pump bearing с магнитной разгрузкой — continuation по mag_share.

Exploratory case. Для каждого mag_share_target в [0, 0.05, 0.10, 0.20, 0.30]:
  1. откалибровать K_mag (в baseline point)
  2. найти новое равновесие (X_eq, Y_eq) через 2D Newton-Raphson
  3. сохранить метрики

Обязательный sanity check: K_mag = 0 воспроизводит baseline.
"""
import sys
import os
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu

from models.magnetic_force import MagneticForceModel, calibrate_Kmag, sanity_checks
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Параметры расчёта ───────────────────────────────────────────
N_PHI = 800
N_Z = 200
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * params.n / 60.0
P_SCALE = 6 * ETA * OMEGA * (params.R / params.c) ** 2
F0 = P_SCALE * params.R * params.L  # масштаб силы в Н

# Applied load: вдоль -Y (вертикально), величина подобрана для ε_baseline ≈ 0.5
W_APPLIED_X = 0.0
W_APPLIED_Y = -0.25 * F0        # подобрать калибровкой
W_APPLIED = np.array([W_APPLIED_X, W_APPLIED_Y])

MAG_SHARE_TARGETS = [0.0, 0.05, 0.10, 0.20, 0.30]


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_H(X, Y, Phi, p=params):
    """H = 1 + X·cos(φ) + Y·sin(φ) + регуляризация шероховатости."""
    H0 = 1.0 + X * np.cos(Phi) + Y * np.sin(Phi)
    H0 = np.sqrt(H0**2 + (p.sigma / p.c) ** 2)
    return H0


def solve_ps(H, dp, dz):
    P, theta, res, nit = _ps_solver(
        H, dp, dz, params.R, params.L, tol=1e-6, max_iter=10_000_000)
    return P, theta


def compute_hydro_force(P, Phi, phi_1D, Z_1D):
    """Hydrodynamic force в Ньютонах (размерная, по pump convention).

    Fx = -∫∫ P_dim·cos(φ) R dφ (L·dZ/2)
    Fy = -∫∫ P_dim·sin(φ) R dφ (L·dZ/2)
    """
    P_dim = P * P_SCALE
    Fx = -np.trapezoid(np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * params.R * params.L / 2
    Fy = -np.trapezoid(np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
                        Z_1D, axis=0) * params.R * params.L / 2
    return float(Fx), float(Fy)


def compute_metrics(P, H, theta=None):
    P_dim = P * P_SCALE
    h_dim = H * params.c
    h_min = float(np.min(h_dim))
    p_max = float(np.max(P_dim))
    if theta is not None:
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    else:
        cav_frac = 0.0

    # Friction (приближённо, Couette component dominates)
    tau_couette = ETA * OMEGA * params.R / h_dim
    F_friction = float(np.sum(tau_couette) * params.R * (2 * np.pi / H.shape[1])
                       * params.L * (2 / H.shape[0]) / 2)
    return h_min, p_max, cav_frac, F_friction


def find_equilibrium(W_applied, mag_model, Phi, Zm, phi_1D, Z_1D, d_phi, d_Z,
                     X0=0.0, Y0=-0.4, max_iter=50, tol=1e-4):
    """2D Newton-Raphson: find (X,Y) such that
       F_hydro(X,Y) + F_mag(X,Y) = W_applied.
    С адаптивным демпфированием (backtracking).
    """
    X, Y = X0, Y0
    dXY = 1e-5
    P_last = None
    theta_last = None
    prev_rel_R = np.inf

    for it in range(max_iter):
        H = make_H(X, Y, Phi)
        P, theta = solve_ps(H, d_phi, d_Z)
        Fx_h, Fy_h = compute_hydro_force(P, Phi, phi_1D, Z_1D)
        Fx_m, Fy_m = mag_model.force(X, Y)

        Rx = Fx_h + Fx_m - W_applied[0]
        Ry = Fy_h + Fy_m - W_applied[1]
        norm_R = np.sqrt(Rx**2 + Ry**2)
        norm_W = np.linalg.norm(W_applied)
        rel_R = norm_R / max(norm_W, 1e-20)

        if rel_R < tol:
            P_last, theta_last = P, theta
            break

        # Jacobian via finite differences
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0), (0, dXY)]):
            H_p = make_H(X + dX_, Y + dY_, Phi)
            P_p, _ = solve_ps(H_p, d_phi, d_Z)
            Fxp, Fyp = compute_hydro_force(P_p, Phi, phi_1D, Z_1D)
            Fxm_p, Fym_p = mag_model.force(X + dX_, Y + dY_)
            J[0, col] = ((Fxp + Fxm_p) - (Fx_h + Fx_m)) / dXY
            J[1, col] = ((Fyp + Fym_p) - (Fy_h + Fy_m)) / dXY

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det

        # Backtracking: уменьшить шаг если residual вырос или step too big
        step_cap = 0.08  # максимум 0.08 по X или Y за шаг
        step = min(1.0, step_cap / max(abs(dX), abs(dY), 1e-10))
        # Ещё ограничиваем X, Y в допустимой области |ε|<0.97
        X_new = X + step * dX
        Y_new = Y + step * dY
        eps_new = np.sqrt(X_new**2 + Y_new**2)
        if eps_new > 0.95:
            step *= 0.5
        X += step * dX
        Y += step * dY
        P_last, theta_last = P, theta
        prev_rel_R = rel_R

    eps = np.sqrt(X**2 + Y**2)
    H = make_H(X, Y, Phi)
    h_min, p_max, cav_frac, fr = compute_metrics(P_last, H, theta_last)
    Fx_h, Fy_h = compute_hydro_force(P_last, Phi, phi_1D, Z_1D)
    Fx_m, Fy_m = mag_model.force(X, Y)
    Rx = Fx_h + Fx_m - W_applied[0]
    Ry = Fy_h + Fy_m - W_applied[1]
    rel_R = np.sqrt(Rx**2 + Ry**2) / max(np.linalg.norm(W_applied), 1e-20)
    attitude = np.rad2deg(np.arctan2(Y, X))
    return dict(
        X=X, Y=Y, eps=float(eps), attitude_deg=float(attitude),
        Fx_hydro=float(Fx_h), Fy_hydro=float(Fy_h),
        Fx_mag=float(Fx_m), Fy_mag=float(Fy_m),
        h_min=h_min, p_max=p_max, cav_frac=cav_frac, friction=fr,
        n_iter=it + 1, rel_residual=float(rel_R),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phi", type=int, default=N_PHI)
    parser.add_argument("--n-z", type=int, default=N_Z)
    parser.add_argument("--W-y-share", type=float, default=0.25,
                        help="applied load вдоль -Y, доля от F₀")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "magnetic_pump")
    os.makedirs(out_dir, exist_ok=True)

    # Sanity
    ok, msgs = sanity_checks(verbose=True)
    if not ok:
        print("FAIL: sanity checks")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("SMOOTH PUMP BEARING + MAGNETIC UNLOADING")
    print(f"R={params.R*1e3:.1f}мм, L={params.L*1e3:.1f}мм, "
          f"c={params.c*1e6:.0f}мкм, n={params.n}об/мин")
    print(f"F₀={F0:.1f} Н, W_applied = ({W_APPLIED[0]:.1f}, "
          f"{-args.W_y_share*F0:.1f}) Н")
    print(f"Сетка {args.n_phi}×{args.n_z}")
    print(f"mag_share targets: {MAG_SHARE_TARGETS}")
    print("=" * 72)

    W_applied = np.array([0.0, -args.W_y_share * F0])
    e_load = -W_applied / np.linalg.norm(W_applied)
    W_norm = float(np.linalg.norm(W_applied))

    phi, Z, Phi, Zm, dp, dz = make_grid(args.n_phi, args.n_z)

    # --- Baseline ---
    print("\n[1/2] Baseline (no magnet)...")
    t0 = time.time()
    mag0 = MagneticForceModel(K_mag=0.0)
    baseline = find_equilibrium(W_applied, mag0, Phi, Zm, phi, Z, dp, dz)
    dt_b = time.time() - t0
    print(f"  X={baseline['X']:.4f}, Y={baseline['Y']:.4f}, "
          f"ε={baseline['eps']:.4f}, "
          f"h_min={baseline['h_min']*1e6:.2f}мкм, "
          f"p_max={baseline['p_max']/1e6:.2f}МПа, "
          f"Fx_h={baseline['Fx_hydro']:.1f}, "
          f"Fy_h={baseline['Fy_hydro']:.1f}")
    print(f"  residual={baseline['rel_residual']:.2e}, "
          f"iter={baseline['n_iter']}, time={dt_b:.1f}с")

    # --- Continuation ---
    print("\n[2/2] Continuation по mag_share...")
    results = []
    X_prev, Y_prev = baseline["X"], baseline["Y"]

    prev_target = 0.0
    for target in MAG_SHARE_TARGETS:
        if target == 0.0:
            mag = MagneticForceModel(K_mag=0.0)
            K_out = 0.0
        else:
            mag = calibrate_Kmag(
                baseline["X"], baseline["Y"], W_applied, target)
            K_out = mag.K_mag

        t0 = time.time()
        r = find_equilibrium(W_applied, mag, Phi, Zm, phi, Z, dp, dz,
                              X0=X_prev, Y0=Y_prev)

        # Retry через промежуточные sub-steps если не сошёлся
        if r["rel_residual"] > 1e-3 and target > prev_target:
            print(f"  share={target*100:.1f}%: NR не сошёлся "
                  f"(res={r['rel_residual']:.1e}), retry с substeps...")
            n_sub = 4
            X_mid, Y_mid = X_prev, Y_prev
            for isub in range(1, n_sub + 1):
                sub_target = prev_target + (target - prev_target) * isub / n_sub
                mag_sub = calibrate_Kmag(
                    baseline["X"], baseline["Y"], W_applied, sub_target)
                r = find_equilibrium(W_applied, mag_sub, Phi, Zm, phi, Z,
                                      dp, dz, X0=X_mid, Y0=Y_mid)
                X_mid, Y_mid = r["X"], r["Y"]
            K_out = mag_sub.K_mag

        dt_r = time.time() - t0

        # load shares
        hydro_share = ((r["Fx_hydro"] * e_load[0] +
                         r["Fy_hydro"] * e_load[1]) / W_norm)
        mag_share = ((r["Fx_mag"] * e_load[0] +
                       r["Fy_mag"] * e_load[1]) / W_norm)
        r.update(dict(
            mag_share_target=target,
            K_mag=K_out,
            hydro_load_share=float(hydro_share),
            mag_load_share=float(mag_share),
        ))
        results.append(r)
        print(f"  share={target*100:4.1f}%: X={r['X']:+.4f}, "
              f"Y={r['Y']:+.4f}, ε={r['eps']:.4f}, "
              f"h_min={r['h_min']*1e6:.2f}мкм, "
              f"p_max={r['p_max']/1e6:.2f}МПа, "
              f"hydro={hydro_share:.3f}, mag={mag_share:.3f}, "
              f"res={r['rel_residual']:.1e}, {dt_r:.1f}с")

        X_prev, Y_prev = r["X"], r["Y"]
        prev_target = target

    # --- Acceptance checks ---
    print("\n" + "=" * 72)
    print("ACCEPTANCE")
    print("=" * 72)
    r0 = results[0]
    delta_eps_baseline = abs(r0["eps"] - baseline["eps"])
    delta_h = abs(r0["h_min"] - baseline["h_min"]) / max(baseline["h_min"], 1e-12)
    delta_p = abs(r0["p_max"] - baseline["p_max"]) / max(baseline["p_max"], 1e-12)
    print(f"  [{'✓' if delta_eps_baseline < 1e-4 else '✗'}] "
          f"K_mag=0 reproduces baseline: |Δε|={delta_eps_baseline:.2e}")
    print(f"  [{'✓' if delta_h < 0.001 else '✗'}] "
          f"|Δh_min|/h_min = {delta_h:.2e}")
    print(f"  [{'✓' if delta_p < 0.001 else '✗'}] "
          f"|Δp_max|/p_max = {delta_p:.2e}")

    eps_sequence = [r["eps"] for r in results]
    monotonic = all(eps_sequence[i+1] <= eps_sequence[i] + 1e-4
                    for i in range(len(eps_sequence) - 1))
    print(f"  [{'✓' if monotonic else '✗'}] "
          f"ε монотонно уменьшается с ростом mag_share: "
          f"{[f'{e:.4f}' for e in eps_sequence]}")

    max_res = max(r["rel_residual"] for r in results)
    print(f"  [{'✓' if max_res < 1e-3 else '✗'}] "
          f"max residual < 1e-3: {max_res:.2e}")

    # --- Save CSV ---
    csv_path = os.path.join(out_dir, "mag_smooth_equilibrium.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nCSV: {csv_path}")

    # --- Save JSON ---
    json_path = os.path.join(out_dir, "mag_smooth_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "N_phi": args.n_phi, "N_Z": args.n_z,
                "F0_N": F0, "W_applied_N": [float(w) for w in W_applied],
                "p_scale_Pa": P_SCALE,
                "sector_angles_deg": [0, 120, 240],
                "mag_share_targets": MAG_SHARE_TARGETS,
            },
            "baseline": {k: (v if isinstance(v, (int, float)) else float(v))
                         for k, v in baseline.items()},
            "continuation": [
                {k: (v if isinstance(v, (int, float)) else float(v))
                 for k, v in r.items()} for r in results],
            "acceptance": {
                "baseline_reproduced": bool(delta_eps_baseline < 1e-4
                                              and delta_h < 0.001
                                              and delta_p < 0.001),
                "monotonic_eps_decrease": bool(monotonic),
                "max_residual": float(max_res),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
