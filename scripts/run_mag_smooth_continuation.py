#!/usr/bin/env python3
"""Smooth pump bearing + radial magnetic unloading.

Stationary Payvar-Salant + radial restoring surrogate. Continuation
через shared driver (models/magnetic_equilibrium). Только accepted
targets попадают в manifest.

Output layout (magnetic_v4):
    results/magnetic_pump/<run_id>/
        manifest.json                — единственный source of truth
        mag_smooth_equilibrium.csv   — debug/inspection dump

Никаких записей в родительский каталог results/magnetic_pump/.
"""
import sys
import os
import csv
import json
import time
import argparse
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_gpu
    _ps_solver = solve_payvar_salant_gpu
except ImportError:
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    _ps_solver = solve_payvar_salant_cpu

from models.magnetic_force import (
    RadialUnloadForceModel, sanity_checks,
)
from models.magnetic_equilibrium import (
    find_equilibrium, run_continuation,
    is_accepted, result_to_dict,
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL

# ─── Расчёт ──────────────────────────────────────────────────────
SCHEMA_VERSION = "magnetic_v4"

N_PHI = 800
N_Z = 200
ETA = MINERAL_OIL["eta_pump"]
OMEGA = 2 * np.pi * params.n / 60.0
P_SCALE = 6 * ETA * OMEGA * (params.R / params.c) ** 2
F0 = P_SCALE * params.R * params.L

UNLOAD_SHARE_TARGETS = [0.0, 0.025, 0.05, 0.10, 0.20, 0.30]

# Единый acceptance tolerance. Записывается в manifest и повторно
# используется textured compare / plot.
TOL_ACCEPT = 5e-3
STEP_CAP = 0.10
EPS_MAX = 0.90


def make_grid(N_phi, N_Z):
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z = np.linspace(-1, 1, N_Z)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def make_smooth_H(X, Y, Phi):
    H0 = 1.0 + X * np.cos(Phi) + Y * np.sin(Phi)
    return np.sqrt(H0**2 + (params.sigma / params.c) ** 2)


def make_H_and_force(Phi, Zm, phi_1D, Z_1D, d_phi, d_Z):
    """Замыкание: build_H(X,Y) + PS solve + metrics."""

    def H_and_force(X, Y):
        H = make_smooth_H(X, Y, Phi)
        P, theta, _, _ = _ps_solver(
            H, d_phi, d_Z, params.R, params.L,
            tol=1e-6, max_iter=10_000_000)
        # Hydro force
        P_dim = P * P_SCALE
        Fx = -np.trapezoid(
            np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * params.R * params.L / 2
        Fy = -np.trapezoid(
            np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
            Z_1D, axis=0) * params.R * params.L / 2
        # Metrics
        h_dim = H * params.c
        h_min = float(np.min(h_dim))
        p_max = float(np.max(P_dim))
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
        tau_c = ETA * OMEGA * params.R / h_dim
        friction = float(
            np.sum(tau_c) * params.R * (2 * np.pi / H.shape[1])
            * params.L * (2 / H.shape[0]) / 2)
        return (float(Fx), float(Fy), h_min, p_max, cav_frac, friction,
                P, theta)

    return H_and_force


def make_run_id(model, n_phi, n_z, w_y_share):
    """Run-id хранит ключевые параметры → разные runs не пересекаются."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    wy = f"{w_y_share:.3f}".replace(".", "")
    return f"{stamp}_{model}_{n_phi}x{n_z}_wy{wy}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-phi", type=int, default=N_PHI)
    parser.add_argument("--n-z", type=int, default=N_Z)
    parser.add_argument("--W-y-share", type=float, default=0.25)
    parser.add_argument("--model", choices=["radial", "linear"],
                        default="radial")
    parser.add_argument("--run-id", type=str, default=None,
                        help="explicit run_id; default: auto-generated")
    parser.add_argument("--tol-accept", type=float, default=TOL_ACCEPT,
                        help="acceptance threshold (rel_residual); "
                             "used by smooth AND textured compare")
    parser.add_argument("--step-cap", type=float, default=STEP_CAP)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    args = parser.parse_args()

    run_id = args.run_id or make_run_id(
        args.model, args.n_phi, args.n_z, args.W_y_share)
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "magnetic_pump", run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run directory: {out_dir}")

    # Sanity
    ok, _ = sanity_checks(verbose=True)
    if not ok:
        print("FAIL: sanity")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"SMOOTH PUMP + RADIAL MAG UNLOAD (model={args.model})")
    print(f"Сетка {args.n_phi}×{args.n_z}, "
          f"W_applied=(0, {-args.W_y_share*F0:.1f}) Н, F₀={F0:.0f}Н")
    print(f"unload_share targets: {UNLOAD_SHARE_TARGETS}")
    print(f"tol_accept={args.tol_accept}, step_cap={args.step_cap}, "
          f"eps_max={args.eps_max}")
    print("=" * 72)

    W_applied = np.array([0.0, -args.W_y_share * F0])
    phi, Z, Phi, Zm, dp, dz = make_grid(args.n_phi, args.n_z)
    H_and_force = make_H_and_force(Phi, Zm, phi, Z, dp, dz)

    # ── Baseline raw (initial solve from seed) ─────────────────────
    print("\nBaseline raw (no magnet, seed X0=0, Y0=-0.4)...")
    zero_model = RadialUnloadForceModel(K_mag=0.0)
    t0 = time.time()
    base_raw = find_equilibrium(
        H_and_force, zero_model, W_applied,
        X0=0.0, Y0=-0.4,
        tol=args.tol_accept, step_cap=args.step_cap, eps_max=args.eps_max,
        tol_accept=args.tol_accept)
    base_raw.unload_share_target = 0.0
    print(f"  X={base_raw.X:+.4f}, Y={base_raw.Y:+.4f}, "
          f"ε={base_raw.eps:.4f}, "
          f"h_min={base_raw.h_min*1e6:.2f}μm, "
          f"p_max={base_raw.p_max/1e6:.2f}MPa, "
          f"res={base_raw.rel_residual:.1e}, n_it={base_raw.n_iter}, "
          f"status={base_raw.status}, "
          f"{time.time()-t0:.1f}с")
    if not base_raw.converged:
        print(f"BASELINE RAW НЕ сошёлся — stopping")
        sys.exit(2)

    # ── Continuation ───────────────────────────────────────────────
    if args.model == "radial":
        template = RadialUnloadForceModel(n_mag=3, H_reg=0.05, H_floor=0.02)
    else:
        from models.magnetic_force import LinearSpringForceModel
        template = LinearSpringForceModel()

    print("\nContinuation...")
    cont = run_continuation(
        UNLOAD_SHARE_TARGETS, base_raw.X, base_raw.Y, W_applied,
        template, H_and_force,
        tol=args.tol_accept,
        step_cap=args.step_cap, eps_max=args.eps_max,
        tol_accept=args.tol_accept,
        verbose=True)

    # canonical baseline = accepted point для target=0.0
    # (см. ТЗ §3.2 — в downstream computations нельзя использовать
    # base_raw, только canonical zero-target entry).
    r0 = cont[0][1] if cont and cont[0][2] else None
    if r0 is None or not is_accepted(r0, args.tol_accept):
        print("\nFAIL: target=0.0 не accepted → baseline_canonical недоступен")
        sys.exit(3)
    base_canonical = r0

    # ── Acceptance checks ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ACCEPTANCE")
    print("=" * 72)

    # K=0 reproduces baseline (shape-only tolerance).
    ok1 = (
        is_accepted(base_canonical, args.tol_accept)
        and abs(base_canonical.eps - base_raw.eps) < 0.10
        and abs(base_canonical.h_min - base_raw.h_min)
            / max(base_raw.h_min, 1e-12) < 0.10
        and abs(base_canonical.p_max - base_raw.p_max)
            / max(base_raw.p_max, 1e-12) < 0.10
    )
    print(f"  [{'✓' if ok1 else '✗'}] K_mag=0 reproduces raw baseline "
          f"(|Δε|={abs(base_canonical.eps - base_raw.eps):.3e}, "
          f"res_canonical={base_canonical.rel_residual:.1e})")

    accepted_pairs = [(t, r) for (t, r, a) in cont if a]
    accepted_targets = [t for t, _ in accepted_pairs]

    ok2 = all(r.unload_share_actual > 0
              for t, r in accepted_pairs if t > 0)
    print(f"  [{'✓' if ok2 else '✗'}] unload_share_actual > 0 на accepted")

    eps_seq = [r.eps for _, r in accepted_pairs]
    ok3 = all(eps_seq[i+1] <= eps_seq[i] + 1e-3
              for i in range(len(eps_seq) - 1))
    print(f"  [{'✓' if ok3 else '✗'}] ε монотонно не возрастает: "
          f"{[f'{e:.4f}' for e in eps_seq]}")

    max_res = max((r.rel_residual for _, r in accepted_pairs), default=0.0)
    ok4 = max_res < args.tol_accept
    print(f"  [{'✓' if ok4 else '✗'}] max residual = {max_res:.2e} "
          f"< {args.tol_accept}")

    ok5 = all(abs(r.hydro_share_actual + r.unload_share_actual - 1.0)
              < 2 * args.tol_accept
              for _, r in accepted_pairs)
    print(f"  [{'✓' if ok5 else '✗'}] |hydro + unload − 1| "
          f"< {2*args.tol_accept}")

    # ── CSV dump ──────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "mag_smooth_equilibrium.csv")
    rows = []
    for t, r, acc in cont:
        if r is None:
            rows.append(dict(unload_share_target=t, accepted=False,
                              note="skipped_after_chain_break"))
        else:
            d = result_to_dict(r)
            d["accepted"] = bool(acc)
            rows.append(d)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nCSV: {csv_path}")

    # ── Manifest (magnetic_v4) ─────────────────────────────────────
    smooth_all = []
    for t, r, a in cont:
        smooth_all.append(dict(
            unload_share_target=float(t),
            result=result_to_dict(r),
            accepted=bool(a),
        ))

    smooth_accepted = [result_to_dict(r) for _, r, a in cont if a]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": args.model,
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "N_phi": int(args.n_phi),
            "N_Z": int(args.n_z),
            "W_applied_N": [float(w_) for w_ in W_applied],
            "F0_N": float(F0),
            "p_scale_Pa": float(P_SCALE),
            "tol_accept": float(args.tol_accept),
            "step_cap": float(args.step_cap),
            "eps_max": float(args.eps_max),
            "targets": [float(t) for t in UNLOAD_SHARE_TARGETS],
        },
        "baseline_raw": result_to_dict(base_raw),
        "baseline_canonical": result_to_dict(base_canonical),
        "smooth_accepted": smooth_accepted,
        "smooth_all": smooth_all,
        "acceptance": {
            "baseline_reproduced": bool(ok1),
            "unload_positive": bool(ok2),
            "eps_monotonic": bool(ok3),
            "max_residual": float(max_res),
            "sum_shares_ok": bool(ok5),
            "accepted_targets": [float(t) for t in accepted_targets],
        },
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest: {manifest_path}")

    # Convenience: pointer file на latest run, чтобы textured compare
    # и plot могли найти manifest без явного run_id.
    latest_path = os.path.join(
        os.path.dirname(__file__), "..",
        "results", "magnetic_pump", "latest_run.txt")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_id + "\n")
    print(f"Latest pointer: {latest_path} → {run_id}")


if __name__ == "__main__":
    main()
