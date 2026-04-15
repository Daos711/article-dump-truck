#!/usr/bin/env python3
"""Расширенный параметрический поиск текстуры с PV+PS.

Этап A: грубый screening (800×200, ~864 комбинаций × 3 ε).
Этап B: перепроверка топ-кандидатов (1600×400, оба масла, PV/noPV).
"""
import sys
import os
import time
import csv
import warnings
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import types

from models.bearing_model import setup_grid, make_H, solve_and_compute
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from config import pump_params as base_params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

CAVITATION = "payvar_salant"
PROFILE = "smoothcap"

# ─── Этап A: параметры ────────────────────────────────────────────
A_NPHI, A_NZ = 800, 200
A_EPS_VALUES = [0.3, 0.5, 0.7]

HP_VALUES = [2e-6, 3e-6, 5e-6, 7e-6, 10e-6, 15e-6, 20e-6, 30e-6]

ZONES = [
    (0, 90), (0, 180), (45, 135), (90, 180), (60, 180),
    (0, 270), (90, 270), (180, 360), (0, 360),
]

DIMPLE_SIZES = [
    (0.5e-3, 0.4e-3),
    (1.0e-3, 0.8e-3),
    (1.5e-3, 1.2e-3),
    (2.0e-3, 1.5e-3),
]

SPACING_FACTORS = [1.5, 2.0, 3.0]

# ─── Этап B: параметры ────────────────────────────────────────────
B_NPHI, B_NZ = 1600, 400

MIN_NODES_PER_DIMPLE = 6


def make_base_params():
    return types.SimpleNamespace(**{k: getattr(base_params, k)
                                    for k in dir(base_params)
                                    if not k.startswith('_')})


def compute_N_tex(zone, a_dim, b_dim, R, L, spacing_factor):
    """Число лунок с заданным spacing factor."""
    ps, pe = zone
    if ps < pe:
        phi_span = np.deg2rad(pe - ps)
    else:
        phi_span = np.deg2rad((360 - ps) + pe)

    B = b_dim / R
    A = 2 * a_dim / L
    N_phi = max(1, int(phi_span / (spacing_factor * 2 * B)))
    N_Z = max(1, int(2.0 / (spacing_factor * 2 * A)))
    return N_phi, N_Z


def setup_texture_custom(p):
    """Центры лунок с отступом от краёв и шва."""
    B = p.b_dim / p.R
    A = 2 * p.a_dim / p.L

    phi_s = np.deg2rad(p.phi_start_deg)
    phi_e = np.deg2rad(p.phi_end_deg)

    if p.phi_start_deg < p.phi_end_deg:
        phi_span = phi_e - phi_s
    else:
        phi_span = (2 * np.pi - phi_s) + phi_e

    margin = B * 1.1
    usable = phi_span - 2 * margin
    if usable <= 0 or p.N_phi_tex < 1:
        return np.array([]), np.array([])

    if p.N_phi_tex == 1:
        phi_c = np.array([phi_s + phi_span / 2])
    else:
        phi_c = phi_s + margin + np.linspace(0, usable, p.N_phi_tex)
    phi_c = phi_c % (2 * np.pi)

    margin_Z = A * 1.1
    usable_Z = 2.0 - 2 * margin_Z
    if usable_Z <= 0 or p.N_Z_tex < 1:
        return np.array([]), np.array([])

    if p.N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_c = -1.0 + margin_Z + np.linspace(0, usable_Z, p.N_Z_tex)

    pg, zg = np.meshgrid(phi_c, Z_c)
    return pg.ravel(), zg.ravel()


def validate_geometry(a_dim, b_dim, zone, N_phi_tex, N_Z_tex,
                      spacing_factor, R, L, d_phi, d_Z):
    """Проверки геометрии. Возвращает (ok, reason)."""
    B = b_dim / R
    A = 2 * a_dim / L

    # Узлов на лунку
    nodes_phi = 2 * B / d_phi
    nodes_Z = 2 * A / d_Z
    if nodes_phi < MIN_NODES_PER_DIMPLE:
        return False, f"nodes_phi={nodes_phi:.1f}<{MIN_NODES_PER_DIMPLE}"
    if nodes_Z < MIN_NODES_PER_DIMPLE:
        return False, f"nodes_Z={nodes_Z:.1f}<{MIN_NODES_PER_DIMPLE}"

    if N_phi_tex < 1 or N_Z_tex < 1:
        return False, "N_tex<1"

    return True, ""


def solve_one(H, d_phi, d_Z, eta, alpha_pv, phi_1D, Z_1D, Phi_mesh):
    """Один расчёт, возвращает (W, f, pmax, cav, n_iter, warmup_ok, dt)."""
    p = base_params
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        _, W, f, _, _, pmax, _, n_out, _, cav = solve_and_compute(
            H, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
            phi_1D, Z_1D, Phi_mesh,
            cavitation=CAVITATION, alpha_pv=alpha_pv)
        dt = time.time() - t0
        warmup_ok = not any("warmup" in str(w.message) for w in caught)
    return W, f, pmax, cav, n_out, warmup_ok, dt


# ===================================================================
#  ЭТАП A
# ===================================================================

def run_stage_a(out_dir):
    print(f"\n{'=' * 100}")
    print(f"ЭТАП A: ГРУБЫЙ SCREENING ({A_NPHI}×{A_NZ})")
    print(f"ε = {A_EPS_VALUES}, масло = минеральное + PV")
    print(f"{'=' * 100}")

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(A_NPHI, A_NZ)
    p_base = make_base_params()
    eta = MINERAL_OIL["eta_pump"]
    alpha_pv = MINERAL_OIL["alpha_pv"]

    # Кэш W_smooth
    smooth_cache = {}
    for eps in A_EPS_VALUES:
        H_s = make_H(eps, Phi_mesh, Z_mesh, p_base, textured=False)
        W_s, f_s, pmax_s, cav_s, _, wup_s, dt_s = solve_one(
            H_s, d_phi, d_Z, eta, alpha_pv, phi_1D, Z_1D, Phi_mesh)
        smooth_cache[eps] = {"W": W_s, "f": f_s, "pmax": pmax_s,
                             "cav": cav_s, "warmup_ok": wup_s}
        print(f"  Smooth ε={eps:.1f}: W={W_s:.0f}, f={f_s:.4f}, "
              f"warmup={'OK' if wup_s else 'FAIL'} ({dt_s:.1f}с)")

    # Генерация вариантов
    all_variants = []
    for hp in HP_VALUES:
        for zone in ZONES:
            for (a, b) in DIMPLE_SIZES:
                for sf in SPACING_FACTORS:
                    Nt_phi, Nt_Z = compute_N_tex(zone, a, b,
                                                  p_base.R, p_base.L, sf)
                    ok, reason = validate_geometry(
                        a, b, zone, Nt_phi, Nt_Z, sf,
                        p_base.R, p_base.L, d_phi, d_Z)
                    all_variants.append({
                        "hp": hp, "zone": zone, "a": a, "b": b,
                        "sf": sf, "Nt_phi": Nt_phi, "Nt_Z": Nt_Z,
                        "valid": ok, "skip_reason": reason,
                    })

    valid = [v for v in all_variants if v["valid"]]
    skipped = [v for v in all_variants if not v["valid"]]
    total_calcs = len(valid) * len(A_EPS_VALUES)
    print(f"\n  Всего комбинаций: {len(all_variants)}, "
          f"валидных: {len(valid)}, пропущено: {len(skipped)}")
    print(f"  Расчётов: {total_calcs}")
    print(f"  Оценка: ~{total_calcs * 3 / 60:.0f} мин\n")

    # Прогон
    results = []
    done = 0
    t0_all = time.time()

    for iv, v in enumerate(valid):
        p = make_base_params()
        p.a_dim = v["a"]
        p.b_dim = v["b"]
        p.h_p = v["hp"]
        p.phi_start_deg = v["zone"][0]
        p.phi_end_deg = v["zone"][1]
        p.N_phi_tex = v["Nt_phi"]
        p.N_Z_tex = v["Nt_Z"]

        phi_c, Z_c = setup_texture_custom(p)
        if len(phi_c) == 0:
            continue

        n_dimples = len(phi_c)
        A_nd = 2 * p.a_dim / p.L
        B_nd = p.b_dim / p.R
        H_p = p.h_p / p.c

        for eps in A_EPS_VALUES:
            H0 = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
            H_tex = create_H_with_ellipsoidal_depressions(
                H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A_nd, B_nd,
                profile=PROFILE)

            W_t, f_t, pmax_t, cav_t, n_it, wup, dt = solve_one(
                H_tex, d_phi, d_Z, eta, alpha_pv, phi_1D, Z_1D, Phi_mesh)

            sc = smooth_cache[eps]
            gw = W_t / sc["W"] if sc["W"] > 0 else 0
            gf = f_t / sc["f"] if sc["f"] > 0 else 0
            gp = pmax_t / sc["pmax"] if sc["pmax"] > 0 else 0

            results.append({
                "hp_um": v["hp"] * 1e6,
                "zone": f"{v['zone'][0]}-{v['zone'][1]}",
                "a_mm": v["a"] * 1e3, "b_mm": v["b"] * 1e3,
                "sf": v["sf"],
                "n_dimples": n_dimples,
                "eps": eps,
                "W_smooth": sc["W"], "W_tex": W_t,
                "gain_W": gw, "gain_f": gf, "gain_pmax": gp,
                "pmax_MPa": pmax_t / 1e6,
                "cav": cav_t, "n_iter": n_it,
                "warmup_ok": wup, "time": dt,
            })
            done += 1

        if (iv + 1) % 50 == 0 or iv == len(valid) - 1:
            elapsed = time.time() - t0_all
            rate = done / max(elapsed, 1)
            remain = (total_calcs - done) / max(rate, 0.01)
            print(f"  [{iv+1}/{len(valid)}] {done}/{total_calcs} расчётов, "
                  f"{elapsed:.0f}с, ~{remain:.0f}с осталось")

    dt_all = time.time() - t0_all
    print(f"\n  Этап A завершён: {done} расчётов за {dt_all:.0f} с "
          f"({dt_all/60:.1f} мин)")

    # Сохранить полную таблицу
    csv_path = os.path.join(out_dir, "screening_full.csv")
    fieldnames = ["hp_um", "zone", "a_mm", "b_mm", "sf", "n_dimples",
                  "eps", "W_smooth", "W_tex", "gain_W", "gain_f",
                  "gain_pmax", "pmax_MPa", "cav", "n_iter",
                  "warmup_ok", "time"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            for k in ["W_smooth", "W_tex", "pmax_MPa", "time"]:
                row[k] = f"{r[k]:.1f}"
            for k in ["gain_W", "gain_f", "gain_pmax", "cav"]:
                row[k] = f"{r[k]:.4f}"
            row["hp_um"] = f"{r['hp_um']:.0f}"
            row["a_mm"] = f"{r['a_mm']:.1f}"
            row["b_mm"] = f"{r['b_mm']:.1f}"
            row["sf"] = f"{r['sf']:.1f}"
            w.writerow(row)
    print(f"  CSV: {csv_path}")

    # Скиппы
    if skipped:
        skip_path = os.path.join(out_dir, "screening_skipped.csv")
        with open(skip_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hp_um", "zone", "a_mm", "b_mm", "sf", "reason"])
            for v in skipped:
                w.writerow([f"{v['hp']*1e6:.0f}",
                            f"{v['zone'][0]}-{v['zone'][1]}",
                            f"{v['a']*1e3:.1f}", f"{v['b']*1e3:.1f}",
                            f"{v['sf']:.1f}", v["skip_reason"]])
        print(f"  Skipped: {skip_path}")

    # Топ-30 для каждого ε
    for eps in A_EPS_VALUES:
        eps_results = [r for r in results if r["eps"] == eps]
        eps_results.sort(key=lambda x: x["gain_W"], reverse=True)
        top30 = eps_results[:30]

        eps_tag = str(eps).replace(".", "")
        top_path = os.path.join(out_dir, f"screening_top30_eps{eps_tag}.csv")
        with open(top_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in top30:
                row = {k: r[k] for k in fieldnames}
                for k in ["W_smooth", "W_tex", "pmax_MPa", "time"]:
                    row[k] = f"{r[k]:.1f}"
                for k in ["gain_W", "gain_f", "gain_pmax", "cav"]:
                    row[k] = f"{r[k]:.4f}"
                row["hp_um"] = f"{r['hp_um']:.0f}"
                row["a_mm"] = f"{r['a_mm']:.1f}"
                row["b_mm"] = f"{r['b_mm']:.1f}"
                row["sf"] = f"{r['sf']:.1f}"
                w.writerow(row)

        # Печать топ-10
        print(f"\n  ТОП-10 при ε={eps}:")
        print(f"  {'#':>3} {'hp':>4} {'зона':>9} {'a/b мм':>8} {'sf':>4} "
              f"{'N':>5} {'gain_W':>8} {'gain_f':>8} {'gpm':>6} "
              f"{'wup':>4}")
        print("  " + "-" * 80)
        for i, r in enumerate(top30[:10]):
            wup = "OK" if r["warmup_ok"] else "[!]"
            marker = " <<<" if r["gain_W"] > 1.0 else ""
            print(f"  {i+1:3d} {r['hp_um']:4.0f} {r['zone']:>9s} "
                  f"{r['a_mm']:.1f}/{r['b_mm']:.1f}  {r['sf']:4.1f} "
                  f"{r['n_dimples']:5d} {r['gain_W']:8.4f} "
                  f"{r['gain_f']:8.4f} {r['gain_pmax']:6.3f} "
                  f"{wup:>4s}{marker}")

    return results


# ===================================================================
#  ЭТАП B
# ===================================================================

def run_stage_b(stage_a_results, out_dir):
    print(f"\n{'=' * 100}")
    print(f"ЭТАП B: ПЕРЕПРОВЕРКА НА ТОНКОЙ СЕТКЕ ({B_NPHI}×{B_NZ})")
    print(f"{'=' * 100}")

    # Отбор кандидатов
    reliable = [r for r in stage_a_results if r["warmup_ok"]]
    reliable.sort(key=lambda x: x["gain_W"], reverse=True)

    # Топ-20 + все с gain_W > 0.995
    top20 = reliable[:20]
    extra = [r for r in reliable[20:] if r["gain_W"] > 0.995]
    candidates_raw = top20 + extra

    # Дедупликация: одна запись на уникальную комбинацию параметров
    seen = {}
    for r in candidates_raw:
        key = (r["hp_um"], r["zone"], r["a_mm"], r["b_mm"], r["sf"])
        if key not in seen or r["gain_W"] > seen[key]["gain_W"]:
            seen[key] = r
    candidates = list(seen.values())
    candidates.sort(key=lambda x: x["gain_W"], reverse=True)

    print(f"  Кандидатов: {len(candidates)}")

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(B_NPHI, B_NZ)
    p_base = make_base_params()

    oils = [("mineral", "Мин.", MINERAL_OIL),
            ("rapeseed", "Рапс.", RAPESEED_OIL)]
    modes = [("pv", True), ("nopv", False)]

    # Кэш smooth
    smooth_cache = {}
    for oil_key, oil_name, oil in oils:
        for mode_key, use_pv in modes:
            eta = oil["eta_pump"]
            alpha_pv = oil["alpha_pv"] if use_pv else None
            for eps in A_EPS_VALUES:
                H_s = make_H(eps, Phi_mesh, Z_mesh, p_base, textured=False)
                W_s, f_s, pmax_s, _, _, wup_s, dt_s = solve_one(
                    H_s, d_phi, d_Z, eta, alpha_pv,
                    phi_1D, Z_1D, Phi_mesh)
                cache_key = (oil_key, mode_key, eps)
                smooth_cache[cache_key] = {"W": W_s, "f": f_s, "pmax": pmax_s}
                print(f"  Smooth {oil_name} {mode_key} ε={eps:.1f}: "
                      f"W={W_s:.0f} ({dt_s:.1f}с)")

    # Прогон кандидатов
    b_results = []
    for ic, cand in enumerate(candidates):
        hp = cand["hp_um"] * 1e-6
        zone_parts = cand["zone"].split("-")
        zone = (int(zone_parts[0]), int(zone_parts[1]))
        a_dim = cand["a_mm"] * 1e-3
        b_dim = cand["b_mm"] * 1e-3
        sf = cand["sf"]
        eps = cand["eps"]

        Nt_phi, Nt_Z = compute_N_tex(zone, a_dim, b_dim,
                                       p_base.R, p_base.L, sf)
        p = make_base_params()
        p.a_dim, p.b_dim, p.h_p = a_dim, b_dim, hp
        p.phi_start_deg, p.phi_end_deg = zone
        p.N_phi_tex, p.N_Z_tex = Nt_phi, Nt_Z

        phi_c, Z_c = setup_texture_custom(p)
        if len(phi_c) == 0:
            continue

        A_nd = 2 * a_dim / p.L
        B_nd = b_dim / p.R
        H_p = hp / p.c
        n_dimples = len(phi_c)

        row = {"hp_um": cand["hp_um"], "zone": cand["zone"],
               "a_mm": cand["a_mm"], "b_mm": cand["b_mm"],
               "sf": sf, "n_dimples": n_dimples, "eps": eps,
               "gain_W_a": cand["gain_W"]}

        print(f"\n  [{ic+1}/{len(candidates)}] hp={cand['hp_um']:.0f} "
              f"zone={cand['zone']} a/b={cand['a_mm']:.1f}/{cand['b_mm']:.1f} "
              f"sf={sf} ε={eps}")

        for oil_key, oil_name, oil in oils:
            eta = oil["eta_pump"]
            for mode_key, use_pv in modes:
                alpha_pv = oil["alpha_pv"] if use_pv else None

                H0 = make_H(eps, Phi_mesh, Z_mesh, p, textured=False)
                H_tex = create_H_with_ellipsoidal_depressions(
                    H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A_nd, B_nd,
                    profile=PROFILE)

                W_t, f_t, pmax_t, cav_t, n_it, wup, dt = solve_one(
                    H_tex, d_phi, d_Z, eta, alpha_pv,
                    phi_1D, Z_1D, Phi_mesh)

                sc = smooth_cache[(oil_key, mode_key, eps)]
                gw = W_t / sc["W"] if sc["W"] > 0 else 0
                gf = f_t / sc["f"] if sc["f"] > 0 else 0

                col = f"gW_{oil_key[:3]}_{mode_key}"
                row[col] = gw
                row[f"gf_{oil_key[:3]}_{mode_key}"] = gf
                row[f"wup_{oil_key[:3]}_{mode_key}"] = wup

                tag = f"{oil_name} {mode_key}"
                wup_s = "OK" if wup else "[!]"
                marker = " <<<" if gw > 1.0 else ""
                print(f"    {tag:>15s}: gW={gw:.4f} gf={gf:.4f} "
                      f"wup={wup_s} ({dt:.1f}с){marker}")

        # Δgain_W
        for okey in ["min", "rap"]:
            pv_key = f"gW_{okey}_pv"
            nopv_key = f"gW_{okey}_nopv"
            if pv_key in row and nopv_key in row:
                row[f"delta_gW_{okey}"] = row[pv_key] - row[nopv_key]

        b_results.append(row)

    # CSV
    if b_results:
        csv_path = os.path.join(out_dir, "refined_top20.csv")
        fieldnames = list(b_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in b_results:
                row = {}
                for k, v in r.items():
                    if isinstance(v, float):
                        row[k] = f"{v:.4f}"
                    else:
                        row[k] = v
                w.writerow(row)
        print(f"\n  CSV: {csv_path}")

    # Итог
    print(f"\n{'=' * 100}")
    any_positive = any(
        r.get("gW_min_pv", 0) > 1.0 or r.get("gW_rap_pv", 0) > 1.0
        for r in b_results)
    if any_positive:
        print("ЕСТЬ кандидаты с gain_W > 1.0 на тонкой сетке!")
    else:
        print("Ни один кандидат НЕ дал gain_W > 1.0 на тонкой сетке.")
    print(f"{'=' * 100}")

    return b_results


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Параметрический поиск текстуры PV+PS")
    parser.add_argument("--stage", type=str, default="AB",
                        help="A, B, или AB (default: AB)")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Загрузить результаты этапа A из CSV для этапа B")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "results", "pump_pv_ps")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 100)
    print("РАСШИРЕННЫЙ ПАРАМЕТРИЧЕСКИЙ ПОИСК ТЕКСТУРЫ — PV+PS")
    print(f"Результаты → {out_dir}")
    print("=" * 100)

    stage_a_results = None

    if "A" in args.stage.upper():
        stage_a_results = run_stage_a(out_dir)

    if "B" in args.stage.upper():
        if stage_a_results is None and args.from_csv:
            # Загрузить из CSV
            csv_path = args.from_csv
            stage_a_results = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r = {}
                    for k, v in row.items():
                        try:
                            r[k] = float(v)
                        except (ValueError, TypeError):
                            r[k] = v
                    # Приведение типов
                    r["warmup_ok"] = str(r.get("warmup_ok", "True")) == "True"
                    stage_a_results.append(r)
            print(f"  Загружено {len(stage_a_results)} записей из {csv_path}")

        if stage_a_results is None:
            print("Нет данных этапа A. Запустите --stage A или укажите --from-csv")
            sys.exit(1)

        run_stage_b(stage_a_results, out_dir)


if __name__ == "__main__":
    main()
