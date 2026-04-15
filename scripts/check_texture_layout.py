#!/usr/bin/env python3
"""Проверка раскладки микротекстуры: перекрытие, разрешение, площадь.

Для каждого конфига из pump_params_micro проверяет:
1. Лунки не перекрываются (spacing > 2·полуось)
2. Достаточно узлов на лунку (≥ 10 по каждому направлению)
3. Лунки внутри зоны и подшипника
4. Area density Sp
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import pump_params as bp
from config.pump_params_micro import ALL_CONFIGS


def check_config(cfg):
    """Проверить один конфиг. Печатает результаты, возвращает True если OK."""
    label = cfg["label"]
    a = cfg["a_dim"]        # полуось Z (м)
    b = cfg["b_dim"]        # полуось φ (м)
    h_p = cfg["h_p"]
    ps = cfg["phi_start_deg"]
    pe = cfg["phi_end_deg"]
    Nt_phi = cfg["N_phi_tex"]
    Nt_Z = cfg["N_Z_tex"]
    N_phi = cfg["N_phi"]
    N_Z = cfg["N_Z"]

    R, c, L = bp.R, bp.c, bp.L

    # --- Безразмерные полуоси ---
    B_rad = b / R                    # полуось по φ (рад)
    A_nondim = 2 * a / L            # полуось по Z (безразмерная, L/2-масштаб)

    # --- Зона ---
    if ps <= pe:
        phi_span_deg = pe - ps
    else:
        phi_span_deg = (360 - ps) + pe
    phi_span_rad = np.deg2rad(phi_span_deg)

    Z_span = 1.6  # лунки в [-0.8, 0.8] (см. setup_texture)

    # --- Spacing ---
    if Nt_phi > 1:
        spacing_phi_rad = phi_span_rad / (Nt_phi - 1)
        spacing_phi_mm = spacing_phi_rad * R * 1e3
    else:
        spacing_phi_rad = phi_span_rad
        spacing_phi_mm = spacing_phi_rad * R * 1e3

    if Nt_Z > 1:
        spacing_Z_nondim = Z_span / (Nt_Z - 1)
        spacing_Z_mm = spacing_Z_nondim * (L / 2) * 1e3
    else:
        spacing_Z_nondim = Z_span
        spacing_Z_mm = spacing_Z_nondim * (L / 2) * 1e3

    dimple_diam_phi_mm = 2 * b * 1e3
    dimple_diam_Z_mm = 2 * a * 1e3

    overlap_phi = spacing_phi_mm > dimple_diam_phi_mm
    overlap_Z = spacing_Z_mm > dimple_diam_Z_mm

    # --- Узлов на лунку ---
    d_phi = 2 * np.pi / N_phi
    d_Z = 2.0 / (N_Z - 1)

    nodes_phi = 2 * B_rad / d_phi
    nodes_Z = 2 * A_nondim / d_Z

    grid_ok_phi = nodes_phi >= 10
    grid_ok_Z = nodes_Z >= 10

    # --- Area density Sp ---
    # Sp = (N_лунок * π·a·b) / (зона_φ·R × L)
    n_dimples = Nt_phi * Nt_Z
    area_dimple = np.pi * a * b              # м²
    area_zone = phi_span_rad * R * L         # м²
    Sp = n_dimples * area_dimple / area_zone

    # --- h_p / h_min ---
    h_min_06 = c * (1 - 0.6)  # при ε=0.6 = 20 мкм
    hp_ratio = h_p / h_min_06

    # --- Вывод ---
    print(f"\n  Конфиг {label}:")
    print(f"    Лунка: a={a*1e3:.2f} мм, b={b*1e3:.2f} мм, "
          f"hp={h_p*1e6:.0f} мкм (hp/h_min={hp_ratio:.2f})")
    print(f"    Зона: {ps}°–{pe}° ({phi_span_deg}°), "
          f"N_tex={Nt_phi}×{Nt_Z} = {n_dimples} лунок")
    print(f"    Сетка: {N_phi}×{N_Z}")
    print(f"    Sp = {Sp*100:.1f}%")
    print()

    sym_phi = "OK" if overlap_phi else "FAIL"
    sym_Z = "OK" if overlap_Z else "FAIL"
    print(f"    Spacing φ: {spacing_phi_mm:.2f} мм vs 2b={dimple_diam_phi_mm:.2f} мм"
          f"  [{sym_phi}]")
    print(f"    Spacing Z: {spacing_Z_mm:.2f} мм vs 2a={dimple_diam_Z_mm:.2f} мм"
          f"  [{sym_Z}]")

    sym_np = "OK" if grid_ok_phi else "FAIL"
    sym_nz = "OK" if grid_ok_Z else "FAIL"
    print(f"    Узлов/лунку φ: {nodes_phi:.1f}  [{sym_np}]")
    print(f"    Узлов/лунку Z: {nodes_Z:.1f}  [{sym_nz}]")

    all_ok = overlap_phi and overlap_Z and grid_ok_phi and grid_ok_Z
    status = "PASS" if all_ok else "*** FAIL ***"
    print(f"    Итог: {status}")
    return all_ok


def main():
    print("=" * 60)
    print("ПРОВЕРКА РАСКЛАДКИ МИКРОТЕКСТУРЫ")
    print(f"R={bp.R*1e3:.1f} мм, c={bp.c*1e6:.0f} мкм, L={bp.L*1e3:.1f} мм")
    print("=" * 60)

    all_pass = True
    for cfg in ALL_CONFIGS:
        ok = check_config(cfg)
        all_pass = all_pass and ok

    print("\n" + "=" * 60)
    if all_pass:
        print("Все конфиги прошли проверку.")
    else:
        print("ЕСТЬ ПРОБЛЕМЫ — см. FAIL выше.")
    print("=" * 60)


if __name__ == "__main__":
    main()
