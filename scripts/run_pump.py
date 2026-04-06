#!/usr/bin/env python3
"""Запуск стационарного расчёта подшипника центробежного насоса.

Sensitivity sweep по h_p = 15, 30, 45, 60 мкм.
Основные графики для рекомендуемого h_p.
Сводная таблица gain (текстура/гладкий) при ε = 0.6.
"""
import sys
import os
import time
import csv
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models.pump_steady as pump_steady
from models.pump_steady import run_pump_analysis, CONFIGS, EPSILON_VALUES
from config import pump_params as params


def make_results_dir():
    hp_um = int(params.h_p * 1e6)
    tag = datetime.now().strftime("%y%m%d_%H%M")
    name = (f"{tag}_{pump_steady.N_PHI}x{pump_steady.N_Z}"
            f"_hp{hp_um}_sigma{params.sigma*1e6:.1f}um")
    d = os.path.join(os.path.dirname(__file__), "..", "results", "pump", name)
    os.makedirs(d, exist_ok=True)
    return d

RESULTS_DIR = None  # устанавливается в main() после --grid

H_P_VALUES_UM = [15, 30, 45, 60]  # мкм
EPS_REF = 0.6  # ε для сводной таблицы и sensitivity
EPS_STABLE_MIN = 0.70  # минимальный порог стабильности для рекомендации


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


def plot_sensitivity(h_p_vals, data_mineral, data_rapeseed, ylabel, filename, title):
    """Построить sensitivity-график: 2 кривые (минеральное, рапсовое) от h_p."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(h_p_vals, data_mineral, "o-", color="blue", linewidth=2,
            markersize=8, label="Минеральное")
    ax.plot(h_p_vals, data_rapeseed, "s--", color="green", linewidth=2,
            markersize=8, label="Рапсовое")
    ax.set_xlabel("Глубина текстуры h_p (мкм)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Сохранён: {filename}")


def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--nphi", type=int, default=800,
                        help="Узлов по φ (default: 800)")
    parser.add_argument("--nz", type=int, default=200,
                        help="Узлов по Z (default: 200)")
    parser.add_argument("--hp", type=float, default=None,
                        help="Одна глубина для теста (мкм). Без — sweep [15,30,45,60]")
    parser.add_argument("--plot-only", action="store_true",
                        help="Загрузить data.npz и перестроить графики без расчёта")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Путь к папке с data.npz для --plot-only")
    args = parser.parse_args()

    pump_steady.N_PHI = args.nphi
    pump_steady.N_Z = args.nz

    if args.hp is not None:
        H_P_VALUES_UM = [int(args.hp)]

    if args.plot_only:
        RESULTS_DIR = args.data_dir or os.path.join(
            os.path.dirname(__file__), "..", "results", "pump")
        print("=" * 60)
        print("РАСЧЁТ ПОДШИПНИКА ЦЕНТРОБЕЖНОГО НАСОСА — PLOT ONLY")
        print(f"Загрузка из {RESULTS_DIR}")
        print("=" * 60)

        d = np.load(os.path.join(RESULTS_DIR, "data.npz"), allow_pickle=True)
        eps = d["epsilon"]
        best_hp = int(d["recommended_hp_um"])

        # Восстановить all_results из data.npz
        all_results = {}
        for h_p_um in H_P_VALUES_UM:
            prefix = f"hp{h_p_um}_"
            r = {"epsilon": eps}
            for key in ["W", "f", "hmin", "Q", "F_tr", "N_loss", "pmax"]:
                r[key] = d[prefix + key]
            all_results[h_p_um] = r
    else:
        RESULTS_DIR = make_results_dir()

        print("=" * 60)
        print("РАСЧЁТ ПОДШИПНИКА ЦЕНТРОБЕЖНОГО НАСОСА")
        print(f"Sensitivity sweep: h_p = {H_P_VALUES_UM} мкм")
        print(f"Сетка: {args.nphi}×{args.nz}, σ = {params.sigma*1e6:.1f} мкм")
        print(f"Результаты → {RESULTS_DIR}")
        print("=" * 60)

    # Индекс ε ближайший к EPS_REF
    ie_ref = np.argmin(np.abs(EPSILON_VALUES - EPS_REF))
    eps_ref_actual = EPSILON_VALUES[ie_ref]
    print(f"Опорный эксцентриситет: ε = {eps_ref_actual:.2f}")

    if not args.plot_only:
        # --- Sweep по h_p ---
        all_results = {}
        t0_total = time.time()

        for h_p_um in H_P_VALUES_UM:
            h_p_m = h_p_um * 1e-6
            print(f"\n{'='*40}")
            print(f"h_p = {h_p_um} мкм")
            print(f"{'='*40}")

            t0 = time.time()
            results = run_pump_analysis(h_p_override=h_p_m)
            dt = time.time() - t0
            print(f"Время: {dt:.1f} с")
            all_results[h_p_um] = results

        dt_total = time.time() - t0_total
        print(f"\nОбщее время sweep: {dt_total:.1f} с")

    # --- Сводная таблица при ε = EPS_REF ---
    # Индексы конфигураций: 0=smooth+mineral, 1=smooth+rapeseed,
    #                       2=texture+mineral, 3=texture+rapeseed
    print(f"\n{'='*60}")
    print(f"СВОДНАЯ ТАБЛИЦА при ε = {eps_ref_actual:.2f}")
    print(f"{'='*60}")

    def gain(a, b):
        if np.isnan(a) or np.isnan(b) or b <= 0:
            return np.nan
        return a / b

    def max_valid_eps(r, cfg_idx):
        """Максимальный ε, при котором конфигурация сошлась (не NaN)."""
        W_row = r["W"][cfg_idx]
        valid = ~np.isnan(W_row)
        if not np.any(valid):
            return 0.0
        return EPSILON_VALUES[np.where(valid)[0][-1]]

    table_rows = []
    gain_rows = []
    stability = {}  # h_p -> max ε среди текстурированных

    for h_p_um in H_P_VALUES_UM:
        r = all_results[h_p_um]
        # Стабильность: минимум из двух текстурированных конфигов
        eps_stable_min = min(max_valid_eps(r, 2), max_valid_eps(r, 3))
        eps_stable_rap = max_valid_eps(r, 3)
        stability[h_p_um] = eps_stable_min
        print(f"  h_p={h_p_um}: стабильно до ε = {eps_stable_min:.2f}")

        for oil_name, i_smooth, i_tex in [("Минеральное", 0, 2),
                                           ("Рапсовое", 1, 3)]:
            row = {
                "h_p_um": h_p_um, "oil": oil_name,
                "W_smooth": r["W"][i_smooth, ie_ref],
                "W_tex": r["W"][i_tex, ie_ref],
                "f_smooth": r["f"][i_smooth, ie_ref],
                "f_tex": r["f"][i_tex, ie_ref],
                "Ftr_smooth": r["F_tr"][i_smooth, ie_ref],
                "Ftr_tex": r["F_tr"][i_tex, ie_ref],
                "Nloss_smooth": r["N_loss"][i_smooth, ie_ref],
                "Nloss_tex": r["N_loss"][i_tex, ie_ref],
                "pmax_smooth": r["pmax"][i_smooth, ie_ref],
                "pmax_tex": r["pmax"][i_tex, ie_ref],
                "Q_smooth": r["Q"][i_smooth, ie_ref],
                "Q_tex": r["Q"][i_tex, ie_ref],
            }
            table_rows.append(row)

            gain_rows.append({
                "h_p_um": h_p_um, "oil": oil_name,
                "W_gain": gain(row["W_tex"], row["W_smooth"]),
                "f_gain": gain(row["f_tex"], row["f_smooth"]),
                "Ftr_gain": gain(row["Ftr_tex"], row["Ftr_smooth"]),
                "Nloss_gain": gain(row["Nloss_tex"], row["Nloss_smooth"]),
                "pmax_gain": gain(row["pmax_tex"], row["pmax_smooth"]),
                "Q_gain": gain(row["Q_tex"], row["Q_smooth"]),
            })

    # Печать в терминал
    print(f"\n{'Абсолютные значения (текстура)':}")
    print(f"{'h_p':>6} {'масло':>12} {'W(Н)':>8} {'f':>8} {'F_tr(Н)':>8} "
          f"{'N_loss(Вт)':>10} {'p_max(МПа)':>10} {'Q(см³/с)':>10} {'ε_max':>6}")
    for row in table_rows:
        hp = row['h_p_um']
        eps_max = stability[hp]
        if np.isnan(row['W_tex']):
            print(f"{hp:>6} {row['oil']:>12}    — расходимость при ε={eps_ref_actual:.2f} —")
        else:
            print(f"{hp:>6} {row['oil']:>12} "
                  f"{row['W_tex']:>8.0f} {row['f_tex']:>8.4f} {row['Ftr_tex']:>8.1f} "
                  f"{row['Nloss_tex']:>10.0f} {row['pmax_tex']/1e6:>10.1f} "
                  f"{row['Q_tex']*1e6:>10.3f} {eps_max:>6.2f}")

    print(f"\n{'Gain (текстура / гладкий)':}")
    print(f"{'h_p':>6} {'масло':>12} {'W':>6} {'f':>6} {'F_tr':>6} "
          f"{'N_loss':>6} {'p_max':>6} {'Q':>6} {'ε_max':>6}")
    for g in gain_rows:
        hp = g['h_p_um']
        eps_max = stability[hp]
        if np.isnan(g['W_gain']):
            print(f"{hp:>6} {g['oil']:>12}    — расходимость —")
        else:
            print(f"{hp:>6} {g['oil']:>12} "
                  f"{g['W_gain']:>6.2f} {g['f_gain']:>6.2f} {g['Ftr_gain']:>6.2f} "
                  f"{g['Nloss_gain']:>6.2f} {g['pmax_gain']:>6.2f} {g['Q_gain']:>6.2f} "
                  f"{eps_max:>6.2f}")

    # CSV
    def csv_val(val, fmt):
        return fmt % val if not np.isnan(val) else "NaN"

    csv_path = os.path.join(RESULTS_DIR, "gain_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["h_p_um", "oil", "eps_max",
                         "W_smooth", "W_tex", "W_gain",
                         "f_smooth", "f_tex", "f_gain",
                         "Ftr_smooth", "Ftr_tex", "Ftr_gain",
                         "Nloss_smooth", "Nloss_tex", "Nloss_gain",
                         "pmax_smooth_MPa", "pmax_tex_MPa", "pmax_gain",
                         "Q_smooth_cm3s", "Q_tex_cm3s", "Q_gain"])
        for row, g in zip(table_rows, gain_rows):
            writer.writerow([
                row["h_p_um"], row["oil"],
                f"{stability[row['h_p_um']]:.2f}",
                csv_val(row['W_smooth'], "%.1f"), csv_val(row['W_tex'], "%.1f"), csv_val(g['W_gain'], "%.3f"),
                csv_val(row['f_smooth'], "%.5f"), csv_val(row['f_tex'], "%.5f"), csv_val(g['f_gain'], "%.3f"),
                csv_val(row['Ftr_smooth'], "%.2f"), csv_val(row['Ftr_tex'], "%.2f"), csv_val(g['Ftr_gain'], "%.3f"),
                csv_val(row['Nloss_smooth'], "%.1f"), csv_val(row['Nloss_tex'], "%.1f"), csv_val(g['Nloss_gain'], "%.3f"),
                csv_val(row['pmax_smooth']/1e6, "%.2f") if not np.isnan(row['pmax_smooth']) else "NaN",
                csv_val(row['pmax_tex']/1e6, "%.2f") if not np.isnan(row['pmax_tex']) else "NaN",
                csv_val(g['pmax_gain'], "%.3f"),
                csv_val(row['Q_smooth']*1e6, "%.4f") if not np.isnan(row['Q_smooth']) else "NaN",
                csv_val(row['Q_tex']*1e6, "%.4f") if not np.isnan(row['Q_tex']) else "NaN",
                csv_val(g['Q_gain'], "%.3f"),
            ])
    print(f"\n  Сохранён: gain_table.csv")

    # TXT
    txt_path = os.path.join(RESULTS_DIR, "gain_table.txt")
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(f"Сводная таблица при ε = {eps_ref_actual:.2f}\n")
        ftxt.write("=" * 90 + "\n\n")

        ftxt.write("Часть 1: абсолютные значения (текстурированный подшипник)\n")
        ftxt.write(f"{'h_p(мкм)':>8} {'масло':>12} {'W(Н)':>8} {'f':>8} {'F_tr(Н)':>8} "
                   f"{'N_loss(Вт)':>10} {'p_max(МПа)':>10} {'Q(см³/с)':>10} {'ε_max':>6}\n")
        ftxt.write("-" * 98 + "\n")
        for row in table_rows:
            hp = row['h_p_um']
            eps_max = stability[hp]
            if np.isnan(row['W_tex']):
                ftxt.write(f"{hp:>8} {row['oil']:>12}    — расходимость —\n")
            else:
                ftxt.write(f"{hp:>8} {row['oil']:>12} "
                           f"{row['W_tex']:>8.0f} {row['f_tex']:>8.4f} {row['Ftr_tex']:>8.1f} "
                           f"{row['Nloss_tex']:>10.0f} {row['pmax_tex']/1e6:>10.1f} "
                           f"{row['Q_tex']*1e6:>10.3f} {eps_max:>6.2f}\n")

        ftxt.write("\n\nЧасть 2: gain (текстура / гладкий)\n")
        ftxt.write(f"{'h_p(мкм)':>8} {'масло':>12} {'W':>6} {'f':>6} {'F_tr':>6} "
                   f"{'N_loss':>6} {'p_max':>6} {'Q':>6} {'ε_max':>6}\n")
        ftxt.write("-" * 68 + "\n")
        for g in gain_rows:
            hp = g['h_p_um']
            eps_max = stability[hp]
            if np.isnan(g['W_gain']):
                ftxt.write(f"{hp:>8} {g['oil']:>12}    — расходимость —\n")
            else:
                ftxt.write(f"{hp:>8} {g['oil']:>12} "
                           f"{g['W_gain']:>6.2f} {g['f_gain']:>6.2f} {g['Ftr_gain']:>6.2f} "
                           f"{g['Nloss_gain']:>6.2f} {g['pmax_gain']:>6.2f} {g['Q_gain']:>6.2f} "
                           f"{eps_max:>6.2f}\n")
    print(f"  Сохранён: gain_table.txt")

    # --- Sensitivity plots (при ε = EPS_REF) ---
    sens_data = {"W_min": [], "W_rap": [], "f_min": [], "f_rap": [],
                 "pmax_min": [], "pmax_rap": [], "Nloss_min": [], "Nloss_rap": []}

    for h_p_um in H_P_VALUES_UM:
        r = all_results[h_p_um]
        # Текстура: индексы 2 (минеральное), 3 (рапсовое)
        sens_data["W_min"].append(r["W"][2, ie_ref])
        sens_data["W_rap"].append(r["W"][3, ie_ref])
        sens_data["f_min"].append(r["f"][2, ie_ref])
        sens_data["f_rap"].append(r["f"][3, ie_ref])
        sens_data["pmax_min"].append(r["pmax"][2, ie_ref] / 1e6)
        sens_data["pmax_rap"].append(r["pmax"][3, ie_ref] / 1e6)
        sens_data["Nloss_min"].append(r["N_loss"][2, ie_ref])
        sens_data["Nloss_rap"].append(r["N_loss"][3, ie_ref])

    plot_sensitivity(H_P_VALUES_UM, sens_data["W_min"], sens_data["W_rap"],
                     "Несущая способность W (Н)", "sensitivity_W.png",
                     f"W(h_p) при ε = {eps_ref_actual:.2f} — текстура")
    plot_sensitivity(H_P_VALUES_UM, sens_data["f_min"], sens_data["f_rap"],
                     "Коэффициент трения f", "sensitivity_f.png",
                     f"f(h_p) при ε = {eps_ref_actual:.2f} — текстура")
    plot_sensitivity(H_P_VALUES_UM, sens_data["pmax_min"], sens_data["pmax_rap"],
                     "Максимальное давление p_max (МПа)", "sensitivity_pmax.png",
                     f"p_max(h_p) при ε = {eps_ref_actual:.2f} — текстура")
    plot_sensitivity(H_P_VALUES_UM, sens_data["Nloss_min"], sens_data["Nloss_rap"],
                     "Потери мощности N_loss (Вт)", "sensitivity_Nloss.png",
                     f"N_loss(h_p) при ε = {eps_ref_actual:.2f} — текстура")

    # --- Выбор рекомендуемого h_p ---
    # Критерий: W_gain × стабильность. Решение должно сходиться хотя бы
    # до ε = EPS_STABLE_MIN. Иначе — дисквалификация.
    best_hp = H_P_VALUES_UM[0]
    best_score = 0
    best_reason = ""

    for i, h_p_um in enumerate(H_P_VALUES_UM):
        g_min = gain_rows[2 * i]
        g_rap = gain_rows[2 * i + 1]

        # Пропускаем NaN gains (расходимость при ε_ref)
        if np.isnan(g_min["W_gain"]) or np.isnan(g_rap["W_gain"]):
            continue

        w_gain_avg = (g_min["W_gain"] + g_rap["W_gain"]) / 2
        pmax_gain_avg = (g_min["pmax_gain"] + g_rap["pmax_gain"]) / 2
        nloss_gain_avg = (g_min["Nloss_gain"] + g_rap["Nloss_gain"]) / 2
        eps_max = stability[h_p_um]

        # Дисквалификация: не сходится до EPS_STABLE_MIN
        if eps_max < EPS_STABLE_MIN:
            continue

        score = w_gain_avg
        if pmax_gain_avg > 6.0:
            score *= 0.3
        if nloss_gain_avg > 1.25:
            score *= 0.5

        if score > best_score:
            best_score = score
            best_hp = h_p_um
            best_reason = (f"W_gain={w_gain_avg:.2f}, "
                           f"pmax_gain={pmax_gain_avg:.2f}, "
                           f"Nloss_gain={nloss_gain_avg:.2f}, "
                           f"ε_max={eps_max:.2f}")

    print(f"\nРекомендуемый h_p = {best_hp} мкм ({best_reason})")

    # recommended_hp.txt
    rec_path = os.path.join(RESULTS_DIR, "recommended_hp.txt")
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write(f"Рекомендуемая глубина текстуры: h_p = {best_hp} мкм\n\n")
        f.write(f"Обоснование (при ε = {eps_ref_actual:.2f}):\n")
        f.write(f"  {best_reason}\n\n")
        f.write("Сравнение всех вариантов:\n")
        for i, h_p_um in enumerate(H_P_VALUES_UM):
            g_min = gain_rows[2 * i]
            g_rap = gain_rows[2 * i + 1]
            eps_max = stability[h_p_um]
            if np.isnan(g_min["W_gain"]) or np.isnan(g_rap["W_gain"]):
                f.write(f"  h_p={h_p_um:>2} мкм: расходимость при ε={eps_ref_actual:.2f}, "
                        f"ε_max={eps_max:.2f}\n")
                continue
            w_avg = (g_min["W_gain"] + g_rap["W_gain"]) / 2
            p_avg = (g_min["pmax_gain"] + g_rap["pmax_gain"]) / 2
            n_avg = (g_min["Nloss_gain"] + g_rap["Nloss_gain"]) / 2
            marker = " ← рекомендуемый" if h_p_um == best_hp else ""
            f.write(f"  h_p={h_p_um:>2} мкм: W_gain={w_avg:.2f}, "
                    f"pmax_gain={p_avg:.2f}, Nloss_gain={n_avg:.2f}, "
                    f"ε_max={eps_max:.2f}{marker}\n")
    print(f"  Сохранён: recommended_hp.txt")

    # --- Основные графики для рекомендуемого h_p ---
    print(f"\nОсновные графики для h_p = {best_hp} мкм:")
    r = all_results[best_hp]
    eps = r["epsilon"]

    plot_curves(eps, r["W"], "Несущая способность W (Н)",
                "W_vs_eps.png", f"W(ε) — насос (h_p={best_hp} мкм)")
    plot_curves(eps, r["f"], "Коэффициент трения f",
                "f_vs_eps.png", f"f(ε) — насос (h_p={best_hp} мкм)")
    plot_curves(eps, r["F_tr"], "Сила трения F_tr (Н)",
                "Ftr_vs_eps.png", f"F_tr(ε) — насос (h_p={best_hp} мкм)")
    plot_curves(eps, r["N_loss"], "Потери мощности N_loss (Вт)",
                "Nloss_vs_eps.png", f"N_loss(ε) — насос (h_p={best_hp} мкм)")
    plot_curves(eps, r["hmin"] * 1e6, "h_min (мкм)",
                "hmin_vs_eps.png",
                f"h_min(ε) — насос (h_min = c·(1−ε), не зависит от вязкости и текстуры)")
    plot_curves(eps, r["Q"] * 1e6, "Расход смазки Q (см³/с)",
                "Q_vs_eps.png", f"Q(ε) — насос (h_p={best_hp} мкм)")
    plot_curves(eps, r["pmax"] / 1e6, "Максимальное давление p_max (МПа)",
                "pmax_vs_eps.png", f"p_max(ε) — насос (h_p={best_hp} мкм)")

    # Сохранить все данные
    save_dict = {"epsilon": eps, "h_p_values_um": np.array(H_P_VALUES_UM),
                 "recommended_hp_um": best_hp,
                 "labels": [c["label"] for c in CONFIGS]}
    for h_p_um in H_P_VALUES_UM:
        r = all_results[h_p_um]
        prefix = f"hp{h_p_um}_"
        for key in ["W", "f", "hmin", "Q", "F_tr", "N_loss", "pmax"]:
            save_dict[prefix + key] = r[key]

    np.savez(os.path.join(RESULTS_DIR, "data.npz"), **save_dict)
    print(f"\nДанные сохранены в results/pump/data.npz")


if __name__ == "__main__":
    main()
