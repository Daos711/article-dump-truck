"""Квазистационарный анализ подшипника ДВС карьерного самосвала.

Методологическая оговорка: для каждого угла поворота коленвала φ решается
СТАЦИОНАРНАЯ задача Рейнольдса при эквивалентной внешней нагрузке F_ext(φ).
Squeeze-эффект (∂h/∂t), инерция вала и полная связка Reynolds + equation of
motion НЕ учитываются (квазистационарная аппроксимация).
"""
import numpy as np
from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
    DEFAULT_CLOSURE, DEFAULT_CAVITATION,
)
from config import diesel_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

N_GRID = 150                           # 150 для отладки, 300 для финала
PHI_CRANK = np.linspace(0, 720, 72)   # шаг 10° для отладки, 360 для финала

CONFIGS = [
    {"label": "Гладкий + минеральное", "textured": False,
     "oil": MINERAL_OIL, "color": "blue", "ls": "-"},
    {"label": "Гладкий + рапсовое", "textured": False,
     "oil": RAPESEED_OIL, "color": "blue", "ls": "--"},
    {"label": "Текстура + минеральное", "textured": True,
     "oil": MINERAL_OIL, "color": "red", "ls": "-"},
    {"label": "Текстура + рапсовое", "textured": True,
     "oil": RAPESEED_OIL, "color": "red", "ls": "--"},
]


def load_diesel(phi_deg, F_max=None):
    """Упрощённая индикаторная диаграмма 4-тактного ДВС.

    phi_deg : 0..720°. Пик при φ ≈ 370° (после ВМТ рабочего хода).
    """
    if F_max is None:
        F_max = params.F_max
    F_gas = F_max * np.exp(-((phi_deg - 370) ** 2) / (2 * 30 ** 2))
    F_inertia = 0.1 * F_max * np.cos(2 * np.deg2rad(phi_deg))
    F_total = F_gas + F_inertia + 0.05 * F_max
    return np.maximum(F_total, 20000.0)


def build_load_table(Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                     oil, textured=False, phi_c=None, Z_c=None,
                     closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Построить таблицу W(ε) для быстрой интерполяции.

    Returns
    -------
    eps_table : (M,)
    W_table   : (M,)
    """
    eta = oil["eta_diesel"]
    eps_table = np.concatenate([
        np.linspace(0.001, 0.05, 10),
        np.linspace(0.06, 0.98, 30),
    ])
    W_table = np.zeros(len(eps_table))
    P_prev = None

    for i, eps in enumerate(eps_table):
        H = make_H(eps, Phi_mesh, Z_mesh, params,
                   textured=textured, phi_c_flat=phi_c, Z_c_flat=Z_c)
        P, F, *_ = solve_and_compute(
            H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
            phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
            closure=closure, cavitation=cavitation,
        )
        W_table[i] = F
        P_prev = P

    print(f"    W_table: min={W_table.min()/1000:.1f} кН, max={W_table.max()/1000:.1f} кН")
    return eps_table, W_table


def find_epsilon_for_load(F_target, eps_table, W_table,
                          Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                          oil, textured=False, phi_c=None, Z_c=None,
                          closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Найти ε для заданной нагрузки: локализация по таблице + 12 шагов бисекции.

    Returns
    -------
    eps_mid, F_hyd, mu, Q, h_min, p_max, status
    """
    eta = oil["eta_diesel"]

    # Локализация корня по таблице
    status = "ok"
    if F_target <= W_table[0]:
        eps_lo, eps_hi = 0.001, eps_table[1]
        status = "below_range"
    elif F_target >= W_table[-1]:
        eps_lo, eps_hi = eps_table[-2], 0.98
        status = "above_range"
    else:
        j = np.searchsorted(W_table, F_target)
        eps_lo, eps_hi = eps_table[j - 1], eps_table[j]

    eps_mid = 0.5 * (eps_lo + eps_hi)
    F_hyd = mu = Qv = h_min = p_max = 0.0

    for _ in range(12):
        eps_mid = 0.5 * (eps_lo + eps_hi)
        H = make_H(eps_mid, Phi_mesh, Z_mesh, params,
                   textured=textured, phi_c_flat=phi_c, Z_c_flat=Z_c)
        _, F_hyd, mu, Qv, h_min, p_max = solve_and_compute(
            H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
            phi_1D, Z_1D, Phi_mesh,
            closure=closure, cavitation=cavitation,
        )
        if F_hyd > F_target:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return eps_mid, F_hyd, mu, Qv, h_min, p_max, status


def run_diesel_analysis(closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Выполнить квазистационарный расчёт ДВС для всех конфигураций.

    Returns
    -------
    results : dict
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_GRID)
    phi_c, Z_c = setup_texture(params)

    F_ext = load_diesel(PHI_CRANK)

    n_phi = len(PHI_CRANK)
    n_cfg = len(CONFIGS)

    eps_arr = np.zeros((n_cfg, n_phi))
    hmin_arr = np.zeros((n_cfg, n_phi))
    f_arr = np.zeros((n_cfg, n_phi))
    pmax_arr = np.zeros((n_cfg, n_phi))
    F_hyd_arr = np.zeros((n_cfg, n_phi))

    for ic, cfg in enumerate(CONFIGS):
        print(f"  [{ic+1}/{n_cfg}] {cfg['label']}...")

        # Шаг 1: таблица W(ε)
        print("    Построение таблицы W(ε)...")
        eps_table, W_table = build_load_table(
            Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
            oil=cfg["oil"], textured=cfg["textured"],
            phi_c=phi_c, Z_c=Z_c,
            closure=closure, cavitation=cavitation,
        )

        print(f"    F_ext:   min={F_ext.min()/1000:.1f} кН, max={F_ext.max()/1000:.1f} кН")

        # Шаг 2: по каждому углу — бисекция
        print("    Расчёт по углам коленвала...")
        n_below = n_above = 0
        for ip, phi_k in enumerate(PHI_CRANK):
            F_target = F_ext[ip]
            eps_mid, F_hyd, mu, Qv, h_min, p_max, st = find_epsilon_for_load(
                F_target, eps_table, W_table,
                Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                oil=cfg["oil"], textured=cfg["textured"],
                phi_c=phi_c, Z_c=Z_c,
                closure=closure, cavitation=cavitation,
            )
            if st == "below_range":
                n_below += 1
            elif st == "above_range":
                n_above += 1
            eps_arr[ic, ip] = eps_mid
            hmin_arr[ic, ip] = h_min
            f_arr[ic, ip] = mu
            pmax_arr[ic, ip] = p_max
            F_hyd_arr[ic, ip] = F_hyd
        print(f"    Вне диапазона: {n_below} ниже, {n_above} выше (из {n_phi})")

    return {
        "phi_crank": PHI_CRANK,
        "F_ext": F_ext,
        "epsilon": eps_arr,
        "hmin": hmin_arr,
        "f": f_arr,
        "pmax": pmax_arr,
        "F_hyd": F_hyd_arr,
        "configs": CONFIGS,
    }
