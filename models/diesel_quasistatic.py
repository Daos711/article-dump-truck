"""Квазистационарный анализ подшипника ДВС карьерного самосвала.

Методологическая оговорка: для каждого угла поворота коленвала φ решается
СТАЦИОНАРНАЯ задача Рейнольдса при эквивалентной внешней нагрузке F_ext(φ).
Squeeze-эффект (∂h/∂t), инерция вала и полная связка Reynolds + equation of
motion НЕ учитываются (квазистационарная аппроксимация).

Stage THD-0 (global_static) extension
-------------------------------------
``run_diesel_analysis(thermal=ThermalConfig(...))`` adds an outer per-angle
fixed-point iteration around the solver's ``global_static_target_C`` law:

    P_loss   = F_tr * U,   U = omega * R
    mdot     = max(rho * |Q|, mdot_floor)
    T_target = T_in + gamma * P_loss / (mdot * cp)

Convergence is on |T_new - T_guess| < tol_T_C with under-relaxation.

``thermal=None`` or ``thermal.mode == "off"`` keeps the legacy isothermal
behaviour exactly (eta = oil["eta_diesel"]); the new result arrays are
filled with consistent values so downstream code can read them
unconditionally.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
    DEFAULT_CLOSURE, DEFAULT_CAVITATION,
)
from models.thermal_coupling import (
    ThermalConfig,
    build_oil_walther,
    viscosity_at_T_C,
    global_static_step,
)
from config import diesel_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

N_GRID = 100                           # 100 для трендов, 300 для финала
PHI_CRANK = np.linspace(0, 720, 36)   # шаг 20° для трендов, 360 для финала

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


# Public registry mapping the short labels exposed in the CLI to entries
# in ``CONFIGS``. Lets the entry script accept e.g.
# ``--configs mineral_smooth,mineral_textured`` without typing Russian
# labels on the command line.
CONFIG_KEYS: Dict[str, int] = {
    "mineral_smooth":   0,
    "rapeseed_smooth":  1,
    "mineral_textured": 2,
    "rapeseed_textured": 3,
}


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

    Reference viscosity is ``oil["eta_diesel"]``. THD path uses this same
    table — load capacity in isoviscous Reynolds scales linearly with η,
    so a thermal-shifted target is found by scaling
    ``F_target_ref = F_target * eta_ref / eta_guess`` (Section 4 of the
    patch spec).

    Returns
    -------
    eps_table : (M,)
    W_table   : (M,)
    """
    eta = oil["eta_diesel"]
    eps_table = np.concatenate([
        np.linspace(0.001, 0.05, 5),
        np.linspace(0.06, 0.98, 15),
    ])
    W_table = np.zeros(len(eps_table))
    P_prev = None

    for i, eps in enumerate(eps_table):
        H = make_H(eps, Phi_mesh, Z_mesh, params,
                   textured=textured, phi_c_flat=phi_c, Z_c_flat=Z_c)
        P, F, _, _, _, _, _, _, _, _ = solve_and_compute(
            H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
            phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
            closure=closure, cavitation=cavitation,
        )
        W_table[i] = F
        P_prev = P

    print(f"    W_table: min={W_table.min()/1000:.1f} кН, "
          f"max={W_table.max()/1000:.1f} кН")
    return eps_table, W_table


def find_epsilon_for_load(F_target, eps_table, W_table,
                          Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                          oil, textured=False, phi_c=None, Z_c=None,
                          closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION,
                          eta=None):
    """Найти ε для заданной нагрузки: локализация по таблице + 5 шагов бисекции.

    ``eta`` overrides the per-oil isothermal viscosity (Section 4 of the
    THD-0 patch). When given, the bracket localisation uses
    ``F_target_ref = F_target * eta_ref / eta`` against the reference
    W_table, then bisection runs at the actual ``eta`` and converges to
    ``F_hyd ≈ F_target``.

    Returns
    -------
    eps_mid, F_hyd, mu, Q, h_min, p_max, F_tr, status
    """
    eta_ref = oil["eta_diesel"]
    eta_eff = float(eta) if eta is not None else float(eta_ref)
    F_target_ref = F_target * eta_ref / eta_eff

    # Локализация корня по таблице (на reference η).
    status = "ok"
    if F_target_ref <= W_table[0]:
        eps_lo, eps_hi = 0.001, eps_table[1]
        status = "below_range"
    elif F_target_ref >= W_table[-1]:
        eps_lo, eps_hi = eps_table[-2], 0.98
        status = "above_range"
    else:
        j = np.searchsorted(W_table, F_target_ref)
        eps_lo, eps_hi = eps_table[j - 1], eps_table[j]

    eps_mid = 0.5 * (eps_lo + eps_hi)
    F_hyd = mu = Qv = h_min = p_max = F_tr = 0.0

    for _ in range(5):
        eps_mid = 0.5 * (eps_lo + eps_hi)
        H = make_H(eps_mid, Phi_mesh, Z_mesh, params,
                   textured=textured, phi_c_flat=phi_c, Z_c_flat=Z_c)
        _, F_hyd, mu, Qv, h_min, p_max, F_tr, _, _, _ = solve_and_compute(
            H, d_phi, d_Z, params.R, params.L, eta_eff, params.n, params.c,
            phi_1D, Z_1D, Phi_mesh,
            closure=closure, cavitation=cavitation,
        )
        if F_hyd > F_target:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, status


def _omega_rad_s() -> float:
    return 2.0 * math.pi * params.n / 60.0


def _angle_thd_step(
    F_target: float,
    eps_table: np.ndarray, W_table: np.ndarray,
    Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
    *,
    oil: Dict[str, Any],
    textured: bool,
    phi_c, Z_c,
    closure: str, cavitation: str,
    walther_fit: Any,
    thermal: ThermalConfig,
    T_init_C: Optional[float] = None,
):
    """Per-angle THD outer loop. Returns dict with all per-angle scalars.

    The outer loop is the textbook fixed point on
    ``T = T_in + gamma * F_tr * U / (rho |Q| cp)`` with under-relaxation
    and a strict ``max_outer`` cap (Section 3 of the patch spec).
    """
    rho = float(oil["rho"])
    cp = thermal.cp_J_kgK
    gamma = thermal.gamma_mix
    T_in = thermal.T_in_C
    U = _omega_rad_s() * params.R

    T_guess = float(T_init_C) if T_init_C is not None else float(T_in)
    converged = False
    last_T_target = float(T_in)
    mdot_floor_hit = False
    outer_iters = 0

    eps_mid = F_hyd = mu = Qv = h_min = p_max = F_tr = 0.0
    eta_eff = float(oil["eta_diesel"])
    status = "ok"

    for it in range(thermal.max_outer):
        outer_iters = it + 1
        eta_eff = viscosity_at_T_C(walther_fit, T_guess)
        eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, status = (
            find_epsilon_for_load(
                F_target, eps_table, W_table,
                Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                oil=oil, textured=textured,
                phi_c=phi_c, Z_c=Z_c,
                closure=closure, cavitation=cavitation,
                eta=eta_eff,
            )
        )
        P_loss = float(F_tr) * U
        mdot_raw = rho * abs(float(Qv))
        mdot = max(mdot_raw, thermal.mdot_floor_kg_s)
        if mdot_raw < thermal.mdot_floor_kg_s:
            mdot_floor_hit = True
        T_target = global_static_step(
            T_in_C=T_in, P_loss_W=P_loss, mdot_kg_s=mdot,
            cp_J_kgK=cp, gamma=gamma,
            model=walther_fit,
        )
        last_T_target = float(T_target)
        if abs(T_target - T_guess) < thermal.tol_T_C:
            T_guess = T_target
            converged = True
            break
        T_guess = T_guess + thermal.underrelax_T * (T_target - T_guess)

    return dict(
        eps=eps_mid, F_hyd=F_hyd, mu=mu, Q=Qv, h_min=h_min,
        p_max=p_max, F_tr=F_tr, status=status,
        T_eff=float(T_guess), T_target=float(last_T_target),
        eta_eff=float(eta_eff),
        P_loss=float(F_tr) * U,
        mdot=max(rho * abs(float(Qv)), thermal.mdot_floor_kg_s),
        outer=int(outer_iters), converged=bool(converged),
        mdot_floor_hit=bool(mdot_floor_hit),
    )


def run_diesel_analysis(closure=DEFAULT_CLOSURE,
                          cavitation=DEFAULT_CAVITATION,
                          *,
                          thermal: Optional[ThermalConfig] = None,
                          phi_crank=None,
                          n_phi_grid=None,
                          n_z_grid=None,
                          configs=None):
    """Выполнить квазистационарный расчёт ДВС для всех конфигураций.

    Parameters
    ----------
    thermal : ThermalConfig | None
        ``None`` or ``mode == "off"`` — legacy isothermal behaviour with
        η = ``oil["eta_diesel"]``. ``mode == "global_static"`` runs the
        per-angle outer fixed-point loop described above.
    phi_crank, n_phi_grid, n_z_grid, configs
        Optional overrides for the module-level defaults; ``None`` keeps
        the existing behaviour (PHI_CRANK / N_GRID / CONFIGS).

    Returns
    -------
    results : dict
        Always contains the legacy keys (``epsilon``, ``hmin``, ``f``,
        ``pmax``, ``F_hyd``, ``configs``, ``phi_crank``, ``F_ext``) PLUS
        the new THD arrays (``T_eff``, ``T_target``, ``eta_eff``,
        ``P_loss``, ``Q``, ``mdot``, ``F_tr``, ``thermal_outer``,
        ``thermal_converged``) and the ``thermal_config`` dict.
        In ``mode="off"`` the THD arrays are filled consistently
        (T_eff = T_in, eta_eff = oil["eta_diesel"], outer=0,
        converged=True).
    """
    if thermal is None:
        thermal = ThermalConfig(mode="off")
    is_off = thermal.is_off()

    phi_crank = (np.asarray(phi_crank, dtype=float)
                  if phi_crank is not None else PHI_CRANK)
    cfg_list = list(configs) if configs is not None else CONFIGS
    Np = int(n_phi_grid) if n_phi_grid is not None else N_GRID
    Nz = int(n_z_grid) if n_z_grid is not None else Np

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(Np, Nz)
    phi_c, Z_c = setup_texture(params)

    F_ext = load_diesel(phi_crank)
    n_phi = len(phi_crank)
    n_cfg = len(cfg_list)

    eps_arr = np.zeros((n_cfg, n_phi))
    hmin_arr = np.zeros((n_cfg, n_phi))
    f_arr = np.zeros((n_cfg, n_phi))
    pmax_arr = np.zeros((n_cfg, n_phi))
    F_hyd_arr = np.zeros((n_cfg, n_phi))

    T_eff_arr = np.zeros((n_cfg, n_phi))
    T_target_arr = np.zeros((n_cfg, n_phi))
    eta_eff_arr = np.zeros((n_cfg, n_phi))
    P_loss_arr = np.zeros((n_cfg, n_phi))
    Q_arr = np.zeros((n_cfg, n_phi))
    mdot_arr = np.zeros((n_cfg, n_phi))
    F_tr_arr = np.zeros((n_cfg, n_phi))
    outer_arr = np.zeros((n_cfg, n_phi), dtype=np.int32)
    converged_arr = np.zeros((n_cfg, n_phi), dtype=bool)
    mdot_floor_hit_arr = np.zeros((n_cfg, n_phi), dtype=bool)

    U = _omega_rad_s() * params.R

    for ic, cfg in enumerate(cfg_list):
        print(f"  [{ic+1}/{n_cfg}] {cfg['label']}...")

        # Шаг 1: таблица W(ε) — построена один раз на reference η.
        print("    Построение таблицы W(ε)...")
        eps_table, W_table = build_load_table(
            Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
            oil=cfg["oil"], textured=cfg["textured"],
            phi_c=phi_c, Z_c=Z_c,
            closure=closure, cavitation=cavitation,
        )

        print(f"    F_ext:   min={F_ext.min()/1000:.1f} кН, "
              f"max={F_ext.max()/1000:.1f} кН")

        # Walther fit (only when THD is active — keeps mode='off' free of
        # solver thermal dependency). Pass cp/gamma into the OilModel so
        # the solver's static balance step uses the same numbers as
        # ThermalConfig.
        if is_off:
            walther_fit = None
        else:
            walther_fit = build_oil_walther(
                cfg["oil"],
                cp_J_kgK=thermal.cp_J_kgK,
                gamma_mix=thermal.gamma_mix,
            )

        # Шаг 2: по каждому углу.
        n_below = n_above = 0
        n_thd_unconverged = 0
        for ip, phi_k in enumerate(phi_crank):
            F_target = F_ext[ip]
            if is_off:
                eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, st = (
                    find_epsilon_for_load(
                        F_target, eps_table, W_table,
                        Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                        oil=cfg["oil"], textured=cfg["textured"],
                        phi_c=phi_c, Z_c=Z_c,
                        closure=closure, cavitation=cavitation,
                    )
                )
                eta_eff = float(cfg["oil"]["eta_diesel"])
                T_eff_C = thermal.T_in_C
                T_target_C = thermal.T_in_C
                P_loss = float(F_tr) * U
                mdot = max(float(cfg["oil"]["rho"]) * abs(float(Qv)),
                            thermal.mdot_floor_kg_s)
                outer_iters = 0
                converged = True
                mdot_floor_hit = (mdot <= thermal.mdot_floor_kg_s + 1e-30)
            else:
                rec = _angle_thd_step(
                    F_target, eps_table, W_table,
                    Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                    oil=cfg["oil"], textured=cfg["textured"],
                    phi_c=phi_c, Z_c=Z_c,
                    closure=closure, cavitation=cavitation,
                    walther_fit=walther_fit,
                    thermal=thermal,
                    T_init_C=None,  # cold-start each angle (no thermal memory)
                )
                eps_mid = rec["eps"]; F_hyd = rec["F_hyd"]; mu = rec["mu"]
                Qv = rec["Q"]; h_min = rec["h_min"]; p_max = rec["p_max"]
                F_tr = rec["F_tr"]; st = rec["status"]
                eta_eff = rec["eta_eff"]
                T_eff_C = rec["T_eff"]; T_target_C = rec["T_target"]
                P_loss = rec["P_loss"]; mdot = rec["mdot"]
                outer_iters = rec["outer"]; converged = rec["converged"]
                mdot_floor_hit = rec["mdot_floor_hit"]
                if not converged:
                    n_thd_unconverged += 1

            if st == "below_range":
                n_below += 1
            elif st == "above_range":
                n_above += 1

            eps_arr[ic, ip] = eps_mid
            hmin_arr[ic, ip] = h_min
            f_arr[ic, ip] = mu
            pmax_arr[ic, ip] = p_max
            F_hyd_arr[ic, ip] = F_hyd
            T_eff_arr[ic, ip] = T_eff_C
            T_target_arr[ic, ip] = T_target_C
            eta_eff_arr[ic, ip] = eta_eff
            P_loss_arr[ic, ip] = P_loss
            Q_arr[ic, ip] = Qv
            mdot_arr[ic, ip] = mdot
            F_tr_arr[ic, ip] = F_tr
            outer_arr[ic, ip] = outer_iters
            converged_arr[ic, ip] = converged
            mdot_floor_hit_arr[ic, ip] = mdot_floor_hit

        print(f"    Вне диапазона: {n_below} ниже, {n_above} выше "
              f"(из {n_phi})")
        if not is_off:
            T_lo = float(np.min(T_eff_arr[ic]))
            T_hi = float(np.max(T_eff_arr[ic]))
            T_mn = float(np.mean(T_eff_arr[ic]))
            n_floor = int(np.sum(mdot_floor_hit_arr[ic]))
            print(f"    THD: T_eff min/mean/max = "
                  f"{T_lo:.1f}/{T_mn:.1f}/{T_hi:.1f} °C, "
                  f"unconverged={n_thd_unconverged}, "
                  f"mdot_floor_hit={n_floor}")
            if T_hi > 180.0:
                print(f"    [WARN] max T_eff = {T_hi:.1f} °C > 180 °C")

    return {
        "phi_crank": phi_crank,
        "F_ext": F_ext,
        "epsilon": eps_arr,
        "hmin": hmin_arr,
        "f": f_arr,
        "pmax": pmax_arr,
        "F_hyd": F_hyd_arr,
        "configs": cfg_list,
        "T_eff": T_eff_arr,
        "T_target": T_target_arr,
        "eta_eff": eta_eff_arr,
        "P_loss": P_loss_arr,
        "Q": Q_arr,
        "mdot": mdot_arr,
        "F_tr": F_tr_arr,
        "thermal_outer": outer_arr,
        "thermal_converged": converged_arr,
        "mdot_floor_hit": mdot_floor_hit_arr,
        "thermal_config": thermal.to_dict(),
    }
