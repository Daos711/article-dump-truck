"""Стационарный анализ подшипника центробежного насоса.

Для диапазона эксцентриситетов ε вычисляются характеристики W, f, h_min, Q,
F_tr, N_loss, p_max для 4 конфигураций (гладкий/текстурированный × минеральное/рапсовое).
"""
import numpy as np
import types
from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
    DEFAULT_CLOSURE, DEFAULT_CAVITATION,
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

N_PHI = 800
N_Z = 200
EPSILON_VALUES = np.linspace(0.1, 0.8, 15)
MAX_OUTER_PV = 50  # порог расходимости пьезовязкого солвера

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


def run_pump_analysis(h_p_override=None,
                      closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Выполнить стационарный расчёт для всех конфигураций.

    Parameters
    ----------
    h_p_override : float or None
        Если задано, использовать эту глубину текстуры вместо params.h_p.

    Returns
    -------
    results : dict
    """
    # Если нужно переопределить h_p — создаём копию params
    if h_p_override is not None:
        p = types.SimpleNamespace(**{k: getattr(params, k)
                                     for k in dir(params) if not k.startswith('_')})
        p.h_p = h_p_override
    else:
        p = params

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_PHI, N_Z)
    phi_c, Z_c = setup_texture(p)

    omega = 2 * np.pi * p.n / 60.0
    U = omega * p.R

    n_eps = len(EPSILON_VALUES)
    n_cfg = len(CONFIGS)

    W = np.zeros((n_cfg, n_eps))
    f = np.zeros((n_cfg, n_eps))
    hmin = np.zeros((n_cfg, n_eps))
    Q = np.zeros((n_cfg, n_eps))
    F_tr = np.zeros((n_cfg, n_eps))
    N_loss = np.zeros((n_cfg, n_eps))
    pmax = np.zeros((n_cfg, n_eps))

    n_outer_max = 0
    n_outer_sum = 0
    n_outer_count = 0

    for ic, cfg in enumerate(CONFIGS):
        eta = cfg["oil"]["eta_pump"]
        alpha_pv = cfg["oil"].get("alpha_pv")
        P_prev = None
        print(f"  [{ic+1}/{n_cfg}] {cfg['label']}...")

        # Параметры текстуры для аналитической квадратуры
        tex_params = None
        if cfg["textured"]:
            tex_params = {
                "phi_c": phi_c,
                "Z_c": Z_c,
                "A": 2 * p.a_dim / p.L,
                "B": p.b_dim / p.R,
                "H_p": p.h_p / p.c,
                "profile": "smoothcap",
            }

        for ie, eps in enumerate(EPSILON_VALUES):
            H = make_H(eps, Phi_mesh, Z_mesh, p,
                       textured=cfg["textured"],
                       phi_c_flat=phi_c, Z_c_flat=Z_c,
                       profile="smoothcap")

            # H_smooth для аналитической квадратуры (только текстура)
            H_smooth = None
            if cfg["textured"]:
                H_smooth = make_H(eps, Phi_mesh, Z_mesh, p,
                                  textured=False)

            P, F, mu, Qv, h_m, p_m, F_friction, n_out = solve_and_compute(
                H, d_phi, d_Z, p.R, p.L, eta, p.n, p.c,
                phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
                closure=closure, cavitation=cavitation,
                alpha_pv=alpha_pv,
                subcell_quad=cfg["textured"],
                H_smooth=H_smooth,
                texture_params=tex_params,
            )
            P_prev = P

            if n_out > 0:
                n_outer_max = max(n_outer_max, n_out)
                n_outer_sum += n_out
                n_outer_count += 1

            # Детекция расходимости пьезовязкого цикла
            diverged = (n_out >= MAX_OUTER_PV)
            if diverged:
                W[ic, ie] = np.nan
                f[ic, ie] = np.nan
                hmin[ic, ie] = h_m  # геометрический, всегда валиден
                Q[ic, ie] = np.nan
                F_tr[ic, ie] = np.nan
                N_loss[ic, ie] = np.nan
                pmax[ic, ie] = np.nan
                P_prev = None  # сбросить начальное приближение
                print(f"    eps={eps:.2f}: РАСХОДИМОСТЬ (n_outer={n_out})")
            else:
                W[ic, ie] = F
                f[ic, ie] = mu
                hmin[ic, ie] = h_m
                Q[ic, ie] = Qv
                F_tr[ic, ie] = F_friction
                N_loss[ic, ie] = F_friction * U
                pmax[ic, ie] = p_m

                pv_tag = f", n_outer={n_out}" if n_out > 0 else ""
                print(f"    eps={eps:.2f}: W={F:.0f} Н, f={mu:.4f}, "
                      f"F_tr={F_friction:.1f} Н, N_loss={F_friction*U:.0f} Вт, "
                      f"h_min={h_m*1e6:.1f} мкм, p_max={p_m/1e6:.1f} МПа{pv_tag}")

    n_outer_avg = n_outer_sum / n_outer_count if n_outer_count > 0 else 0
    if n_outer_count > 0:
        print(f"  Пьезовязкость: n_outer avg={n_outer_avg:.1f}, max={n_outer_max}")

    return {
        "epsilon": EPSILON_VALUES,
        "W": W, "f": f, "hmin": hmin, "Q": Q,
        "F_tr": F_tr, "N_loss": N_loss, "pmax": pmax,
        "configs": CONFIGS,
        "n_outer_avg": n_outer_avg,
        "n_outer_max": n_outer_max,
    }
