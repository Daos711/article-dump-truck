"""Стационарный анализ подшипника центробежного насоса.

Для диапазона эксцентриситетов ε вычисляются характеристики W, f, h_min, Q,
F_tr, N_loss для 4 конфигураций (гладкий/текстурированный × минеральное/рапсовое).
"""
import numpy as np
from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
    DEFAULT_CLOSURE, DEFAULT_CAVITATION,
)
from config import pump_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

N_GRID = 300
EPSILON_VALUES = np.linspace(0.1, 0.8, 15)

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


def run_pump_analysis(closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Выполнить стационарный расчёт для всех конфигураций.

    Returns
    -------
    results : dict
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N_GRID)
    phi_c, Z_c = setup_texture(params)

    omega = 2 * np.pi * params.n / 60.0
    U = omega * params.R  # линейная скорость вала

    n_eps = len(EPSILON_VALUES)
    n_cfg = len(CONFIGS)

    W = np.zeros((n_cfg, n_eps))
    f = np.zeros((n_cfg, n_eps))
    hmin = np.zeros((n_cfg, n_eps))
    Q = np.zeros((n_cfg, n_eps))
    F_tr = np.zeros((n_cfg, n_eps))
    N_loss = np.zeros((n_cfg, n_eps))

    for ic, cfg in enumerate(CONFIGS):
        eta = cfg["oil"]["eta_pump"]
        P_prev = None
        print(f"  [{ic+1}/{n_cfg}] {cfg['label']}...")

        for ie, eps in enumerate(EPSILON_VALUES):
            H = make_H(eps, Phi_mesh, Z_mesh, params,
                       textured=cfg["textured"],
                       phi_c_flat=phi_c, Z_c_flat=Z_c)

            P, F, mu, Qv, h_m, p_m, F_friction = solve_and_compute(
                H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
                phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
                closure=closure, cavitation=cavitation,
            )
            P_prev = P

            W[ic, ie] = F
            f[ic, ie] = mu
            hmin[ic, ie] = h_m
            Q[ic, ie] = Qv
            F_tr[ic, ie] = F_friction
            N_loss[ic, ie] = F_friction * U

            print(f"    eps={eps:.2f}: W={F:.0f} Н, f={mu:.4f}, "
                  f"F_tr={F_friction:.1f} Н, N_loss={F_friction*U:.0f} Вт, "
                  f"h_min={h_m*1e6:.1f} мкм, p_max={p_m/1e6:.1f} МПа")

    return {
        "epsilon": EPSILON_VALUES,
        "W": W, "f": f, "hmin": hmin, "Q": Q,
        "F_tr": F_tr, "N_loss": N_loss,
        "configs": CONFIGS,
    }
