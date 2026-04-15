"""Безразмеризация для дизельного кейса.

Convention (Ausas 2008):
  τ = ω·t           (безразмерное время)
  H = h/c           (безразмерный зазор)
  P = p / p_scale   (безразмерное давление)
  W_nd = F / F₀     (безразмерная сила)

  p_scale = 6·η·ω·(R/c)²
  F₀ = 24π·η·ω·R⁴/c²
  M_nd = m_eff·c³·ω / (24π·η·R⁴)

Один 4-тактный цикл = 720° CA. По безразмерному времени:
  t_cycle = 720° / ω_shaft × (2π/360°) = 4π/ω_shaft
  τ_cycle = ω·t_cycle = 4π
"""
import numpy as np

try:
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─── Масштабы ─────────────────────────────────────────────────────

def omega_from_rpm(n_rpm):
    return 2 * np.pi * n_rpm / 60


def pressure_scale(eta, omega, R, c):
    return 6 * eta * omega * (R / c)**2


def force_scale(eta, omega, R, c):
    return 24 * np.pi * eta * omega * R**4 / c**2


def mass_nondim(m_eff_kg, eta, omega, R, c):
    return m_eff_kg * c**3 * omega / (24 * np.pi * eta * R**4)


# ─── Прямое / обратное безразмеривание ─────────────────────────────

def nondim_force(F_dim_N, eta, omega, R, c):
    return F_dim_N / force_scale(eta, omega, R, c)


def dim_force(F_nd, eta, omega, R, c):
    return F_nd * force_scale(eta, omega, R, c)


def nondim_pressure(p_Pa, eta, omega, R, c):
    return p_Pa / pressure_scale(eta, omega, R, c)


def dim_pressure(P_nd, eta, omega, R, c):
    return P_nd * pressure_scale(eta, omega, R, c)


def nondim_length(h_m, c):
    return h_m / c


def dim_length(H_nd, c):
    return H_nd * c


# ─── Crank angle ↔ безразмерное время ─────────────────────────────

def crank_deg_to_tau(crank_deg):
    """720° CA = 4π безразмерного времени τ (один 4-тактный цикл).

    360° CA вала = 2π радиан ω·t = 2π безразм. времени.
    720° CA = 4π τ.
    """
    return np.deg2rad(crank_deg) * 2


def tau_to_crank_deg(tau):
    return np.rad2deg(tau / 2)


CYCLE_TAU = 4 * np.pi   # безразмерный период одного 4-тактного цикла


# ─── Периодический интерполятор нагрузки ──────────────────────────

def make_load_fn(tau_arr, WaX_arr, WaY_arr, T_cycle=CYCLE_TAU):
    """Периодический интерполятор нагрузки для solver load_fn(t).

    Parameters
    ----------
    tau_arr : (N,) безразмерное время в пределах [0, T_cycle]
    WaX_arr, WaY_arr : (N,) безразмерные силы
    T_cycle : длина цикла по τ (default CYCLE_TAU = 4π)

    Returns
    -------
    load_fn(t) -> (WaX, WaY)  с периодичностью T_cycle
    """
    tau_arr = np.asarray(tau_arr)
    WaX_arr = np.asarray(WaX_arr)
    WaY_arr = np.asarray(WaY_arr)

    if _HAS_SCIPY:
        ix = interp1d(tau_arr, WaX_arr, kind='linear',
                       fill_value='extrapolate', bounds_error=False)
        iy = interp1d(tau_arr, WaY_arr, kind='linear',
                       fill_value='extrapolate', bounds_error=False)

        def load_fn(t):
            t_mod = t % T_cycle
            return float(ix(t_mod)), float(iy(t_mod))
    else:
        # np.interp — линейная интерполяция, периодическая через mod
        def load_fn(t):
            t_mod = t % T_cycle
            return (float(np.interp(t_mod, tau_arr, WaX_arr)),
                    float(np.interp(t_mod, tau_arr, WaY_arr)))

    return load_fn


def make_load_fn_from_crank(crank_deg_arr, WaX_arr, WaY_arr):
    """Вариант: вход — crank_deg ∈ [0, 720°], конвертация в τ."""
    tau_arr = crank_deg_to_tau(crank_deg_arr)
    return make_load_fn(tau_arr, WaX_arr, WaY_arr, T_cycle=CYCLE_TAU)
