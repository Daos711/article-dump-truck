"""Нестационарная модель подшипника ДВС (time-stepping).

Совместное решение уравнения Рейнольдса со squeeze и уравнения движения вала.
Интегратор: semi-implicit Euler.
"""
import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from models.bearing_model import setup_grid, setup_texture, DEFAULT_CLOSURE, DEFAULT_CAVITATION
from config import diesel_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

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


def load_diesel(phi_deg, F_max=None, sigma=None):
    """Surrogate индикаторной диаграммы 4-тактного ДВС.

    Не полная КШМ-модель, а гауссов пик при ~370° ПКВ + базовый уровень.

    Parameters
    ----------
    phi_deg : float or array — угол ПКВ (0..720°)
    F_max : float — пиковая нагрузка (Н)
    sigma : float — ширина пика (°)

    Returns
    -------
    Fx, Fy : float — компоненты внешней нагрузки (Н).
             Fx = 0, Fy = -F_total (вал прижимается вниз).
    """
    if F_max is None:
        F_max = params.F_max
    if sigma is None:
        sigma = params.sigma_deg
    F_gas = F_max * np.exp(-((phi_deg - 370) ** 2) / (2 * sigma ** 2))
    F_total = F_gas + params.F_base
    Fx = np.zeros_like(np.atleast_1d(F_total), dtype=float)
    Fy = -np.asarray(F_total, dtype=float)
    return Fx, Fy


def build_H_2d(eps_x, eps_y, Phi_mesh, Z_mesh, p,
               textured=False, phi_c_flat=None, Z_c_flat=None):
    """Зазор для 2D-эксцентриситета: H = 1 + εx·cos(θ) + εy·sin(θ) [+ текстура]."""
    H0 = 1.0 + eps_x * np.cos(Phi_mesh) + eps_y * np.sin(Phi_mesh)
    if not textured:
        return H0
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c
    return create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A, B
    )


def squeeze_to_api_params(vx, vy, c, omega_shaft, d_phi_mesh):
    """Пересчёт скоростей вала (vx, vy) в параметры squeeze для солвера.

    Солвер ожидает безразмерные xprime, yprime (= d(ε)/d(θ_shaft)),
    и beta = (L/D)² (отношение длины к диаметру, в квадрате).

    xprime = (1/c) · vx / omega_shaft  =  dεx / dθ
    yprime = (1/c) · vy / omega_shaft  =  dεy / dθ
    beta используется солвером для масштабирования squeeze-члена.
    """
    xprime = vx / (c * omega_shaft)
    yprime = vy / (c * omega_shaft)
    beta = 1.0  # стандартный масштаб, squeeze уже в безразмерной форме
    return xprime, yprime, beta


def compute_hydro_forces(P, p_scale, Phi_mesh, phi_1D, Z_1D, R, L):
    """Вычислить компоненты гидродинамической силы на вал.

    Знаки соответствуют bearing_model.py: при ey < 0 → Fy_hyd > 0.
    """
    P_dim = P * p_scale
    Fx = -np.trapz(np.trapz(P_dim * np.cos(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    Fy = -np.trapz(np.trapz(P_dim * np.sin(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    return Fx, Fy


def compute_friction(P, p_scale, H, Phi_mesh, phi_1D, Z_1D,
                     eta, omega, R, L, c):
    """Сила трения и коэффициент трения (как в bearing_model.py)."""
    P_dim = P * p_scale
    h_dim = H * c
    tau_couette = eta * omega * R / h_dim
    dP_dphi = np.gradient(P_dim, phi_1D[1] - phi_1D[0], axis=1)
    tau_pressure = h_dim / 2.0 * dP_dphi / R
    tau = tau_couette + tau_pressure
    F_friction = np.trapz(np.trapz(np.abs(tau), phi_1D, axis=1),
                          Z_1D, axis=0) * R * L / 2
    return F_friction


def run_transient(F_max=None, debug=False,
                  closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Нестационарный расчёт для всех 4 конфигураций.

    Parameters
    ----------
    F_max : float or None — пиковая нагрузка. При None берётся из params.
    debug : bool — если True, использует F_max_debug.

    Returns
    -------
    results : dict
    """
    if F_max is None:
        F_max = params.F_max_debug if debug else params.F_max

    N = params.N_grid_transient
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N)
    phi_c, Z_c = setup_texture(params)

    omega = 2 * np.pi * params.n / 60.0  # рад/с вала
    U = omega * params.R
    d_phi_crank_rad = np.deg2rad(params.d_phi_crank_deg)
    dt = d_phi_crank_rad / omega
    n_steps_per_cycle = int(round(720.0 / params.d_phi_crank_deg))
    n_steps = n_steps_per_cycle * params.n_cycles

    print(f"  Параметры: F_max={F_max/1e3:.0f} кН, dt={dt*1e6:.1f} мкс, "
          f"{n_steps} шагов ({params.n_cycles} цикла × {n_steps_per_cycle})")

    n_cfg = len(CONFIGS)
    # Массивы для всех шагов
    eps_x_all = np.zeros((n_cfg, n_steps))
    eps_y_all = np.zeros((n_cfg, n_steps))
    hmin_all = np.zeros((n_cfg, n_steps))
    pmax_all = np.zeros((n_cfg, n_steps))
    f_all = np.zeros((n_cfg, n_steps))
    Ftr_all = np.zeros((n_cfg, n_steps))
    Nloss_all = np.zeros((n_cfg, n_steps))
    Fx_hyd_all = np.zeros((n_cfg, n_steps))
    Fy_hyd_all = np.zeros((n_cfg, n_steps))

    phi_crank_deg = np.arange(n_steps) * params.d_phi_crank_deg

    for ic, cfg in enumerate(CONFIGS):
        eta = cfg["oil"]["eta_diesel"]
        alpha_pv = cfg["oil"].get("alpha_pv")
        p_scale = 6.0 * eta * omega * (params.R / params.c) ** 2

        print(f"\n  [{ic+1}/{n_cfg}] {cfg['label']}...")

        # Начальные условия
        ex = params.eps_x0 * params.c
        ey = params.eps_y0 * params.c
        vx, vy = 0.0, 0.0
        P_prev = None
        contact_count = 0

        for step in range(n_steps):
            phi_deg = phi_crank_deg[step] % 720.0
            eps_x = ex / params.c
            eps_y = ey / params.c

            # 1. Зазор
            H = build_H_2d(eps_x, eps_y, Phi_mesh, Z_mesh, params,
                           textured=cfg["textured"],
                           phi_c_flat=phi_c, Z_c_flat=Z_c)

            # 2. Squeeze
            xprime, yprime, beta = squeeze_to_api_params(
                vx, vy, params.c, omega, d_phi)

            # 3. Решить Reynolds
            solver_kw = dict(
                closure=closure,
                cavitation=cavitation,
                omega=1.5,
                tol=1e-5,
                max_iter=50000,
                P_init=P_prev,
                xprime=xprime,
                yprime=yprime,
                beta=beta,
            )
            if alpha_pv is not None:
                solver_kw["alpha_pv"] = alpha_pv
                solver_kw["p_scale"] = p_scale
                solver_kw["relax_pv"] = 0.4
                solver_kw["max_outer_pv"] = 50

            result = solve_reynolds(H, d_phi, d_Z, params.R, params.L,
                                    **solver_kw)
            if len(result) == 4:
                P, residual, n_iter, n_outer = result
            else:
                P, residual, n_iter = result
            P_prev = P

            # 4. Гидродинамические силы
            Fx_hyd, Fy_hyd = compute_hydro_forces(
                P, p_scale, Phi_mesh, phi_1D, Z_1D, params.R, params.L)

            # 5. Внешняя нагрузка
            Fx_ext, Fy_ext = load_diesel(phi_deg, F_max=F_max)
            Fx_ext = float(Fx_ext)
            Fy_ext = float(Fy_ext)

            # DEBUG: первые 10 шагов — печать ключевых величин
            if step < 10 and ic == 0:
                print(f"    step={step}: εx={eps_x:.4f} εy={eps_y:.4f} "
                      f"|ε|={np.sqrt(eps_x**2+eps_y**2):.4f}")
                print(f"      vx={vx:.4f} vy={vy:.4f} "
                      f"xprime={xprime:.4e} yprime={yprime:.4e}")
                print(f"      P: min={np.min(P):.4e} max={np.max(P):.4e} "
                      f"p_scale={p_scale:.2e}")
                print(f"      Fx_hyd={Fx_hyd:.1f} Fy_hyd={Fy_hyd:.1f} "
                      f"Fx_ext={Fx_ext:.1f} Fy_ext={Fy_ext:.1f}")
                print(f"      H: min={np.min(H):.4f} max={np.max(H):.4f}")

            # Проверка overflow
            if not (np.isfinite(Fx_hyd) and np.isfinite(Fy_hyd)):
                print(f"    OVERFLOW на шаге {step}! Остановка конфигурации.")
                # Заполнить остаток NaN
                eps_x_all[ic, step:] = np.nan
                eps_y_all[ic, step:] = np.nan
                hmin_all[ic, step:] = np.nan
                pmax_all[ic, step:] = np.nan
                f_all[ic, step:] = np.nan
                Ftr_all[ic, step:] = np.nan
                Nloss_all[ic, step:] = np.nan
                Fx_hyd_all[ic, step:] = np.nan
                Fy_hyd_all[ic, step:] = np.nan
                break

            # 6. Semi-implicit Euler
            ax = (Fx_ext + Fx_hyd) / params.m_shaft
            ay = (Fy_ext + Fy_hyd) / params.m_shaft
            vx += ax * dt
            vy += ay * dt
            ex += vx * dt
            ey += vy * dt

            # 7. Clamp — защита от контакта
            eps_mag = np.sqrt(ex**2 + ey**2) / params.c
            if eps_mag > params.eps_max:
                scale = params.eps_max / eps_mag
                ex *= scale
                ey *= scale
                # Обнулить радиальную компоненту скорости (к стенке)
                e_hat_x = ex / (eps_mag * params.c)
                e_hat_y = ey / (eps_mag * params.c)
                v_radial = vx * e_hat_x + vy * e_hat_y
                if v_radial > 0:  # движется к стенке
                    vx -= v_radial * e_hat_x
                    vy -= v_radial * e_hat_y
                contact_count += 1

            # 8. Характеристики
            h_dim = H * params.c
            h_min = np.min(h_dim)
            p_max = np.max(P * p_scale)
            F_friction = compute_friction(
                P, p_scale, H, Phi_mesh, phi_1D, Z_1D,
                eta, omega, params.R, params.L, params.c)
            F_hyd_mag = np.sqrt(Fx_hyd**2 + Fy_hyd**2)
            mu_val = F_friction / max(F_hyd_mag, 1.0)

            eps_x_all[ic, step] = ex / params.c
            eps_y_all[ic, step] = ey / params.c
            hmin_all[ic, step] = h_min
            pmax_all[ic, step] = p_max
            f_all[ic, step] = mu_val
            Ftr_all[ic, step] = F_friction
            Nloss_all[ic, step] = F_friction * U
            Fx_hyd_all[ic, step] = Fx_hyd
            Fy_hyd_all[ic, step] = Fy_hyd

            # Прогресс (каждые 10%)
            if (step + 1) % (n_steps // 10) == 0:
                pct = 100 * (step + 1) / n_steps
                eps_now = np.sqrt(ex**2 + ey**2) / params.c
                print(f"    {pct:3.0f}%: φ={phi_deg:6.1f}°, "
                      f"|ε|={eps_now:.3f}, h_min={h_min*1e6:.1f} мкм, "
                      f"p_max={p_max/1e6:.1f} МПа")

        print(f"    Контакт (clamp): {contact_count} из {n_steps} шагов")

    # Углы ПКВ для последнего цикла
    last_start = n_steps_per_cycle * (params.n_cycles - 1)
    phi_last = phi_crank_deg[last_start:last_start + n_steps_per_cycle]

    # Внешняя нагрузка для последнего цикла
    Fx_ext_last, Fy_ext_last = load_diesel(phi_last % 720.0, F_max=F_max)

    return {
        "phi_crank_deg": phi_crank_deg,
        "phi_last": phi_last,
        "last_start": last_start,
        "n_steps_per_cycle": n_steps_per_cycle,
        "eps_x": eps_x_all,
        "eps_y": eps_y_all,
        "hmin": hmin_all,
        "pmax": pmax_all,
        "f": f_all,
        "F_tr": Ftr_all,
        "N_loss": Nloss_all,
        "Fx_hyd": Fx_hyd_all,
        "Fy_hyd": Fy_hyd_all,
        "Fy_ext_last": Fy_ext_last,
        "F_max": F_max,
        "configs": CONFIGS,
        "dt": dt,
    }
