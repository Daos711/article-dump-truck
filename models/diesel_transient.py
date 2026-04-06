"""Нестационарная модель подшипника ДВС (time-stepping).

Совместное решение уравнения Рейнольдса со squeeze и уравнения движения вала.
Интегратор: semi-implicit Euler.
"""
import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
from reynolds_solver.squeeze import squeeze_to_api_params
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


LAMBDA_CRANK = 0.27  # R_crank / L_rod для БелАЗ


def load_diesel(phi_deg, F_max=None):
    """Нагрузка ДВС: Вибе-функция + КШМ-разложение на Fx, Fy.

    Parameters
    ----------
    phi_deg : float or array — угол ПКВ (0..720°)
    F_max : float — пиковая нагрузка (Н)

    Returns
    -------
    Fx, Fy : float or array — компоненты нагрузки (Н).
    """
    if F_max is None:
        F_max = params.F_max

    phi = np.atleast_1d(np.asarray(phi_deg, dtype=float)) % 720.0

    # Вибе-функция: асимметричный пик
    phi_s = 345.0   # начало нарастания (°)
    phi_p = 370.0   # пик (°)
    m_vibe = 2.0
    k_vibe = 1.2

    x = np.clip((phi - phi_s) / (phi_p - phi_s), 0, None)
    F_vibe = np.where(x > 0,
        (F_max - params.F_base) * x**m_vibe * np.exp(m_vibe / k_vibe * (1 - x**k_vibe)),
        0.0)
    F_total = F_vibe + params.F_base

    # КШМ-разложение: проекции одной и той же F_total
    beta = np.arcsin(LAMBDA_CRANK * np.sin(np.deg2rad(phi)))
    Fx = F_total * np.sin(beta)
    Fy = -F_total * np.cos(beta)

    return Fx, Fy


def build_H_2d(eps_x, eps_y, Phi_mesh, Z_mesh, p,
               textured=False, phi_c_flat=None, Z_c_flat=None):
    """Зазор для 2D-эксцентриситета: H = 1 − εx·cos(θ) − εy·sin(θ) [+ текстура]."""
    H0 = 1.0 - eps_x * np.cos(Phi_mesh) - eps_y * np.sin(Phi_mesh)
    H0 = np.sqrt(H0**2 + (2e-6 / p.c)**2)  # регуляризация σ = 2 мкм
    if not textured:
        return H0
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c
    return create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A, B
    )



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

    for ic, cfg in enumerate(CONFIGS[:1]):  # TEST: 1 конфиг, вернуть CONFIGS
        eta = cfg["oil"]["eta_diesel"]
        alpha_pv = None  # пьезовязкость отключена до стабилизации transient
        p_scale = 6.0 * eta * omega * (params.R / params.c) ** 2

        print(f"\n  [{ic+1}/{n_cfg}] {cfg['label']}...")

        # Начальные условия
        ex = params.eps_x0 * params.c
        ey = params.eps_y0 * params.c
        vx, vy = 0.0, 0.0
        ax_prev, ay_prev = 0.0, 0.0  # ускорение на предыдущем шаге
        P_prev = None
        contact_count = 0

        def _solve_at(ex_, ey_, vx_, vy_, P_init_):
            """Решить Reynolds в точке (ex_, ey_) и вернуть силы."""
            eps_x_ = ex_ / params.c
            eps_y_ = ey_ / params.c
            H_ = build_H_2d(eps_x_, eps_y_, Phi_mesh, Z_mesh, params,
                            textured=cfg["textured"],
                            phi_c_flat=phi_c, Z_c_flat=Z_c)
            xp_, yp_, bt_ = squeeze_to_api_params(
                -vx_, -vy_, params.c, omega, d_phi)
            kw = dict(
                closure=closure, cavitation=cavitation,
                omega=1.5, tol=1e-5, max_iter=50000,
                P_init=P_init_,
                xprime=xp_, yprime=yp_, beta=bt_,
            )
            if alpha_pv is not None:
                kw["alpha_pv"] = alpha_pv
                kw["p_scale"] = p_scale
                kw["relax_pv"] = 0.4
                kw["max_outer_pv"] = 50
            res = solve_reynolds(H_, d_phi, d_Z, params.R, params.L, **kw)
            P_ = res[0]
            Fx_, Fy_ = compute_hydro_forces(
                P_, p_scale, Phi_mesh, phi_1D, Z_1D, params.R, params.L)
            return P_, H_, Fx_, Fy_

        def _clamp(ex_, ey_, vx_, vy_):
            """Clamp |ε| < eps_max, обнулить радиальную скорость к стенке."""
            clamped = False
            eps_mag_ = np.hypot(ex_, ey_) / params.c
            if eps_mag_ > params.eps_max:
                scale_ = params.eps_max / eps_mag_
                ex_ *= scale_
                ey_ *= scale_
                e_hat_x_ = ex_ / (eps_mag_ * params.c)
                e_hat_y_ = ey_ / (eps_mag_ * params.c)
                v_rad_ = vx_ * e_hat_x_ + vy_ * e_hat_y_
                if v_rad_ > 0:
                    vx_ -= v_rad_ * e_hat_x_
                    vy_ -= v_rad_ * e_hat_y_
                clamped = True
            return ex_, ey_, vx_, vy_, clamped

        for step in range(n_steps):
            phi_deg = phi_crank_deg[step] % 720.0

            # --- Velocity Verlet ---
            # 1. Predict position
            ex_pred = ex + vx * dt + 0.5 * ax_prev * dt**2
            ey_pred = ey + vy * dt + 0.5 * ay_prev * dt**2

            # Clamp predicted position
            ex_pred, ey_pred, vx, vy, clamped_pred = _clamp(
                ex_pred, ey_pred, vx, vy)
            if clamped_pred:
                contact_count += 1
                P_prev = None

            # 2. Solve Reynolds at predicted position
            P, H, Fx_hyd, Fy_hyd = _solve_at(
                ex_pred, ey_pred, vx, vy, P_prev)
            P_prev = P

            # 3. External load at current angle
            Fx_ext, Fy_ext = load_diesel(phi_deg, F_max=F_max)
            Fx_ext = float(Fx_ext)
            Fy_ext = float(Fy_ext)

            # 4. New acceleration
            ax_new = (Fx_ext + Fx_hyd) / params.m_shaft
            ay_new = (Fy_ext + Fy_hyd) / params.m_shaft

            # 5. Correct velocity
            vx += 0.5 * (ax_prev + ax_new) * dt
            vy += 0.5 * (ay_prev + ay_new) * dt

            # 6. Accept position
            ex, ey = ex_pred, ey_pred
            ax_prev, ay_prev = ax_new, ay_new

            # 7. Final clamp (safety)
            ex, ey, vx, vy, clamped_final = _clamp(ex, ey, vx, vy)
            if clamped_final:
                contact_count += 1
                P_prev = None

            # 8. Характеристики
            h_dim = H * params.c
            h_min = np.min(h_dim)
            p_max = np.max(P * p_scale)
            F_friction = compute_friction(
                P, p_scale, H, Phi_mesh, phi_1D, Z_1D,
                eta, omega, params.R, params.L, params.c)
            F_hyd_mag = np.hypot(Fx_hyd, Fy_hyd)
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
