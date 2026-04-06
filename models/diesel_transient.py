"""Нестационарная модель подшипника ДВС (time-stepping).

Совместное решение уравнения Рейнольдса со squeeze и уравнения движения вала.
Интегратор: Velocity Verlet с sub-iterations.
"""
import time as _time
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


def load_diesel(phi_deg, F_max=None):
    """Нагрузка ДВС: Вибе-функция + КШМ-разложение на Fx, Fy."""
    if F_max is None:
        F_max = params.F_max

    phi = np.atleast_1d(np.asarray(phi_deg, dtype=float)) % 720.0

    phi_s = 345.0
    phi_p = 370.0
    m_vibe = 2.0
    k_vibe = 1.2

    x = np.clip((phi - phi_s) / (phi_p - phi_s), 0, None)
    F_vibe = np.where(x > 0,
        (F_max - params.F_base) * x**m_vibe * np.exp(m_vibe / k_vibe * (1 - x**k_vibe)),
        0.0)
    F_total = F_vibe + params.F_base

    phi_rad = np.deg2rad(phi)
    beta = np.arcsin(params.lambda_crank * np.sin(phi_rad))
    Fx = F_total * np.sin(beta)
    Fy = -F_total * np.cos(beta)

    return Fx, Fy


def build_H_2d(eps_x, eps_y, Phi_mesh, Z_mesh, p,
               textured=False, phi_c_flat=None, Z_c_flat=None):
    """Зазор для 2D-эксцентриситета: H = 1 − εx·cos(θ) − εy·sin(θ) [+ текстура]."""
    H0 = 1.0 - eps_x * np.cos(Phi_mesh) - eps_y * np.sin(Phi_mesh)
    H0 = np.sqrt(H0**2 + (p.sigma / p.c)**2)  # регуляризация шероховатости
    if not textured:
        return H0
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c
    return create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A, B
    )


def compute_hydro_forces(P, p_scale, Phi_mesh, phi_1D, Z_1D, R, L):
    """Вычислить компоненты гидродинамической силы на вал."""
    P_dim = P * p_scale
    Fx = -np.trapz(np.trapz(P_dim * np.cos(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    Fy = -np.trapz(np.trapz(P_dim * np.sin(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    return Fx, Fy


def compute_friction(P, p_scale, H, Phi_mesh, phi_1D, Z_1D,
                     eta, omega, R, L, c):
    """Сила трения."""
    P_dim = P * p_scale
    h_dim = H * c
    tau_couette = eta * omega * R / h_dim
    dP_dphi = np.gradient(P_dim, phi_1D[1] - phi_1D[0], axis=1)
    tau_pressure = h_dim / 2.0 * dP_dphi / R
    tau = tau_couette + tau_pressure
    F_friction = np.trapz(np.trapz(np.abs(tau), phi_1D, axis=1),
                          Z_1D, axis=0) * R * L / 2
    return F_friction


def get_step_deg(phi_deg):
    """Адаптивный шаг: 0.25° у пика Вибе, базовый вне."""
    phi_mod = phi_deg % 720.0
    if 330.0 <= phi_mod <= 420.0:
        return 0.25
    return params.d_phi_crank_deg


def run_transient(F_max=None, debug=False,
                  closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION):
    """Нестационарный расчёт для всех 4 конфигураций."""
    if F_max is None:
        F_max = params.F_max_debug if debug else params.F_max

    N = params.N_grid_transient
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N)
    phi_c, Z_c = setup_texture(params)

    omega = 2 * np.pi * params.n / 60.0
    U = omega * params.R
    N_SUB = params.n_sub_iterations

    # Предвычислить адаптивную сетку углов
    phi_list = []
    phi_cur = 0.0
    phi_total = 720.0 * params.n_cycles
    while phi_cur < phi_total - 1e-9:
        phi_list.append(phi_cur)
        phi_cur += get_step_deg(phi_cur)
    phi_crank_deg = np.array(phi_list)
    n_steps = len(phi_crank_deg)

    # Индексы последнего цикла: найти первый шаг >= 720*(n_cycles-1)
    last_cycle_start_deg = 720.0 * (params.n_cycles - 1)
    last_start = int(np.searchsorted(phi_crank_deg, last_cycle_start_deg))
    n_last = n_steps - last_start

    print(f"  Параметры: F_max={F_max/1e3:.0f} кН, "
          f"{n_steps} шагов (адаптивный: 0.25° пик / {params.d_phi_crank_deg}° вне), "
          f"N_SUB={N_SUB}, {params.n_cycles} цикла")

    n_cfg = len(CONFIGS)
    eps_x_all = np.zeros((n_cfg, n_steps))
    eps_y_all = np.zeros((n_cfg, n_steps))
    hmin_all = np.zeros((n_cfg, n_steps))
    pmax_all = np.zeros((n_cfg, n_steps))
    f_all = np.zeros((n_cfg, n_steps))
    Ftr_all = np.zeros((n_cfg, n_steps))
    Nloss_all = np.zeros((n_cfg, n_steps))
    Fx_hyd_all = np.zeros((n_cfg, n_steps))
    Fy_hyd_all = np.zeros((n_cfg, n_steps))
    cfg_times = []  # время каждого конфига

    for ic, cfg in enumerate(CONFIGS):
        eta = cfg["oil"]["eta_diesel"]
        alpha_pv = None  # PV отключена — overflow при 850 кН
        p_scale = 6.0 * eta * omega * (params.R / params.c) ** 2

        print(f"\n  [{ic+1}/{n_cfg}] {cfg['label']}...")
        t_cfg_start = _time.time()

        # Начальные условия
        ex = params.eps_x0 * params.c
        ey = params.eps_y0 * params.c
        vx, vy = 0.0, 0.0
        ax_prev, ay_prev = 0.0, 0.0
        P_prev = None
        contact_count = 0
        pv_fallback_count = 0

        def _solve_at(ex_, ey_, vx_, vy_, P_init_, phi_deg_):
            """Решить Reynolds в точке (ex_, ey_) и вернуть P, H, Fx, Fy."""
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
                phi_mod = phi_deg_ % 720.0
                near_peak = 330.0 <= phi_mod <= 420.0
                kw["alpha_pv"] = alpha_pv
                kw["p_scale"] = p_scale
                kw["relax_pv"] = 0.2 if near_peak else 0.4
                kw["max_outer_pv"] = 80 if near_peak else 50
            res = solve_reynolds(H_, d_phi, d_Z, params.R, params.L, **kw)
            P_ = res[0]
            n_outer_ = res[3] if len(res) == 4 else 0
            max_outer_ = kw.get("max_outer_pv", 0)
            Fx_, Fy_ = compute_hydro_forces(
                P_, p_scale, Phi_mesh, phi_1D, Z_1D, params.R, params.L)
            return P_, H_, Fx_, Fy_, n_outer_, max_outer_

        def _clamp(ex_, ey_, vx_, vy_):
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

        progress_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            phi_deg = phi_crank_deg[step] % 720.0
            dt_step = np.deg2rad(get_step_deg(phi_crank_deg[step])) / omega

            # Сохранить состояние на начало шага
            ex_n, ey_n = ex, ey
            vx_n, vy_n = vx, vy

            # --- Velocity Verlet с sub-iterations ---
            # Начальный predict
            ex_pred = ex_n + vx_n * dt_step + 0.5 * ax_prev * dt_step**2
            ey_pred = ey_n + vy_n * dt_step + 0.5 * ay_prev * dt_step**2
            ex_pred, ey_pred, vx_corr, vy_corr, clamped_p = _clamp(
                ex_pred, ey_pred, vx_n, vy_n)
            if clamped_p:
                contact_count += 1
                P_prev = None

            vx_corr, vy_corr = vx_n, vy_n  # reset for sub-iteration
            ax_new, ay_new = ax_prev, ay_prev

            for k in range(N_SUB):
                # Solve Reynolds at current predicted position
                P, H, Fx_hyd, Fy_hyd, n_outer, max_outer = _solve_at(
                    ex_pred, ey_pred, vx_corr, vy_corr, P_prev, phi_deg)

                # Пьезовязкость: откат к изовязкому при расходимости
                if alpha_pv is not None and max_outer > 0 and n_outer >= max_outer:
                    pv_fallback_count += 1
                    kw_iso = dict(
                        closure=closure, cavitation=cavitation,
                        omega=1.5, tol=1e-5, max_iter=50000,
                        P_init=None,
                        xprime=0.0, yprime=0.0, beta=2.0,
                    )
                    xp_, yp_, bt_ = squeeze_to_api_params(
                        -vx_corr, -vy_corr, params.c, omega, d_phi)
                    kw_iso["xprime"] = xp_
                    kw_iso["yprime"] = yp_
                    kw_iso["beta"] = bt_
                    eps_x_ = ex_pred / params.c
                    eps_y_ = ey_pred / params.c
                    H_fb = build_H_2d(eps_x_, eps_y_, Phi_mesh, Z_mesh, params,
                                      textured=cfg["textured"],
                                      phi_c_flat=phi_c, Z_c_flat=Z_c)
                    res_fb = solve_reynolds(H_fb, d_phi, d_Z, params.R, params.L,
                                           **kw_iso)
                    P = res_fb[0]
                    H = H_fb
                    Fx_hyd, Fy_hyd = compute_hydro_forces(
                        P, p_scale, Phi_mesh, phi_1D, Z_1D, params.R, params.L)

                P_prev = P

                # External load
                Fx_ext, Fy_ext = load_diesel(phi_deg, F_max=F_max)
                Fx_ext = float(Fx_ext)
                Fy_ext = float(Fy_ext)

                # New acceleration
                ax_new = (Fx_ext + Fx_hyd) / params.m_shaft
                ay_new = (Fy_ext + Fy_hyd) / params.m_shaft

                # Correct velocity
                vx_corr = vx_n + 0.5 * (ax_prev + ax_new) * dt_step
                vy_corr = vy_n + 0.5 * (ay_prev + ay_new) * dt_step

                # Re-predict for next sub-iteration (except last)
                if k < N_SUB - 1:
                    ex_pred = ex_n + vx_corr * dt_step + 0.5 * ax_new * dt_step**2
                    ey_pred = ey_n + vy_corr * dt_step + 0.5 * ay_new * dt_step**2
                    ex_pred, ey_pred, vx_corr, vy_corr, cl = _clamp(
                        ex_pred, ey_pred, vx_corr, vy_corr)
                    if cl:
                        contact_count += 1
                        P_prev = None

            # Accept
            ex, ey = ex_pred, ey_pred
            vx, vy = vx_corr, vy_corr
            ax_prev, ay_prev = ax_new, ay_new

            # Final clamp
            ex, ey, vx, vy, clamped_final = _clamp(ex, ey, vx, vy)
            if clamped_final:
                contact_count += 1
                P_prev = None

            # Характеристики
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

            if (step + 1) % progress_interval == 0:
                pct = 100 * (step + 1) / n_steps
                eps_now = np.hypot(ex, ey) / params.c
                print(f"    {pct:3.0f}%: φ={phi_deg:6.1f}°, "
                      f"|ε|={eps_now:.3f}, h_min={h_min*1e6:.1f} мкм, "
                      f"p_max={p_max/1e6:.1f} МПа")

        t_cfg = _time.time() - t_cfg_start
        cfg_times.append(t_cfg)
        print(f"    Контакт (clamp): {contact_count} из {n_steps} шагов")
        if pv_fallback_count > 0:
            print(f"    Пьезовязкость: {pv_fallback_count} откатов к изовязкому")
        print(f"    Время: {t_cfg:.1f} с")

    # Углы для последнего цикла
    phi_last = phi_crank_deg[last_start:]
    Fx_ext_last, Fy_ext_last = load_diesel(phi_last % 720.0, F_max=F_max)

    return {
        "phi_crank_deg": phi_crank_deg,
        "phi_last": phi_last,
        "last_start": last_start,
        "n_steps_per_cycle": n_last,
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
        "cfg_times": cfg_times,
    }
