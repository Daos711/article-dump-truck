"""Общие функции модели подшипника скольжения.

Сетка, зазор (гладкий/текстурированный), решение уравнения Рейнольдса,
вычисление интегральных характеристик (F, μ, Q, h_min, p_max).
"""
import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

DEFAULT_CLOSURE = "laminar"
DEFAULT_CAVITATION = "half_sommerfeld"


def setup_grid(N_phi, N_Z=None):
    """Создать сетку φ ∈ [0, 2π), Z ∈ [-1, 1] размером (N_Z, N_phi).

    Parameters
    ----------
    N_phi : int — узлов по φ (endpoint=False, периодическое)
    N_Z   : int or None — узлов по Z (если None, = N_phi)

    Returns
    -------
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z
    """
    if N_Z is None:
        N_Z = N_phi
    phi_1D = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    Z_1D = np.linspace(-1, 1, N_Z)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)  # (N_Z, N_phi)
    return phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z


def setup_texture(params):
    """Создать массивы центров углублений из параметров подшипника.

    Поддерживает зону через 0°: если phi_start_deg > phi_end_deg,
    лунки размещаются от phi_start до 360° и от 0° до phi_end.

    Returns
    -------
    phi_c_flat : (M,) — φ-координаты центров (рад)
    Z_c_flat   : (M,) — Z-координаты центров (безразмерные, -1..1)
    """
    phi_s = params.phi_start_deg
    phi_e = params.phi_end_deg
    if phi_s <= phi_e:
        # Обычная зона (например 90°→270°)
        phi_centers = np.linspace(np.deg2rad(phi_s), np.deg2rad(phi_e),
                                  params.N_phi_tex)
    else:
        # Зона через 0° (например 270°→90° = 270°→360° + 0°→90°)
        span = (360 - phi_s) + phi_e  # полная ширина в градусах
        angles_deg = np.linspace(0, span, params.N_phi_tex) + phi_s
        angles_deg = angles_deg % 360.0
        phi_centers = np.deg2rad(angles_deg)

    Z_centers = np.linspace(-0.8, 0.8, params.N_Z_tex)
    phi_grid, Z_grid = np.meshgrid(phi_centers, Z_centers)
    return phi_grid.ravel(), Z_grid.ravel()


def make_H(epsilon, Phi_mesh, Z_mesh, params, textured=False,
           phi_c_flat=None, Z_c_flat=None, profile="sqrt"):
    """Построить безразмерный зазор H (гладкий или текстурированный).

    H0 = 1 + ε·cos(φ)  — гладкий профиль
    profile : "sqrt" (эллипсоид) или "smoothcap" ((1-r²)²)

    Угловая конвенция (проверено 2026-04):
        H_min при φ = π (180°),  H_max при φ = 0.
        dH/dφ = −ε·sin(φ).
        Конвергентная зона: φ ∈ (0, π)   — зазор убывает.
        Дивергентная зона:  φ ∈ (π, 2π)  — зазор растёт.
        Текстуру для повышения W размещать в дивергентной зоне (180°–360°).
    """
    H0 = 1.0 + epsilon * np.cos(Phi_mesh)
    H0 = np.sqrt(H0**2 + (params.sigma / params.c)**2)  # регуляризация шероховатости
    if not textured:
        return H0

    A = 2 * params.a_dim / params.L      # полуось по Z (безразмерная)
    B = params.b_dim / params.R           # полуось по φ (в радианах)
    H_p = params.h_p / params.c           # безразмерная глубина

    H = create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A, B,
        profile=profile)
    return H


def solve_and_compute(H, d_phi, d_Z, R, L, eta, n, c,
                      phi_1D, Z_1D, Phi_mesh, P_init=None,
                      closure=DEFAULT_CLOSURE,
                      cavitation=DEFAULT_CAVITATION,
                      alpha_pv=None,
                      subcell_quad=False,
                      H_smooth=None,
                      texture_params=None):
    """Решить уравнение Рейнольдса и вычислить интегральные характеристики.

    H, Phi_mesh, Z_mesh имеют shape (N_Z, N_phi) — формат солвера.

    Parameters
    ----------
    alpha_pv : float or None
        Коэффициент пьезовязкости Баруса (Па⁻¹). При None — изовязкий режим.
        НЕ совместим с cavitation="payvar_salant" (игнорируется).

    Returns
    -------
    P       : (N_Z, N_phi) — поле давления (безразмерное)
    F       : float — несущая способность (Н)
    mu      : float — коэффициент трения
    Q       : float — расход смазки (м³/с)
    h_min   : float — минимальный зазор (м)
    p_max   : float — максимальное давление (Па)
    F_tr    : float — сила трения (Н)
    n_outer : int — число внешних итераций (PV) или итераций PS (0 для HS)
    theta   : ndarray or None — поле заполнения θ (только payvar_salant)
    cav_frac: float — доля кавитированной зоны (0.0 для half_sommerfeld)
    """
    omega = 2 * np.pi * n / 60.0  # угловая скорость (рад/с)

    # Масштаб давления: p* = 6·η·ω·(R/c)² (стандартная безразмерная форма)
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    is_ps = (cavitation == "payvar_salant")

    if is_ps:
        # Payvar-Salant: минимальный набор kwargs, без subcell_quad/PV
        solver_kw = dict(
            closure=closure,
            cavitation=cavitation,
            tol=1e-6,
            max_iter=50000,
            P_init=P_init,
            phi_1D=phi_1D,
            Z_1D=Z_1D,
        )
    else:
        solver_kw = dict(
            closure=closure,
            cavitation=cavitation,
            tol=1e-5,
            max_iter=50000,
            P_init=P_init,
            subcell_quad=subcell_quad,
            H_smooth=H_smooth,
            texture_params=texture_params,
            phi_1D=phi_1D,
            Z_1D=Z_1D,
            return_converged=True,
        )
        if alpha_pv is not None:
            solver_kw["alpha_pv"] = alpha_pv
            solver_kw["p_scale"] = p_scale
            solver_kw["relax_pv"] = 0.4
            solver_kw["max_outer_pv"] = 50

    result = solve_reynolds(H, d_phi, d_Z, R, L, **solver_kw)

    theta = None
    cav_frac = 0.0

    if is_ps:
        # Payvar-Salant: формат определяется по длине result
        # Возможные варианты:
        #   3: (P, theta, (residual, n_iter))  — вложенный tuple
        #   5: (P, theta, residual, n_outer, n_inner) — как JFO
        #   4: (P, theta, residual, n_iter)
        if len(result) == 3:
            P, theta, info = result
            residual, n_iter = info
        elif len(result) == 5:
            P, theta, residual, _n_outer, n_iter = result
        elif len(result) == 4:
            P, theta, residual, n_iter = result
        else:
            raise ValueError(
                f"Payvar-Salant: неожиданный формат result, "
                f"len={len(result)}, types={[type(r).__name__ for r in result]}")
        n_outer = n_iter
        converged = True
        cav_frac = float(np.mean(theta < 1.0))
    elif alpha_pv is not None:
        # PV-путь: (P, delta, n_iter, n_outer)
        P, residual, n_iter, n_outer = result
        converged = True  # PV-path пока не возвращает converged
    else:
        # Стандартный: (P, delta, n_iter, converged)
        P, residual, n_iter, converged = result
        n_outer = 0

    if not converged:
        import warnings
        warnings.warn(f"SOR не сошёлся: delta={residual:.2e}, n_iter={n_iter}")

    # Размерное давление
    P_dim = P * p_scale  # Па

    # Несущая способность: интегрирование давления
    # Меши (N_Z, N_phi): axis=0 — Z, axis=1 — φ
    Fx = -np.trapz(np.trapz(P_dim * np.cos(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    Fy = -np.trapz(np.trapz(P_dim * np.sin(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    F = np.sqrt(Fx**2 + Fy**2)

    # Сила трения на поверхности вала
    h_dim = H * c  # размерный зазор
    tau_couette = eta * omega * R / h_dim
    dP_dphi = np.gradient(P_dim, phi_1D[1] - phi_1D[0], axis=1)
    tau_pressure = h_dim / 2.0 * dP_dphi / R
    tau = tau_couette + tau_pressure

    F_friction = np.trapz(np.trapz(np.abs(tau), phi_1D, axis=1),
                          Z_1D, axis=0) * R * L / 2

    # Коэффициент трения
    mu_val = F_friction / max(F, 1.0) if F > 1.0 else 0.0

    # Расход смазки (утечка с торцов Z=-1 и Z=1)
    dP_dZ = np.gradient(P_dim, Z_1D[1] - Z_1D[0], axis=0)
    q_z0 = h_dim[0, :] ** 3 / (12.0 * eta) * np.abs(dP_dZ[0, :]) * R
    q_z1 = h_dim[-1, :] ** 3 / (12.0 * eta) * np.abs(dP_dZ[-1, :]) * R
    Q = (np.trapz(q_z0, phi_1D) + np.trapz(q_z1, phi_1D))

    h_min = np.min(h_dim)
    p_max = np.max(P_dim)

    return P, F, mu_val, Q, h_min, p_max, F_friction, n_outer, theta, cav_frac
