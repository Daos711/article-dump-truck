"""Общие функции модели подшипника скольжения.

Сетка, зазор (гладкий/текстурированный), решение уравнения Рейнольдса,
вычисление интегральных характеристик (F, μ, Q, h_min, p_max).
"""
import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

DEFAULT_CLOSURE = "laminar"
DEFAULT_CAVITATION = "half_sommerfeld"


def setup_grid(N):
    """Создать сетку φ ∈ [0, 2π], Z ∈ [-1, 1] размером (N_Z, N_phi).

    Returns
    -------
    phi_1D : (N,) — узлы по φ
    Z_1D   : (N,) — узлы по Z (безразмерные, -1..1)
    Phi_mesh, Z_mesh : (N_Z, N_phi) — меш-гриды (стандартный meshgrid)
    d_phi, d_Z : шаги сетки
    """
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z_1D = np.linspace(-1, 1, N)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)  # (N_Z, N_phi)
    return phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z


def setup_texture(params):
    """Создать массивы центров углублений из параметров подшипника.

    Parameters
    ----------
    params : module с атрибутами R, L, phi_start_deg, phi_end_deg,
             N_phi_tex, N_Z_tex

    Returns
    -------
    phi_c_flat : (M,) — φ-координаты центров (рад)
    Z_c_flat   : (M,) — Z-координаты центров (безразмерные, -1..1)
    """
    phi_start = np.deg2rad(params.phi_start_deg)
    phi_end = np.deg2rad(params.phi_end_deg)
    phi_centers = np.linspace(phi_start, phi_end, params.N_phi_tex)
    Z_centers = np.linspace(-0.8, 0.8, params.N_Z_tex)
    phi_grid, Z_grid = np.meshgrid(phi_centers, Z_centers)
    return phi_grid.ravel(), Z_grid.ravel()


def make_H(epsilon, Phi_mesh, Z_mesh, params, textured=False,
           phi_c_flat=None, Z_c_flat=None):
    """Построить безразмерный зазор H (гладкий или текстурированный).

    H0 = 1 + ε·cos(φ)  — гладкий профиль
    При textured=True добавляются эллипсоидальные углубления.
    """
    H0 = 1.0 + epsilon * np.cos(Phi_mesh)
    if not textured:
        return H0

    A = 2 * params.a_dim / params.L      # полуось по Z (безразмерная)
    B = params.b_dim / params.R           # полуось по φ (в радианах)
    H_p = params.h_p / params.c           # безразмерная глубина

    H = create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A, B
    )
    return H


def solve_and_compute(H, d_phi, d_Z, R, L, eta, n, c,
                      phi_1D, Z_1D, Phi_mesh, P_init=None,
                      closure=DEFAULT_CLOSURE,
                      cavitation=DEFAULT_CAVITATION,
                      alpha_pv=None):
    """Решить уравнение Рейнольдса и вычислить интегральные характеристики.

    H, Phi_mesh, Z_mesh имеют shape (N_Z, N_phi) — формат солвера.

    Parameters
    ----------
    alpha_pv : float or None
        Коэффициент пьезовязкости Баруса (Па⁻¹). При None — изовязкий режим.

    Returns
    -------
    P     : (N_Z, N_phi) — поле давления (безразмерное)
    F     : float — несущая способность (Н)
    mu    : float — коэффициент трения
    Q     : float — расход смазки (м³/с)
    h_min : float — минимальный зазор (м)
    p_max : float — максимальное давление (Па)
    F_tr  : float — сила трения (Н)
    n_outer : int — число внешних итераций пьезовязкости (0 для изовязкого)
    """
    omega = 2 * np.pi * n / 60.0  # угловая скорость (рад/с)

    # Масштаб давления: p* = 6·η·ω·(R/c)² (стандартная безразмерная форма)
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    solver_kw = dict(
        closure=closure,
        cavitation=cavitation,
        omega=1.5,
        tol=1e-5,
        max_iter=50000,
        P_init=P_init,
    )
    if alpha_pv is not None:
        solver_kw["alpha_pv"] = alpha_pv
        solver_kw["p_scale"] = p_scale

    result = solve_reynolds(H, d_phi, d_Z, R, L, **solver_kw)

    # Пьезовязкий солвер возвращает 4 элемента (P, delta, n_iter, n_outer),
    # изовязкий — 3 (P, delta, n_iter).
    if len(result) == 4:
        P, residual, n_iter, n_outer = result
    else:
        P, residual, n_iter = result
        n_outer = 0

    # Размерное давление
    P_dim = P * p_scale  # Па

    # Несущая способность: интегрирование давления
    # Меши (N_Z, N_phi): axis=0 — Z, axis=1 — φ
    # F_x = -∫∫ p·cos(φ) R dφ (dZ·L/2)
    # Интегрируем по φ (axis=1), потом по Z (axis=0)
    # Z ∈ [-1,1], размерный = Z·L/2
    Fx = -np.trapz(np.trapz(P_dim * np.cos(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    Fy = -np.trapz(np.trapz(P_dim * np.sin(Phi_mesh), phi_1D, axis=1),
                   Z_1D, axis=0) * R * L / 2
    F = np.sqrt(Fx**2 + Fy**2)

    # Сила трения на поверхности вала
    h_dim = H * c  # размерный зазор
    tau_couette = eta * omega * R / h_dim
    # Градиент давления по φ (axis=1)
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

    return P, F, mu_val, Q, h_min, p_max, F_friction, n_outer
