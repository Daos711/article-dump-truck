"""Equilibrium wrapper for co-design (ТЗ §6 Phase E1/E2).

Использует существующий driver `models.magnetic_equilibrium.find_equilibrium`
(он же не «магнитный» — это generic 2D Newton-Raphson на (X,Y) с
inject-able H_and_force и mag_model). Магнит на co-design phase отключён
(K_mag=0).

Critical (clarification 4): NR seed FIXED — (X0=0.0, Y0=-0.4) для всех
профилей и обоих case (smooth/textured), независимо от порядка top-6.
Это убирает зависимость результата от очерёдности обхода.

Pair invariant (ТЗ §13.13): smooth и textured equilibrium для одного
профиля используют ОДИН И ТОТ ЖЕ bore_profile_fn (через
make_H_pair_builders).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .magnetic_equilibrium import (
    find_equilibrium, is_accepted, result_to_dict, EquilibriumResult,
)
from .magnetic_force import RadialUnloadForceModel
from .coexp_pairing import PairBuilders, make_H_pair_builders
from .coexp_schema import (
    EQUILIBRIUM_NR_SEED_X0, EQUILIBRIUM_NR_SEED_Y0,
    DEFAULT_TOL_ACCEPT, DEFAULT_STEP_CAP, DEFAULT_EPS_MAX,
    ExperimentSpec,
)


def _make_H_and_force_for_case(
        H_builder: Callable[..., np.ndarray],
        Phi_mesh: np.ndarray, Z_mesh: np.ndarray,
        phi_1D: np.ndarray, Z_1D: np.ndarray,
        d_phi: float, d_Z: float,
        R: float, L: float, c: float, sigma: float,
        eta: float, omega: float,
        ps_solver: Callable[..., Tuple],
) -> Callable[[float, float], Tuple]:
    """Замыкание H_and_force(X, Y) → (Fx, Fy, h_min, p_max, cav, fr, P, θ).

    Использует bore-aware H_builder(eps, Phi_mesh, Z_mesh) — eps здесь
    определяется через NR-state как ε = sqrt(X²+Y²) с фазой
    arctan(Y, X) … НО на самом деле в существующем equilibrium driver'е
    eps входит как eccentricity ratio с распределением по cos/sin
    отдельно: H = 1 + X cos(phi) + Y sin(phi) + bore + tex.
    Поэтому H_and_force внутри строит H напрямую от X,Y, не через eps.
    """
    p_scale = 6.0 * eta * omega * (R / c) ** 2

    def H_and_force(X: float, Y: float) -> Tuple:
        # Inline bore-aware H(X, Y) build:
        H0 = 1.0 + float(X) * np.cos(Phi_mesh) + float(Y) * np.sin(Phi_mesh)
        # NB: bore_fn captured внутри H_builder; передаём через trick:
        bore_fn = getattr(H_builder, "__bore_profile_fn__", None)
        if bore_fn is not None:
            H0 = H0 + bore_fn(Phi_mesh)
        H0_reg = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
        if getattr(H_builder, "__case__", "smooth") == "textured":
            # Reuse the textured builder by extracting its texture
            # piece. To stay generic, we delegate to the builder:
            # (it accepts (eps, Phi, Z) but we substitute "smooth-eq H0"
            # through a trick — actually easier: rebuild from same
            # texture relief data via a helper attribute.)
            #
            # Простое и правильное: H_builder на eps=hypot(X,Y) даст
            # H использующий cos(phi)·eps_only — НЕ X·cos+Y·sin.
            # Поэтому повторно строим textured H вручную через атрибуты.
            relief_data = getattr(H_builder, "__relief__", None)
            if relief_data is None:
                raise RuntimeError(
                    "textured H_builder must expose __relief__ for "
                    "X,Y-aware equilibrium build")
            H = relief_data["fn"](
                H0_reg, relief_data["depth_nondim"], Phi_mesh, Z_mesh,
                relief_data["phi_c"], relief_data["Z_c"],
                relief_data["a_Z"], relief_data["a_phi"],
                profile=relief_data["profile"])
        else:
            H = H0_reg

        P, theta, _, _ = ps_solver(
            H, d_phi, d_Z, R, L, tol=1e-6, max_iter=10_000_000)
        P_dim = P * p_scale
        Fx = -np.trapezoid(
            np.trapezoid(P_dim * np.cos(Phi_mesh), phi_1D, axis=1),
            Z_1D, axis=0) * R * L / 2.0
        Fy = -np.trapezoid(
            np.trapezoid(P_dim * np.sin(Phi_mesh), phi_1D, axis=1),
            Z_1D, axis=0) * R * L / 2.0
        h_dim = H * c
        h_min = float(np.min(h_dim))
        p_max = float(np.max(P_dim))
        cav_frac = float(np.mean(theta < 1.0 - 1e-6))
        tau_c = eta * omega * R / h_dim
        friction = float(
            np.sum(tau_c) * R * (2.0 * np.pi / H.shape[1])
            * L * (2.0 / H.shape[0]) / 2.0)
        return (float(Fx), float(Fy), h_min, p_max, cav_frac, friction,
                P, theta)

    return H_and_force


def attach_relief_to_textured_builder(
        textured_fn: Callable, R: float, L: float, c: float,
        texture_spec_dict: Dict[str, Any],
        texture_relief_fn: Optional[Callable] = None,
) -> None:
    """Прикрепить __relief__ metadata к textured closure для NR-режима.

    Это нужно потому что NR работает в (X,Y)-пространстве, а pair builder
    из coexp_pairing удобен в (eps, Phi, Z)-режиме. Для equilibrium-фазы
    мы пересобираем H через X,Y и вынуждены повторно знать reference на
    relief-параметры.
    """
    if texture_relief_fn is None:
        from reynolds_solver.utils import (
            create_H_with_ellipsoidal_depressions as texture_relief_fn,
        )
    from .coexp_pairing import _build_texture_centers
    phi_c, Z_c, a_phi, a_Z, depth_m, profile = _build_texture_centers(
        texture_spec_dict, R, L)
    textured_fn.__relief__ = dict(
        fn=texture_relief_fn,
        phi_c=phi_c, Z_c=Z_c, a_phi=a_phi, a_Z=a_Z,
        depth_nondim=depth_m / c,
        profile=profile,
    )


def solve_equilibrium_pair(
        experiment: ExperimentSpec,
        Phi_mesh: np.ndarray, Z_mesh: np.ndarray,
        phi_1D: np.ndarray, Z_1D: np.ndarray,
        d_phi: float, d_Z: float,
        R: float, L: float, c: float, sigma: float,
        eta: float, omega: float, F0: float,
        Wy_share: float,
        ps_solver: Callable[..., Tuple],
        texture_relief_fn: Optional[Callable] = None,
        tol_accept: float = DEFAULT_TOL_ACCEPT,
        step_cap: float = DEFAULT_STEP_CAP,
        eps_max: float = DEFAULT_EPS_MAX,
        max_iter: int = 80,
) -> Dict[str, Any]:
    """Solve smooth+textured equilibrium на ОДНОМ bore profile.

    NR seed для обоих case — (EQUILIBRIUM_NR_SEED_X0/Y0). Магнит OFF.

    max_iter forwards to find_equilibrium; diagnostic runs may raise it
    (e.g. 200) to let NR escape high-ε regions. Pure passthrough —
    никаких изменений NR-логики.
    """
    pair = make_H_pair_builders(
        experiment, R, L, c, sigma,
        texture_relief_fn=texture_relief_fn)
    attach_relief_to_textured_builder(
        pair.textured, R, L, c,
        experiment.texture.as_dict(),
        texture_relief_fn=texture_relief_fn)

    W_applied = np.array([0.0, -float(Wy_share) * float(F0)])
    zero_mag = RadialUnloadForceModel(K_mag=0.0)

    def _solve(case_builder):
        H_and_force = _make_H_and_force_for_case(
            case_builder, Phi_mesh, Z_mesh, phi_1D, Z_1D,
            d_phi, d_Z, R, L, c, sigma, eta, omega, ps_solver)
        r = find_equilibrium(
            H_and_force, zero_mag, W_applied,
            X0=EQUILIBRIUM_NR_SEED_X0, Y0=EQUILIBRIUM_NR_SEED_Y0,
            tol=tol_accept, step_cap=step_cap, eps_max=eps_max,
            tol_accept=tol_accept, max_iter=int(max_iter))
        return r

    rs = _solve(pair.smooth)
    rt = _solve(pair.textured)

    return dict(
        experiment_id=pair.experiment_id,
        profile_hash=pair.profile_hash,
        profile_spec=experiment.profile.as_dict(),
        smooth=result_to_dict(rs),
        textured=result_to_dict(rt),
        smooth_accepted=bool(is_accepted(rs, tol_accept)),
        textured_accepted=bool(is_accepted(rt, tol_accept)),
    )


__all__ = [
    "solve_equilibrium_pair",
    "attach_relief_to_textured_builder",
]
