"""Runtime helpers for coexp scripts: PS import, grid, case metrics.

Локализует все side-effect'ы (GPU PS, texture relief) в одном месте,
чтобы CLI-скрипты оставались тонкими и тестопригодными.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import pump_params as params
from config.oil_properties import MINERAL_OIL


def load_ps_solver() -> Callable:
    try:
        from reynolds_solver.cavitation.payvar_salant import (
            solve_payvar_salant_gpu,
        )
        return solve_payvar_salant_gpu
    except ImportError:
        from reynolds_solver.cavitation.payvar_salant import (
            solve_payvar_salant_cpu,
        )
        return solve_payvar_salant_cpu


def load_relief_fn() -> Callable:
    from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
    return create_H_with_ellipsoidal_depressions


def mineral_constants():
    """Return (eta, omega, p_scale, F0, R, L, c, sigma)."""
    R = params.R
    L = params.L
    c = params.c
    sigma = params.sigma
    eta = MINERAL_OIL["eta_pump"]
    omega = 2.0 * np.pi * params.n / 60.0
    p_scale = 6.0 * eta * omega * (R / c) ** 2
    F0 = p_scale * R * L
    return dict(R=R, L=L, c=c, sigma=sigma,
                eta=eta, omega=omega,
                p_scale=p_scale, F0=F0)


def make_grid(N_phi: int, N_Z: int):
    phi = np.linspace(0.0, 2.0 * np.pi, int(N_phi), endpoint=False)
    Z = np.linspace(-1.0, 1.0, int(N_Z))
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    Phi, Zm = np.meshgrid(phi, Z)
    return phi, Z, Phi, Zm, d_phi, d_Z


def solve_case_metrics(
        H: np.ndarray, Phi: np.ndarray,
        phi_1D: np.ndarray, Z_1D: np.ndarray,
        d_phi: float, d_Z: float,
        constants: Dict[str, float],
        ps_solver: Callable,
) -> Dict[str, float]:
    """Run PS → integrate → return metrics dict (h_min, p_max, friction, cav_frac)."""
    R = constants["R"]
    L = constants["L"]
    c = constants["c"]
    eta = constants["eta"]
    omega = constants["omega"]
    p_scale = constants["p_scale"]

    P, theta, _, _ = ps_solver(
        H, d_phi, d_Z, R, L, tol=1e-6, max_iter=10_000_000)
    P_dim = P * p_scale
    Fx = -np.trapezoid(
        np.trapezoid(P_dim * np.cos(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R * L / 2.0
    Fy = -np.trapezoid(
        np.trapezoid(P_dim * np.sin(Phi), phi_1D, axis=1),
        Z_1D, axis=0) * R * L / 2.0
    h_dim = H * c
    h_min = float(np.min(h_dim))
    p_max = float(np.max(P_dim))
    cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    tau_c = eta * omega * R / h_dim
    friction = float(
        np.sum(tau_c) * R * (2.0 * np.pi / H.shape[1])
        * L * (2.0 / H.shape[0]) / 2.0)
    return dict(
        Fx=float(Fx), Fy=float(Fy),
        h_min=h_min, p_max=p_max,
        cav_frac=cav_frac, friction=friction,
    )


def git_sha() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None
