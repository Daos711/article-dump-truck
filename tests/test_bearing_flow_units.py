"""Stage THD-0B — axial leakage unit-of-measure regression.

These tests are independent of ``reynolds_solver``: the helper
``compute_axial_leakage_m3_s`` operates on user-supplied ``P_dim`` /
``h_dim`` arrays. They exist specifically to defend against a recurrence
of the bug fixed in Stage THD-0B: the inlined Q formula in
``solve_and_compute`` was missing the ``2/L`` Jacobian that maps the
non-dimensional axial coordinate Z ∈ [-1, 1] back to the physical
``z = (L/2) * Z``. The bug under-counted axial leakage by a factor of
``2/L`` (≈ 25 for L = 80 mm BelAZ bearing), which fed back into the
THD outer loop as artificially low ``mdot`` and inflated ``T_eff``.

The first test compares the helper against a closed-form expression for
a synthetic parabolic pressure field with constant film thickness; the
second freezes the ``2/L`` factor against the legacy formula so a future
refactor cannot silently delete it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from models.bearing_model import compute_axial_leakage_m3_s

# numpy 2.x removed np.trapz in favour of np.trapezoid; older numpy
# only has np.trapz. Use whichever is available so the test runs on
# both the user's environment and the CI sandbox.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _build_synthetic_linear_field(*, P0: float, h0: float,
                                    N_phi: int = 64,
                                    N_Z: int = 33,
                                    phi_min: float = 0.0,
                                    phi_max: float = 2.0 * math.pi):
    """Synthetic field with closed-form axial leakage.

    P_dim(Z) = P0 * Z,   independent of phi.
    h_dim    = h0   (constant)
    Z ∈ [-1, 1].

    A linear pressure profile is differentiated exactly by both central
    and one-sided finite differences (np.gradient on uniform spacing),
    which lets us compare against the analytic answer at machine
    precision rather than FD-truncation precision.

    dP/dZ = P0 (constant). dP/dz = (2/L) * P0. At each boundary
    |∂p/∂z| = |2*P0/L|, and the closed-form leakage is

        Q_analytic = R * 2 * (h0**3 / (12 * eta)) * (2*|P0|/L)
                       * (phi_max - phi_min)
    """
    phi_1D = np.linspace(phi_min, phi_max, N_phi)
    Z_1D = np.linspace(-1.0, 1.0, N_Z)
    _, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    P_dim = P0 * Z_mesh
    h_dim = np.full_like(Z_mesh, h0)
    return phi_1D, Z_1D, P_dim, h_dim


def test_axial_leakage_uses_physical_z_coordinate():
    """Helper Q matches analytic formula to rtol <= 1e-10."""
    P0 = 5.0e6      # Pa
    h0 = 30e-6      # m
    eta = 0.010     # Pa*s
    R = 0.100       # m
    L = 0.080       # m
    phi_max = 2.0 * math.pi

    phi_1D, Z_1D, P_dim, h_dim = _build_synthetic_linear_field(
        P0=P0, h0=h0,
        N_phi=512, N_Z=129,
    )
    Q = compute_axial_leakage_m3_s(
        P_dim=P_dim, h_dim=h_dim,
        phi_1D=phi_1D, Z_1D=Z_1D,
        eta=eta, R=R, L=L,
    )

    # |dp/dz| = 2*|P0|/L at each end; total Q over both ends:
    Q_analytic = (
        R * 2.0 * (h0 ** 3 / (12.0 * eta)) * (2.0 * abs(P0) / L) * phi_max
    )
    rel = abs(Q - Q_analytic) / Q_analytic
    assert rel < 1e-10, (
        f"Q={Q:.6e}, expected {Q_analytic:.6e}, rel error {rel:.3e}"
    )


def test_q_factor_2_over_L_is_present():
    """Ratio guard: helper Q is exactly 2/L bigger than the legacy
    formula that omitted the Jacobian.

    This freezes the units fix so a future refactor cannot silently
    drop the 2/L factor again.
    """
    P0 = 5.0e6
    h0 = 30e-6
    eta = 0.010
    R = 0.100
    L = 0.080
    phi_min = 0.0
    phi_max = 2.0 * math.pi

    phi_1D, Z_1D, P_dim, h_dim = _build_synthetic_linear_field(
        P0=P0, h0=h0,
        N_phi=256, N_Z=65,
    )

    Q_helper = compute_axial_leakage_m3_s(
        P_dim=P_dim, h_dim=h_dim,
        phi_1D=phi_1D, Z_1D=Z_1D,
        eta=eta, R=R, L=L,
    )

    # Legacy formula: same as the helper but WITHOUT the 2/L scaling.
    dZ = float(Z_1D[1] - Z_1D[0])
    dP_dZ = np.gradient(P_dim, dZ, axis=0)
    q_z0 = h_dim[0, :] ** 3 / (12.0 * eta) * np.abs(dP_dZ[0, :]) * R
    q_z1 = h_dim[-1, :] ** 3 / (12.0 * eta) * np.abs(dP_dZ[-1, :]) * R
    Q_legacy = float(_trapz(q_z0, phi_1D) + _trapz(q_z1, phi_1D))

    ratio = Q_helper / Q_legacy
    assert math.isclose(ratio, 2.0 / L, rel_tol=1e-10), (
        f"Q_new / Q_legacy = {ratio:.6e}, expected 2/L = {2.0/L:.6e}"
    )


def test_axial_leakage_zero_for_zero_pressure():
    """Sanity: zero pressure field => zero leakage."""
    phi_1D, Z_1D, _, h_dim = _build_synthetic_linear_field(
        P0=0.0, h0=20e-6, N_phi=33, N_Z=17,
    )
    P_dim = np.zeros_like(h_dim)
    Q = compute_axial_leakage_m3_s(
        P_dim=P_dim, h_dim=h_dim,
        phi_1D=phi_1D, Z_1D=Z_1D,
        eta=0.010, R=0.05, L=0.04,
    )
    assert Q == 0.0


def test_axial_leakage_scales_inversely_with_eta():
    """Sanity: doubling viscosity halves leakage (linear scaling)."""
    P0 = 1.0e6
    h0 = 20e-6
    R = 0.05
    L = 0.04
    phi_1D, Z_1D, P_dim, h_dim = _build_synthetic_linear_field(
        P0=P0, h0=h0, N_phi=129, N_Z=33,
    )
    Q1 = compute_axial_leakage_m3_s(P_dim=P_dim, h_dim=h_dim,
                                      phi_1D=phi_1D, Z_1D=Z_1D,
                                      eta=0.010, R=R, L=L)
    Q2 = compute_axial_leakage_m3_s(P_dim=P_dim, h_dim=h_dim,
                                      phi_1D=phi_1D, Z_1D=Z_1D,
                                      eta=0.020, R=R, L=L)
    assert math.isclose(Q1 / Q2, 2.0, rel_tol=1e-10)
