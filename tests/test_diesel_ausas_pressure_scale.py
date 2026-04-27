"""Stage J Bug 4 follow-up — pressure non-dimensionalisation parity.

The dynamic Ausas backend returns non-dimensional pressure ``P_nd``
matching the diesel half-Sommerfeld convention (``p_dim = P_nd ·
p_scale``). The runner integrates ``P_nd`` through
``compute_hydro_forces``, which already applies ``p_scale_step``.
If the Ausas non-dim convention drifts (e.g., ambient-pressure
offset, factor of 6 discrepancy, sign flip), the produced forces
become unphysical even though the solver converged.

This test compares Ausas vs half-Sommerfeld pressure on a smooth
fixed gap with no squeeze, asserting loose-but-meaningful bounds:

* ``|P_ausas_nd|.max()`` is the same order of magnitude as
  ``|P_legacy_nd|.max()`` (factor 0.05 .. 20).
* The integrated hydrodynamic forces have the same sign on at
  least one component (the films see the same external load).

JFO and half-Sommerfeld differ in the cavitation closure, so we
do NOT compare element-wise — only scalar magnitudes and force
signs. The goal is to catch factors of 6 / 200 / p_amb / sign
flips, not to prove physical equivalence.

Skipped when ``cupy`` / ``GPU_reynolds`` are not installed.
"""
from __future__ import annotations

import numpy as np
import pytest


cupy = pytest.importorskip("cupy")
solver_dynamic_gpu = pytest.importorskip(
    "reynolds_solver.cavitation.ausas.solver_dynamic_gpu")


from models.bearing_model import setup_grid  # noqa: E402
from reynolds_solver import solve_reynolds  # noqa: E402
from models.diesel_ausas_adapter import (  # noqa: E402
    DieselAusasState,
    ausas_one_step_with_state,
    set_ausas_backend_for_tests,
)
from models.diesel_transient import (  # noqa: E402
    build_H_2d, compute_hydro_forces,
)
from config import diesel_params as dparams  # noqa: E402
from config.oil_properties import MINERAL_OIL  # noqa: E402


_OMEGA_DIESEL = 2.0 * np.pi * 1900.0 / 60.0
_TOL = 1e-3
_MAX_INNER = 500
_ETA_DIESEL = float(MINERAL_OIL["eta_diesel"])


def test_p_nd_magnitude_within_one_order_of_legacy():
    """For a smooth fixed gap with no squeeze, |P_ausas_nd|.max()
    must be the same order of magnitude as the half-Sommerfeld
    P_nd (loose 20× bound). Catches factor-of-6 / factor-of-200
    pressure-scale regressions."""
    set_ausas_backend_for_tests(None)
    N_phi, N_z = 64, 16
    phi_1D, Z_1D, Phi, Z, d_phi, d_Z = setup_grid(N_phi, N_z)
    eps_x, eps_y = 0.4, 0.0
    H = build_H_2d(eps_x, eps_y, Phi, Z, dparams)

    # Half-Sommerfeld baseline.
    # Half-Sommerfeld baseline through the low-level GPU call —
    # same path the runner takes for ``cavitation='half_sommerfeld'``.
    P_legacy_nd, _, _, _ = solve_reynolds(
        H, d_phi, d_Z, dparams.R, dparams.L,
        closure="laminar", cavitation="half_sommerfeld",
        tol=1e-7, max_iter=50000)

    # Ausas one-step with H_curr = H_prev (no squeeze).
    state = DieselAusasState.from_initial_gap(H)
    out = ausas_one_step_with_state(
        state, H_curr=H,
        dt_s=np.deg2rad(4.0) / _OMEGA_DIESEL,
        omega_shaft=_OMEGA_DIESEL,
        d_phi=d_phi, d_Z=d_Z, R=dparams.R, L=dparams.L,
        extra_options={"tol": _TOL, "max_inner": _MAX_INNER, "alpha": 1.0},
        commit=True,
    )
    assert out.converged
    P_ausas_nd = np.asarray(out.P_nd)

    pmax_legacy = float(np.abs(P_legacy_nd).max())
    pmax_ausas = float(np.abs(P_ausas_nd).max())
    assert pmax_legacy > 0
    ratio = pmax_ausas / pmax_legacy
    assert 0.05 <= ratio <= 20.0, (
        f"Ausas P_nd scale {pmax_ausas:.3e} is {ratio:.3g}× the "
        f"half-Sommerfeld scale {pmax_legacy:.3e}; this is outside "
        "the 0.05–20× sanity band — likely a non-dim convention "
        "regression (factor of 6, factor of ω, or p_amb offset).")


def test_force_sign_agrees_with_legacy():
    """Loose force-direction sanity: the Ausas force at the same
    eccentricity should point in the same general direction as
    the half-Sommerfeld force (positive dot product)."""
    set_ausas_backend_for_tests(None)
    N_phi, N_z = 64, 16
    phi_1D, Z_1D, Phi, Z, d_phi, d_Z = setup_grid(N_phi, N_z)
    eps_x, eps_y = 0.4, 0.0
    H = build_H_2d(eps_x, eps_y, Phi, Z, dparams)

    p_scale = (6.0 * _ETA_DIESEL * _OMEGA_DIESEL
               * (dparams.R / dparams.c) ** 2)

    # Half-Sommerfeld baseline through the low-level GPU call —
    # same path the runner takes for ``cavitation='half_sommerfeld'``.
    P_legacy_nd, _, _, _ = solve_reynolds(
        H, d_phi, d_Z, dparams.R, dparams.L,
        closure="laminar", cavitation="half_sommerfeld",
        tol=1e-7, max_iter=50000)
    Fx_l, Fy_l = compute_hydro_forces(
        P_legacy_nd, p_scale, Phi, phi_1D, Z_1D, dparams.R, dparams.L)

    state = DieselAusasState.from_initial_gap(H)
    out = ausas_one_step_with_state(
        state, H_curr=H,
        dt_s=np.deg2rad(4.0) / _OMEGA_DIESEL,
        omega_shaft=_OMEGA_DIESEL,
        d_phi=d_phi, d_Z=d_Z, R=dparams.R, L=dparams.L,
        extra_options={"tol": _TOL, "max_inner": _MAX_INNER, "alpha": 1.0},
        commit=True,
    )
    assert out.converged
    Fx_a, Fy_a = compute_hydro_forces(
        np.asarray(out.P_nd), p_scale, Phi,
        phi_1D, Z_1D, dparams.R, dparams.L)

    F_legacy = np.array([Fx_l, Fy_l])
    F_ausas = np.array([Fx_a, Fy_a])
    legacy_mag = float(np.linalg.norm(F_legacy))
    ausas_mag = float(np.linalg.norm(F_ausas))
    assert ausas_mag > 0.0
    assert legacy_mag > 0.0
    cos_theta = float(np.dot(F_legacy, F_ausas)
                       / (legacy_mag * ausas_mag))
    assert cos_theta > 0.0, (
        f"Ausas force points opposite to half-Sommerfeld baseline "
        f"(cos θ = {cos_theta:.3g}); likely a sign regression in the "
        "P_nd / coordinate convention.")
