"""Stage J Bug 4 follow-up — force-direction sanity at the adapter
level (TZ §3.4).

A direct test that does NOT go through the full ``run_transient``
mechanics. Build a bearing at a known modest eccentricity, take ONE
Ausas one-step (``H_curr = H_prev`` so the squeeze term is zero
and the result is the steady JFO pressure for that gap), integrate
the hydrodynamic force, and assert that:

* ``|F_hyd|`` is NOT near zero (the film actually carries load);
* ``cos(F_hyd, -F_ext_unit) > 0`` — the hydrodynamic force points
  along the resisting direction.

The classical journal-bearing argument: when the journal is
displaced **toward** the loaded side (``eps`` aligned with
``+F_ext``), the convergent half builds high pressure on the
loaded side and the integrated force points back **opposite** to
the displacement, i.e. opposite to ``F_ext`` (positive cosine
against ``-F_ext_unit``). A negative or near-zero cosine is the
signature of a sign / non-dim convention regression in the Ausas
coupling.

Skipped when ``cupy`` / ``GPU_reynolds`` are not installed.
"""
from __future__ import annotations

import numpy as np
import pytest


cupy = pytest.importorskip("cupy")
solver_dynamic_gpu = pytest.importorskip(
    "reynolds_solver.cavitation.ausas.solver_dynamic_gpu")


from models.bearing_model import setup_grid  # noqa: E402
from models.diesel_ausas_adapter import (  # noqa: E402
    DieselAusasState,
    ausas_one_step_with_state,
    ausas_result_is_physical,
    set_ausas_backend_for_tests,
)
from models.diesel_transient import (  # noqa: E402
    build_H_2d, compute_hydro_forces, load_diesel,
)
from config import diesel_params as dparams  # noqa: E402
from config.oil_properties import MINERAL_OIL  # noqa: E402


_OMEGA_DIESEL = 2.0 * np.pi * 1900.0 / 60.0
_ETA_DIESEL = float(MINERAL_OIL["eta_diesel"])
_TOL = 1e-5
# The solver's own default is 5000 (Jacobi scheme is slow). A smoke
# test with a realistic eccentricity easily eats >2000 sweeps; use
# the solver default so the gate is on physics, not iteration budget.
_MAX_INNER = 5000


def test_ausas_force_opposes_external_load_at_modest_eps():
    """A modest eccentricity aligned with the external load should
    produce a hydrodynamic force pointing in the **opposite**
    direction (positive cosine against ``-F_ext_unit``)."""
    set_ausas_backend_for_tests(None)
    N_phi, N_z = 64, 16
    phi_1D, Z_1D, Phi, Z, d_phi, d_Z = setup_grid(N_phi, N_z)

    # Pick a crank angle where the diesel load is firmly downward.
    # ``load_diesel`` returns (Fx, Fy); we then set the eccentricity
    # along the SAME direction as F_ext so the bearing is loaded
    # along its convergent half. This is the setup used in every
    # textbook journal-bearing sanity check.
    phi_test = 360.0   # firing TDC ish — load is large here
    Fx_ext_arr, Fy_ext_arr = load_diesel(phi_test, F_max=255_000.0)
    Fx_ext = float(np.asarray(Fx_ext_arr).item())
    Fy_ext = float(np.asarray(Fy_ext_arr).item())
    F_ext_mag = float(np.hypot(Fx_ext, Fy_ext))
    assert F_ext_mag > 0.0
    # Unit external-load direction.
    ux = Fx_ext / F_ext_mag
    uy = Fy_ext / F_ext_mag
    # Modest eccentricity ALIGNED with the load.
    eps = 0.30
    eps_x = eps * ux
    eps_y = eps * uy

    H = build_H_2d(eps_x, eps_y, Phi, Z, dparams)
    state = DieselAusasState.from_initial_gap(H)

    # ``H_curr = H_prev`` zeroes the squeeze term — the result is
    # the steady JFO pressure for this gap.
    out = ausas_one_step_with_state(
        state, H_curr=H,
        dt_s=np.deg2rad(4.0) / _OMEGA_DIESEL,
        omega_shaft=_OMEGA_DIESEL,
        d_phi=d_phi, d_Z=d_Z,
        R=dparams.R, L=dparams.L,
        extra_options={"tol": _TOL, "max_inner": _MAX_INNER,
                        "alpha": 1.0},
        commit=True,
    )
    assert out.converged, (
        f"Ausas did not converge in {_MAX_INNER} inner iters at "
        f"eps={eps}; residual={out.residual}, n_inner={out.n_inner}")
    assert ausas_result_is_physical(
        out, tol=_TOL, max_inner=_MAX_INNER), (
        f"physical-contract failed: residual={out.residual}, "
        f"n_inner={out.n_inner}, theta=[{out.theta_min}, "
        f"{out.theta_max}]")

    p_scale = (6.0 * _ETA_DIESEL * _OMEGA_DIESEL
               * (dparams.R / dparams.c) ** 2)
    Fx_hyd, Fy_hyd = compute_hydro_forces(
        np.asarray(out.P_nd), p_scale, Phi,
        phi_1D, Z_1D, dparams.R, dparams.L)
    F_hyd_mag = float(np.hypot(Fx_hyd, Fy_hyd))

    # Sanity: the film produced a measurable, non-trivial response.
    # The threshold ``F_hyd_mag > 100 N`` corresponds to a 12 kPa
    # mean pressure on the projected bearing area (R·L = 8·10⁻³ m²)
    # — an extremely loose floor that fires only when ``P_nd ≈ 0``
    # everywhere (under-convergence or unit catastrophe).
    #
    # We deliberately do NOT compare against ``F_ext`` here: at the
    # diesel bearing's L/D = 0.4 with ε = 0.30, the integrated
    # dimensionless pressure force is F̂ ≈ 0.03, so the actual load
    # capacity is ``p_scale·R·L·F̂ ≈ 2 kN`` — much less than the
    # 200 kN peak external load. The classical "F_hyd > 5% F_ext"
    # rule-of-thumb only holds when ε is set to match the
    # operational point of the load, which is ε ≈ 0.85+ for diesel
    # firing peak — not a regime where this contract test should
    # operate (it would push Jacobi past max_inner). The real
    # magnitude parity is already enforced by
    # ``test_diesel_ausas_pressure_scale.py`` against the legacy
    # half-Sommerfeld baseline.
    assert F_hyd_mag > 100.0, (
        f"|F_hyd| = {F_hyd_mag:.3e} N — the film produced "
        "essentially no pressure response. Likely a "
        "near-zero P_nd / under-convergence regression.")

    # cos(F_hyd, -F_ext_unit) — projection of the hydrodynamic
    # force onto the resisting direction. THIS is the actual
    # sign-correctness gate: at a modest eccentricity aligned with
    # the load, the film pressure must point back along ``-F_ext``
    # (the journal is supported by the convergent half).
    cos_resist = (-(float(Fx_hyd) * ux + float(Fy_hyd) * uy)
                   / F_hyd_mag)
    assert cos_resist > 0.0, (
        f"F_hyd points along F_ext (cos against -F_ext_unit = "
        f"{cos_resist:.3g}); the film is NOT resisting the load. "
        "Likely a sign / coordinate-convention regression in the "
        "Ausas P_nd coupling.")
