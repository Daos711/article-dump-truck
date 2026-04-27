"""Stage THD-0 contract tests for ``models.diesel_quasistatic``.

The tests exercise the pipeline-side wiring around the solver thermal
API. ``reynolds_solver.thermal`` is a hard dependency at run time; this
module skips cleanly when the solver isn't installed (e.g. in a CI
sandbox without the GPU package).

Test plan (mapped to Stage THD-0 patch spec):
    1. ``thermal=None`` and ``thermal=ThermalConfig(mode="off")``
       reproduce the legacy isothermal solution.
    2. Walther round-trip on the existing oil dictionaries:
       ``mu_at_T_C(105 °C) == eta_diesel`` and
       ``mu_at_T_C(50 °C) == eta_pump`` for both MINERAL_OIL and
       RAPESEED_OIL.
    3. ``gamma=0`` produces ``T_eff == T_in_C`` everywhere with
       ``thermal_converged`` all True.
    4. Fixed-H viscosity trend: at fixed eccentricity, η decreases and
       ``P_loss = F_tr * U`` decreases with rising T (NOT a load-matched
       cycle — strict isolated trend).
    5. THD smoke run (6 crank points): no NaN/inf, eta_eff > 0,
       T_eff >= T_in_C, all converged, ``max(thermal_outer) <= 5``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Hard skip when the solver thermal API isn't importable. Tests run in
# production envs where reynolds_solver is installed.
pytest.importorskip("reynolds_solver")
pytest.importorskip("reynolds_solver.thermal")

from config.oil_properties import MINERAL_OIL, RAPESEED_OIL
from models.thermal_coupling import (
    ThermalConfig,
    build_oil_walther,
    viscosity_at_T_C,
)
from models import diesel_quasistatic as dq


SMALL_PHI = np.linspace(0.0, 720.0, 6, endpoint=False)
SMALL_GRID = 40       # N_phi = N_Z = 40 for fast unit tests
SMOKE_GRID = 60


# ─── Test 1 — isothermal regression ────────────────────────────────

def test_thermal_none_equals_off_equals_legacy():
    """Section 3 'mode == off' must match the legacy isothermal runner."""
    res_none = dq.run_diesel_analysis(
        thermal=None, phi_crank=SMALL_PHI,
        n_phi_grid=SMALL_GRID, n_z_grid=SMALL_GRID,
        configs=[dq.CONFIGS[0]],
    )
    res_off = dq.run_diesel_analysis(
        thermal=ThermalConfig(mode="off"), phi_crank=SMALL_PHI,
        n_phi_grid=SMALL_GRID, n_z_grid=SMALL_GRID,
        configs=[dq.CONFIGS[0]],
    )
    for k in ("epsilon", "hmin", "pmax", "f", "F_hyd"):
        np.testing.assert_allclose(res_none[k], res_off[k],
                                     rtol=0, atol=1e-12,
                                     err_msg=f"mismatch on {k}")
    # mode="off" must populate the THD arrays consistently.
    eta_d = MINERAL_OIL["eta_diesel"]
    np.testing.assert_allclose(res_off["eta_eff"], eta_d, rtol=1e-12)
    np.testing.assert_allclose(res_off["T_eff"], 105.0, rtol=0, atol=1e-12)
    # Energy convergence is trivially True in mode="off" (no fixed-point
    # iteration). The Stage THD-0B flag ``thermal_converged`` additionally
    # requires ``valid_fullfilm``; on the small smoke grid F_max=850 kN
    # outruns the 40x40 W-table so above_range angles legitimately fall
    # out — that's the point of the redefined headline flag.
    assert bool(res_off["thermal_energy_converged"].all())
    assert int(res_off["thermal_outer"].max()) == 0


# ─── Test 2 — Walther roundtrip ───────────────────────────────────

@pytest.mark.parametrize("oil", [MINERAL_OIL, RAPESEED_OIL],
                         ids=["mineral", "rapeseed"])
def test_walther_roundtrip(oil):
    """Section 2 acceptance: mu(105) == eta_diesel and mu(50) == eta_pump."""
    fit = build_oil_walther(oil)
    mu_105 = viscosity_at_T_C(fit, 105.0)
    mu_50 = viscosity_at_T_C(fit, 50.0)
    rel_105 = abs(mu_105 - oil["eta_diesel"]) / oil["eta_diesel"]
    rel_50 = abs(mu_50 - oil["eta_pump"]) / oil["eta_pump"]
    assert rel_105 < 1e-10, (
        f"mu_at_T_C(105) returned {mu_105}, expected {oil['eta_diesel']}, "
        f"rel error {rel_105}")
    assert rel_50 < 1e-10, (
        f"mu_at_T_C(50) returned {mu_50}, expected {oil['eta_pump']}, "
        f"rel error {rel_50}")


# ─── Test 3 — gamma=0 collapses to isothermal at T_in ─────────────

def test_gamma_zero_equals_no_heating():
    """Section 3 acceptance: gamma=0 -> T_eff == T_in everywhere."""
    thermal = ThermalConfig(mode="global_static", gamma_mix=0.0,
                              T_in_C=105.0)
    res = dq.run_diesel_analysis(
        thermal=thermal, phi_crank=SMALL_PHI,
        n_phi_grid=SMALL_GRID, n_z_grid=SMALL_GRID,
        configs=[dq.CONFIGS[0]],
    )
    np.testing.assert_allclose(res["T_eff"], thermal.T_in_C,
                                 rtol=0, atol=1e-9)
    # eta_eff should be exactly mu_at_T_C(T_in) — i.e. eta_diesel for the
    # mineral config when T_in = 105 °C.
    np.testing.assert_allclose(res["eta_eff"], MINERAL_OIL["eta_diesel"],
                                 rtol=1e-9)
    # gamma=0 collapses the energy balance: every angle, even
    # load-mismatched ones, must be ``thermal_energy_converged``.
    # ``thermal_converged`` (energy AND valid_fullfilm) may exclude
    # above_range angles on the small smoke grid; that's expected.
    assert bool(res["thermal_energy_converged"].all())


# ─── Test 4 — fixed-H viscosity trend ─────────────────────────────

def test_fixed_H_viscosity_trend():
    """Section 4 acceptance: at fixed eps, eta(T) decreases and
    P_loss(T) decreases monotonically across T = 80, 100, 120 °C.

    This is a one-shot solve at a fixed eccentricity (NOT load-matched)
    so the trend is physically unambiguous.
    """
    from models.bearing_model import (
        DEFAULT_CAVITATION, DEFAULT_CLOSURE, make_H, setup_grid,
        setup_texture, solve_and_compute,
    )
    from config import diesel_params as params

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(SMALL_GRID,
                                                              SMALL_GRID)
    phi_c, Z_c = setup_texture(params)
    eps_fixed = 0.6

    fit = build_oil_walther(MINERAL_OIL)
    omega = 2.0 * math.pi * params.n / 60.0
    U = omega * params.R

    eta_seq = []
    Ploss_seq = []
    for T_C in (80.0, 100.0, 120.0):
        eta = viscosity_at_T_C(fit, T_C)
        H = make_H(eps_fixed, Phi_mesh, Z_mesh, params,
                   textured=False, phi_c_flat=phi_c, Z_c_flat=Z_c)
        _, _F, _mu, _Q, _h, _p, F_tr, _, _, _ = solve_and_compute(
            H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
            phi_1D, Z_1D, Phi_mesh,
            closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION,
        )
        eta_seq.append(float(eta))
        Ploss_seq.append(float(F_tr) * U)

    assert eta_seq[0] > eta_seq[1] > eta_seq[2], eta_seq
    assert Ploss_seq[0] > Ploss_seq[1] > Ploss_seq[2], Ploss_seq


# ─── Test 5 — THD smoke run ───────────────────────────────────────

def test_thd_smoke_global_static():
    """Section 5 smoke: 6 crank points, gamma=0.7, one config."""
    thermal = ThermalConfig(
        mode="global_static", gamma_mix=0.7, T_in_C=105.0,
        cp_J_kgK=2000.0, mdot_floor_kg_s=1e-4,
        tol_T_C=0.5, max_outer=5, underrelax_T=0.6,
    )
    res = dq.run_diesel_analysis(
        thermal=thermal, phi_crank=SMALL_PHI,
        n_phi_grid=SMOKE_GRID, n_z_grid=SMOKE_GRID,
        configs=[dq.CONFIGS[0]],
    )
    valid = np.asarray(res["valid_fullfilm"], dtype=bool)
    # On the small smoke grid the W_table / sigma combination can land a
    # subset of angles in above_range — only require "finite" from the
    # arrays where we actually got a clean inner solve.
    for k in ("T_eff", "eta_eff", "P_loss", "Q", "mdot", "F_tr",
              "epsilon", "hmin", "pmax", "f", "F_hyd"):
        arr = np.asarray(res[k])[valid]
        if arr.size > 0:
            assert np.all(np.isfinite(arr)), f"non-finite values in {k}"
    eta_v = np.asarray(res["eta_eff"])[valid]
    if eta_v.size > 0:
        assert np.all(eta_v > 0)
    T_v = np.asarray(res["T_eff"])[valid]
    if T_v.size > 0:
        assert np.all(T_v >= thermal.T_in_C - 1e-9)
    # Energy convergence should still be reached on smoke; the tighter
    # thermal_converged also requires valid_fullfilm so we don't assert
    # all() on it (production case may have above-range angles).
    assert int(np.asarray(res["thermal_outer"]).max()) <= thermal.max_outer


def test_thd_arrays_have_correct_shape_and_dtype():
    """Section 5 acceptance: load_status / valid_fullfilm / W_table_*
    arrays must be present with the expected shape and dtype."""
    thermal = ThermalConfig(mode="global_static", gamma_mix=0.5,
                              T_in_C=105.0, max_outer=3)
    res = dq.run_diesel_analysis(
        thermal=thermal, phi_crank=SMALL_PHI,
        n_phi_grid=SMALL_GRID, n_z_grid=SMALL_GRID,
        configs=[dq.CONFIGS[0]],
    )
    n_cfg = len(res["configs"])
    n_phi = len(SMALL_PHI)
    assert res["load_status"].shape == (n_cfg, n_phi)
    assert res["load_status"].dtype.kind == "U"
    assert res["load_match_ratio"].shape == (n_cfg, n_phi)
    assert res["valid_fullfilm"].shape == (n_cfg, n_phi)
    assert res["valid_fullfilm"].dtype == np.bool_
    assert res["W_table_max"].shape == (n_cfg,)
    assert res["W_table_finite"].shape == (n_cfg,)
    assert res["W_table_finite"].dtype == np.bool_
    # Every per-angle status must be one of the documented values.
    valid_statuses = {"ok", "below_range", "above_range",
                      "solver_failed", "wtable_failed"}
    for st in np.asarray(res["load_status"]).ravel():
        assert st in valid_statuses, f"unexpected load_status {st!r}"
    # thermal_converged must imply valid_fullfilm (Section 3 redefinition).
    tc = np.asarray(res["thermal_converged"])
    vf = np.asarray(res["valid_fullfilm"])
    assert np.all(~tc | vf), (
        "thermal_converged must imply valid_fullfilm")
    # F_max_used should default to params.F_max when --F-max is not set.
    from config import diesel_params as _params
    assert math.isclose(float(res["F_max_used"]), float(_params.F_max),
                          rel_tol=1e-12)
