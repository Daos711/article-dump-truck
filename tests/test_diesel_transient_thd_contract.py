"""Stage Diesel Transient THD-0 contract tests.

Tests fall into two groups:

* **Pipeline-side, no solver**: contract / parsing / paired-mask /
  retry plumbing tests that exercise the helpers in
  ``models.diesel_transient`` directly. These run unconditionally.

* **Solver-dependent**: full ``run_transient`` smokes that need the
  ``reynolds_solver`` package to actually solve a couple of mechanical
  steps. Skipped when the solver is unavailable in the sandbox.
"""
from __future__ import annotations

import math
import warnings as _warnings

import numpy as np
import pytest

# Pipeline-side helpers (always importable, no solver needed at import).
from models.thermal_coupling import (
    ThermalConfig,
    global_relax_step,
)
from models.diesel_transient import (
    CONFIGS, CONFIG_KEYS, _compute_paired_transient,
    _solve_dynamic_with_retry,
)
from models.diesel_quasistatic import SolverRetryConfig

SOLVER_AVAILABLE = True
try:
    import reynolds_solver  # noqa: F401
    from reynolds_solver.thermal import mu_at_T_C  # noqa: F401
    from models.diesel_transient import run_transient  # noqa: F401
except ImportError:
    SOLVER_AVAILABLE = False


# ─── 1. Paired comparison (synthetic) ─────────────────────────────

def _mk_cfg(*, label, oil_name, textured):
    return dict(label=label, textured=textured,
                color="k", ls="-",
                oil={"name": oil_name, "eta_diesel": 0.01,
                     "eta_pump": 0.022, "rho": 875.0})


def test_paired_transient_uses_common_no_clamp_only():
    """Common_valid_no_clamp = smooth_valid_no_clamp & textured_valid_no_clamp.

    Stats must be computed on this intersection only, never on per-side
    masks (Section 10 of the patch spec).
    """
    cfgs = [
        _mk_cfg(label="s_min", oil_name="mineral", textured=False),
        _mk_cfg(label="t_min", oil_name="mineral", textured=True),
    ]
    n_phi = 8
    last_start = 0
    valid_dyn = np.array([
        [True, True, True, True, True, True, False, False],
        [True, True, True, True, False, False, True, True],
    ])
    # Last two angles of textured are clamped → excluded from no_clamp.
    valid_no_clamp = np.array([
        [True, True, True, True, True, True, False, False],
        [True, True, True, True, False, False, False, False],
    ])
    T_eff_used = np.array([
        [110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 999, 999],
        [120.0, 121.0, 122.0, 123.0, 999, 999, 888, 888],
    ])
    z = np.zeros((2, n_phi))
    paired = _compute_paired_transient(
        cfgs, last_start, valid_dyn, valid_no_clamp,
        T_eff_used, z, z, z, z, z)
    assert len(paired) == 1
    rec = paired[0]
    # Common no_clamp: angles 0..3 (4 steps).
    assert rec["common_no_clamp_count"] == 4
    # Mean ΔT_eff on those 4 steps should be uniformly +10.
    assert rec["mean_dT_eff"] == pytest.approx(10.0)


def test_paired_transient_zero_overlap_clean_nan():
    cfgs = [
        _mk_cfg(label="s", oil_name="mineral", textured=False),
        _mk_cfg(label="t", oil_name="mineral", textured=True),
    ]
    n = 4
    valid_dyn = np.array([[True, False, True, False],
                           [False, True, False, True]])
    valid_noc = valid_dyn.copy()
    z = np.zeros((2, n))
    paired = _compute_paired_transient(
        cfgs, 0, valid_dyn, valid_noc, z, z, z, z, z, z)
    rec = paired[0]
    assert rec["common_no_clamp_count"] == 0
    assert math.isnan(rec["mean_dT_eff"])


def test_paired_transient_skips_pairs_without_both_sides():
    cfgs = [
        _mk_cfg(label="s_min", oil_name="mineral", textured=False),
        _mk_cfg(label="s_rape", oil_name="rapeseed", textured=False),
    ]
    valid = np.zeros((2, 4), dtype=bool)
    z = np.zeros((2, 4))
    paired = _compute_paired_transient(
        cfgs, 0, valid, valid, z, z, z, z, z, z)
    assert paired == []


# ─── 2. global_relax_step monotonicity ────────────────────────────

def test_global_relax_slower_than_global_static():
    """For dt < tau_th, relax produces smaller |T - T_prev| than the
    static target (which would jump straight to T_target)."""
    T_prev = 100.0
    T_target = 140.0
    T_static = T_target
    T_relax = global_relax_step(
        T_prev_C=T_prev, T_target_C=T_target,
        dt_s=0.1, tau_th_s=0.5)
    # static jumps a full +40; relax with dt=0.1, tau=0.5 jumps
    # (1 - exp(-0.2))*40 ≈ 7.25.
    assert abs(T_relax - T_prev) < abs(T_static - T_prev)
    # Sanity bounds.
    assert T_prev < T_relax < T_target


def test_global_relax_zero_tau_collapses_to_target():
    out = global_relax_step(
        T_prev_C=100.0, T_target_C=140.0,
        dt_s=0.0, tau_th_s=0.5)
    assert out == pytest.approx(140.0)


def test_global_relax_huge_dt_reaches_target():
    out = global_relax_step(
        T_prev_C=100.0, T_target_C=140.0,
        dt_s=10.0, tau_th_s=0.001)
    assert out == pytest.approx(140.0, abs=1e-9)


# ─── 3. Solver retry contract via _solve_dynamic_with_retry ────────

def _fake_solve_reynolds_factory(*, recovery_omega=None,
                                    always_fail=False):
    """Synthetic substitute for ``reynolds_solver.solve_reynolds``.

    Mirrors the shape: returns ``(P, residual, n_iter, converged)``.
    Emits the SOR-non-convergence warning + ``converged=False`` when
    ``always_fail=True`` or when omega is not the recovery value.
    """
    log = []

    def fake(H, d_phi, d_Z, R, L, **kw):
        omega = kw.get("omega")
        max_iter = kw.get("max_iter")
        p_init = kw.get("P_init")
        log.append(dict(p_init_is_none=(p_init is None),
                          omega=omega, max_iter=max_iter))
        succeed = False
        if always_fail:
            succeed = False
        elif recovery_omega is not None:
            succeed = (omega is not None and
                        abs(float(omega) - float(recovery_omega))
                        < 1e-9)
        else:
            succeed = True
        if not succeed:
            _warnings.warn(
                "SOR не сошёлся: delta=2.0e-04, n_iter=50000",
                UserWarning)
            return (np.full(H.shape, np.nan), 1.0, 50000, False)
        return (np.full(H.shape, 1.0), 1e-6, 100, True)

    return fake, log


def test_retry_passes_omega_and_max_iter_to_solver():
    """Stage Texture Stability 2 contract reused by transient: retry
    must call solve_reynolds with closure_kw merged into the kw dict
    (i.e. pass ``omega=...`` and ``max_iter=...`` directly)."""
    import models.diesel_transient as dt
    fake, log = _fake_solve_reynolds_factory(recovery_omega=1.70)
    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake
        H = np.ones((4, 4))
        Phi = np.zeros((4, 4))
        phi = np.linspace(0, 1, 4)
        Z = np.linspace(-1, 1, 4)
        out = _solve_dynamic_with_retry(
            H, 0.1, 0.1, 0.05, 0.04,
            base_kw=dict(closure="laminar", cavitation="half_sommerfeld",
                          tol=1e-5, max_iter=50000,
                          P_init=np.full((4, 4), 5e5),
                          xprime=0.0, yprime=0.0, beta=2.0),
            p_scale=1.0,
            Phi_mesh=Phi, phi_1D=phi, Z_1D=Z,
            retry_config=SolverRetryConfig(
                omega_values=(1.70, 1.55),
                max_iter_retry=99_000,
                cold_start=True,
            ),
            textured=True,
        )
    finally:
        dt.solve_reynolds = target
    # Primary attempt sees no omega; first retry sees omega=1.70.
    assert log[0]["omega"] is None
    assert log[1]["omega"] == pytest.approx(1.70)
    assert log[1]["max_iter"] == 99_000
    assert log[1]["p_init_is_none"] is True   # cold_start
    P_, Fx_, Fy_, _, ok_, reason_ = out
    assert ok_ is True
    assert reason_ == "ok_retry_omega_1p700"


def test_retry_disabled_for_smooth_textured_only_default():
    """Default policy is textured_only=True; smooth configs must NOT
    trigger any retry even on failure."""
    import models.diesel_transient as dt
    fake, log = _fake_solve_reynolds_factory(always_fail=True)
    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake
        H = np.ones((4, 4))
        Phi = np.zeros((4, 4))
        phi = np.linspace(0, 1, 4)
        Z = np.linspace(-1, 1, 4)
        out = _solve_dynamic_with_retry(
            H, 0.1, 0.1, 0.05, 0.04,
            base_kw=dict(closure="laminar", cavitation="half_sommerfeld",
                          tol=1e-5, max_iter=50000,
                          P_init=np.full((4, 4), 5e5),
                          xprime=0.0, yprime=0.0, beta=2.0),
            p_scale=1.0,
            Phi_mesh=Phi, phi_1D=phi, Z_1D=Z,
            retry_config=SolverRetryConfig(),  # textured_only=True
            textured=False,
        )
    finally:
        dt.solve_reynolds = target
    # Single primary call only.
    assert len(log) == 1
    assert out[4] is False
    assert out[5] == "SOR_did_not_converge"


def test_retry_failed_pressure_never_propagated():
    """Failed primary returns NaN P; retry must call solve with
    P_init=None when cold_start=True, never with the poisoned NaN."""
    import models.diesel_transient as dt
    fake, log = _fake_solve_reynolds_factory(recovery_omega=1.55)
    target = dt.solve_reynolds
    try:
        dt.solve_reynolds = fake
        H = np.ones((4, 4))
        Phi = np.zeros((4, 4))
        phi = np.linspace(0, 1, 4)
        Z = np.linspace(-1, 1, 4)
        warm = np.full((4, 4), 5e5)
        _solve_dynamic_with_retry(
            H, 0.1, 0.1, 0.05, 0.04,
            base_kw=dict(closure="laminar", cavitation="half_sommerfeld",
                          tol=1e-5, max_iter=50000,
                          P_init=warm,
                          xprime=0.0, yprime=0.0, beta=2.0),
            p_scale=1.0,
            Phi_mesh=Phi, phi_1D=phi, Z_1D=Z,
            retry_config=SolverRetryConfig(
                omega_values=(1.70, 1.55), cold_start=True),
            textured=True,
        )
    finally:
        dt.solve_reynolds = target
    # 3 calls: primary (warm), retry omega=1.70 (cold), retry omega=1.55 (cold).
    assert len(log) == 3
    assert log[0]["p_init_is_none"] is False
    assert log[1]["p_init_is_none"] is True
    assert log[2]["p_init_is_none"] is True


# ─── 4. No W_table / find_epsilon_for_load in transient ────────────

def test_transient_does_not_use_quasistatic_load_matcher():
    """Section 16 red line: transient THD must not import or use the
    quasistatic load-matching machinery."""
    import models.diesel_transient as dt
    # Force utf-8 — Windows defaults to cp1251 which can't decode the
    # cyrillic comments in this module.
    with open(dt.__file__, encoding="utf-8") as f:
        src = f.read()
    assert "build_load_table" not in src
    assert "find_epsilon_for_load" not in src
    assert "_eps_max_hydro" not in src


# ─── 5. SolverRetryConfig defaults ────────────────────────────────

def test_solver_retry_default_textured_only():
    cfg = SolverRetryConfig()
    assert cfg.enabled is True
    assert cfg.textured_only is True
    assert cfg.cold_start is True
    assert tuple(cfg.omega_values) == (1.70, 1.55)
    assert cfg.applicable(textured=True) is True
    assert cfg.applicable(textured=False) is False


# ─── Solver-dependent smokes ──────────────────────────────────────

@pytest.mark.skipif(not SOLVER_AVAILABLE,
                     reason="reynolds_solver not installed in this env")
def test_legacy_off_regression_thermal_none_equals_off():
    """Test 1 in the patch spec: ``thermal=None`` must reproduce
    ``ThermalConfig(mode='off')`` bit-for-bit on a tiny smoke run."""
    smoke_kwargs = dict(
        F_max=200_000.0,
        configs=[CONFIGS[0]],
        n_grid=40,
        n_cycles=1,
        d_phi_base_deg=30.0,
        d_phi_peak_deg=10.0,
    )
    res_none = run_transient(thermal=None, **smoke_kwargs)
    res_off = run_transient(thermal=ThermalConfig(mode="off"),
                              **smoke_kwargs)
    for k in ("eps_x", "eps_y", "hmin", "pmax", "F_tr", "N_loss"):
        a = np.asarray(res_none[k])
        b = np.asarray(res_off[k])
        # Both arrays may carry NaNs on failed steps; allow_equal_nan.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12,
                                     equal_nan=True,
                                     err_msg=f"mismatch on {k}")


@pytest.mark.skipif(not SOLVER_AVAILABLE,
                     reason="reynolds_solver not installed in this env")
def test_gamma_zero_collapse_to_T_in():
    """Test 2: gamma=0 + global_relax → T_eff == T_in everywhere
    (no heating)."""
    thermal = ThermalConfig(
        mode="global_relax", gamma_mix=0.0, T_in_C=105.0,
        tau_th_s=0.5)
    res = run_transient(
        F_max=200_000.0,
        configs=[CONFIGS[0]],
        thermal=thermal,
        n_grid=40, n_cycles=1,
        d_phi_base_deg=30.0, d_phi_peak_deg=10.0,
    )
    # Solver-success steps must have T_eff == T_in.
    valid = np.asarray(res["solver_success"], dtype=bool)
    T_eff = np.asarray(res["T_eff"])[valid]
    if T_eff.size > 0:
        np.testing.assert_allclose(T_eff, 105.0, atol=1e-6)


@pytest.mark.skipif(not SOLVER_AVAILABLE,
                     reason="reynolds_solver not installed in this env")
def test_thd_smoke_runs_clean_global_relax():
    """Test 5 (arrays shape) + Test 4 (Q/mdot finite on valid steps)."""
    thermal = ThermalConfig(
        mode="global_relax", gamma_mix=0.7, T_in_C=105.0,
        tau_th_s=0.5)
    res = run_transient(
        F_max=200_000.0,
        configs=[CONFIGS[0]],
        thermal=thermal,
        n_grid=40, n_cycles=1,
        d_phi_base_deg=30.0, d_phi_peak_deg=10.0,
    )
    n_steps = int(res["phi_crank_deg"].size)
    # Per-spec shapes (n_cfg, n_steps).
    for k in ("T_eff", "T_target", "eta_eff", "P_loss", "Q", "mdot",
              "valid_dynamic", "valid_no_clamp", "contact_clamp",
              "solver_success", "retry_used"):
        arr = np.asarray(res[k])
        assert arr.shape == (1, n_steps), (k, arr.shape)
    # On valid_dynamic steps, Q/mdot must be finite, eta>0, mdot>=floor.
    vd = np.asarray(res["valid_dynamic"], dtype=bool)
    Q_v = np.asarray(res["Q"])[vd]
    mdot_v = np.asarray(res["mdot"])[vd]
    eta_v = np.asarray(res["eta_eff"])[vd]
    if Q_v.size > 0:
        assert np.all(np.isfinite(Q_v))
        assert np.all(np.isfinite(mdot_v))
        assert np.all(eta_v > 0)
        assert np.all(mdot_v >= thermal.mdot_floor_kg_s - 1e-30)
