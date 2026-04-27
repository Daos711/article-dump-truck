"""Test-env shims.

reynolds_solver не установлен в CI окружении контракт-тестов; стабим
минимум, нужный для импорта `models.bearing_model` и
`models.coexp_pairing`. Все coexp-тесты используют свои инъекции
texture_relief_fn / ps_solver, эти стабы только разблокируют импорт.
"""
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)


def _install_reynolds_stub():
    """Install no-op reynolds_solver only if it's not already importable."""
    try:
        import reynolds_solver  # noqa: F401
        return  # real one available — leave alone
    except ImportError:
        pass

    import numpy as np

    rs = types.ModuleType("reynolds_solver")

    def _solve_reynolds_stub(*a, **kw):
        # Returns shape compatible with bearing_model.solve_and_compute
        # (P, residual, n_iter, converged) — but tests don't call it.
        raise RuntimeError(
            "reynolds_solver stub: real solver required for this path")

    rs.solve_reynolds = _solve_reynolds_stub
    sys.modules["reynolds_solver"] = rs

    cav = types.ModuleType("reynolds_solver.cavitation")
    sys.modules["reynolds_solver.cavitation"] = cav

    ps = types.ModuleType("reynolds_solver.cavitation.payvar_salant")
    def _ps_stub(H, *a, **kw):
        H = np.asarray(H, dtype=float)
        P = np.zeros_like(H)
        theta = np.ones_like(H)
        return P, theta, None, None
    ps.solve_payvar_salant_gpu = _ps_stub
    ps.solve_payvar_salant_cpu = _ps_stub
    sys.modules["reynolds_solver.cavitation.payvar_salant"] = ps

    pv_mod = types.ModuleType(
        "reynolds_solver.piezoviscous.solver_pv_payvar_salant")
    pv_pkg = types.ModuleType("reynolds_solver.piezoviscous")
    pv_mod.solve_payvar_salant_piezoviscous = _ps_stub
    sys.modules["reynolds_solver.piezoviscous"] = pv_pkg
    sys.modules[
        "reynolds_solver.piezoviscous.solver_pv_payvar_salant"] = pv_mod

    utils = types.ModuleType("reynolds_solver.utils")
    def _relief_stub(H0, depth, Phi, Z, phi_c, Z_c, a_Z, a_phi, profile=None):
        return np.asarray(H0, dtype=float).copy()
    utils.create_H_with_ellipsoidal_depressions = _relief_stub
    sys.modules["reynolds_solver.utils"] = utils

    # squeeze helper (used by diesel_transient). Stub both the modern
    # nested path (reynolds_solver.dynamic.squeeze) and the historical
    # flat path (reynolds_solver.squeeze) so the import guard in
    # diesel_transient finds one of them in the sandbox.
    def _squeeze_to_api_params_stub(*a, **kw):
        return 0.0, 0.0, 2.0
    sq_flat = types.ModuleType("reynolds_solver.squeeze")
    sq_flat.squeeze_to_api_params = _squeeze_to_api_params_stub
    sys.modules["reynolds_solver.squeeze"] = sq_flat
    dyn_pkg = types.ModuleType("reynolds_solver.dynamic")
    sq_nested = types.ModuleType("reynolds_solver.dynamic.squeeze")
    sq_nested.squeeze_to_api_params = _squeeze_to_api_params_stub
    sys.modules["reynolds_solver.dynamic"] = dyn_pkg
    sys.modules["reynolds_solver.dynamic.squeeze"] = sq_nested

    # thermal (used by thermal_coupling). The pipeline-side
    # thermal_coupling module already wraps this in a try/except, so
    # leaving it un-stubbed makes SOLVER_THERMAL_AVAILABLE=False in
    # tests — which is what the existing test_diesel_thd_contract
    # tests already key on via importorskip.


_install_reynolds_stub()
