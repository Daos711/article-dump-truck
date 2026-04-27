"""Stage J Bug 4 follow-up — force-direction sanity at runner level.

After Stage J adds the ``force_balance_projection`` diagnostic to
the result dict, this test pins the contract: the median of the
projection over the first ~120° of crank angle must be **positive**
on a smooth-bearing low-load smoke run. A consistently negative or
near-zero projection is the signature of a sign / coordinate
regression that pushes the journal into the wall instead of
resisting the external load.

The full smoke runs ``run_transient`` with ``cavitation=
ausas_dynamic`` and a tiny grid; it skips cleanly when the GPU
package is absent.
"""
from __future__ import annotations

import numpy as np
import pytest


cupy = pytest.importorskip("cupy")
solver_dynamic_gpu = pytest.importorskip(
    "reynolds_solver.cavitation.ausas.solver_dynamic_gpu")


from models.diesel_ausas_adapter import (  # noqa: E402
    set_ausas_backend_for_tests,
)
from models.diesel_transient import (  # noqa: E402
    CONFIGS, EnvelopeAbortConfig, run_transient,
)
from models.thermal_coupling import ThermalConfig  # noqa: E402


def test_force_balance_projection_positive_on_smooth_smoke():
    """Smooth + mineral, F = 0.3·F_max, low fidelity. The
    runner's per-step ``force_balance_projection`` array is
    ``F_hyd · (-F_ext / |F_ext|)`` — physically the projection of
    the hydrodynamic force onto the resisting direction. Over the
    first 120° (compression stroke pre-firing) on a healthy
    backend the median MUST be positive: the film resists the
    external load."""
    # Make sure no leftover stub interferes with the real backend.
    set_ausas_backend_for_tests(None)
    thermal = ThermalConfig(
        mode="off", T_in_C=105.0, gamma_mix=0.7,
        cp_J_kgK=2000.0, mdot_floor_kg_s=1e-4, tau_th_s=0.5,
    )
    res = run_transient(
        F_max=0.3 * 850_000.0,
        configs=[CONFIGS[0]],   # smooth + mineral
        thermal=thermal,
        cavitation="ausas_dynamic",
        texture_kind="none",
        n_grid=32,
        n_cycles=1,
        d_phi_base_deg=4.0,
        d_phi_peak_deg=1.0,
        envelope_abort=EnvelopeAbortConfig.disabled(),
        ausas_options={"tol": 1e-3, "max_inner": 500, "alpha": 1.0},
    )
    assert "force_balance_projection" in res
    proj = np.asarray(res["force_balance_projection"][0])
    phi = np.asarray(res["phi_crank_deg"])
    valid = np.asarray(res["valid_dynamic"][0], dtype=bool)
    early = (phi < 120.0) & valid & np.isfinite(proj)
    finite_proj = proj[early]
    assert finite_proj.size > 5, (
        f"Only {finite_proj.size} early-phi valid steps survive — "
        "the run aborted before enough samples accumulated.")
    median = float(np.median(finite_proj))
    assert median > 0.0, (
        f"Force-balance projection median over first 120° is "
        f"{median:.3e} (≤ 0). The hydrodynamic force is not "
        "resisting the external load — likely a sign / non-dim "
        "regression in the Ausas pressure coupling.")
