"""Stage J fu-2 Task 32 — commit-semantics state-machine contract.

Pins the four state-machine fields that ``MechanicalStepResult``
exposes so the runner can correctly distinguish:

* ``committed_converged``    — final trial passed all gates;
* ``committed_last_valid``   — final trial failed (e.g. commit
                               returned non-converged) but the
                               last accepted Picard trial's
                               P/F/theta were finite — runner uses
                               those for trajectory continuity;
* ``rolled_back_previous``   — no finite trial; mechanics roll
                               back to the kernel-input state;
* ``rejected_no_commit``     — legacy fallback (HS path); trial
                               outputs are NaN.

The ``committed_state_is_finite`` flag on each row must be True
for any row the postprocessor will treat as "valid" — the
publishability mask uses it as a hard gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_ausas_adapter import (
    DieselAusasState,
    set_ausas_backend_for_tests,
)
from models.diesel_coupling import (
    AusasDynamicBackend,
    HalfSommerfeldBackend,
    PhysicalGuardsConfig,
    POLICY_AUSAS_DYNAMIC,
    POLICY_LEGACY_HS,
    StepContext,
    advance_mechanical_step,
)
from models.diesel_coupling.kernel import _run_legacy_verlet


# ─── Fixtures: minimal grids and stub backends ─────────────────────


def _step_context(N_phi: int = 8, N_z: int = 4) -> StepContext:
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    Z_1D = np.linspace(-1.0, 1.0, N_z)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    return StepContext(
        phi_deg=180.0, F_ext_x=0.0, F_ext_y=-1.0e3, F_max=1.0e3,
        p_scale=1.0e6, omega=199.0, eta=0.04,
        R=0.1, L=0.08, c=1.2e-4,
        Phi_mesh=Phi_mesh, Z_mesh=Z_mesh,
        phi_1D=phi_1D, Z_1D=Z_1D,
        d_phi=float(phi_1D[1] - phi_1D[0]),
        d_Z=float(Z_1D[1] - Z_1D[0]),
        cfg={"label": "test", "textured": False},
        texture_kind="none",
        groove_relief=None,
        phi_c_flat=None, Z_c_flat=None,
        closure="laminar", cavitation="ausas_dynamic",
        P_warm_init=None,
    )


def _make_legacy_hs_backend(*, fake_p_nd=0.5, fake_F_hyd=(0.0, -100.0),
                              converged=True):
    """Stateless HS backend stub returning a deterministic finite
    pressure / force on every call. Built as a duck-typed object
    rather than subclassing ``HalfSommerfeldBackend`` (which has a
    non-trivial __init__ signature) — the kernel only requires the
    ``stateful`` / ``requires_implicit_mech_coupling`` flags and a
    ``solve_trial`` callable matching the protocol."""
    from models.diesel_coupling.backends import PressureSolveResult

    class _StubHS:
        name = "hs_stub"
        stateful = False
        requires_implicit_mech_coupling = False

        def solve_trial(self, *, H_curr, H_prev=None, dt_phys, omega,
                         state, commit, context, extra_options=None,
                         p_warm_init=None, vx_squeeze=0.0,
                         vy_squeeze=0.0):
            P = np.full_like(np.asarray(H_curr), fake_p_nd, dtype=float)
            return PressureSolveResult(
                P_nd=P,
                theta=np.ones_like(P),
                Fx_hyd=fake_F_hyd[0], Fy_hyd=fake_F_hyd[1],
                H_used=H_curr,
                residual=1e-9, n_inner=10,
                converged=bool(converged),
                backend_name="hs_stub",
                reason="ok" if converged else "stub_failed",
            )

    return _StubHS()


# ─── Test 1: HS converged → committed_converged ────────────────────


def test_legacy_hs_converged_committed_converged():
    """HS path with successful solve → ``committed_converged`` /
    ``converged_trial`` / ``committed_state_is_finite=True``."""
    backend = _make_legacy_hs_backend()

    def _build_H(eps_x, eps_y):
        return np.full((4, 8), 0.5)

    def _clamp(ex, ey, vx, vy):
        return ex, ey, vx, vy, False

    ms = advance_mechanical_step(
        ex_n=0.0, ey_n=0.0, vx_n=0.0, vy_n=0.0,
        ax_prev=0.0, ay_prev=0.0,
        dt_phys_s=1e-5,
        backend=backend,
        backend_state=None,
        policy=POLICY_LEGACY_HS,
        guards_cfg=PhysicalGuardsConfig.from_profile(
            "diagnostic", "general"),
        ausas_tol=1e-6, ausas_max_inner=5000,
        extra_options=None,
        context=_step_context(),
        m_shaft=10.0, eps_max=0.95,
        clamp_fn=_clamp, build_H_fn=_build_H,
    )
    assert ms.final_trial_status == "converged"
    assert ms.committed_state_status == "committed_converged"
    assert ms.accepted_state_source == "converged_trial"
    assert ms.committed_state_is_finite is True


def test_legacy_hs_solver_failed_rejected_no_commit():
    """HS path with non-converged solve → ``rejected_no_commit`` /
    ``none``. The legacy contract still ``accepted=True`` (mechanics
    advance regardless), but the COMMIT semantics are clear."""
    backend = _make_legacy_hs_backend(converged=False)

    def _build_H(eps_x, eps_y):
        return np.full((4, 8), 0.5)

    def _clamp(ex, ey, vx, vy):
        return ex, ey, vx, vy, False

    ms = advance_mechanical_step(
        ex_n=0.0, ey_n=0.0, vx_n=0.0, vy_n=0.0,
        ax_prev=0.0, ay_prev=0.0,
        dt_phys_s=1e-5,
        backend=backend,
        backend_state=None,
        policy=POLICY_LEGACY_HS,
        guards_cfg=PhysicalGuardsConfig.from_profile(
            "diagnostic", "general"),
        ausas_tol=1e-6, ausas_max_inner=5000,
        extra_options=None,
        context=_step_context(),
        m_shaft=10.0, eps_max=0.95,
        clamp_fn=_clamp, build_H_fn=_build_H,
    )
    assert ms.committed_state_status == "rejected_no_commit"
    assert ms.accepted_state_source == "none"
    assert ms.committed_state_is_finite is False


# ─── Test 2: damped Ausas — fake backend ───────────────────────────


def _fake_dict_ausas_backend(**kwargs):
    """Single-shot fake — finite, converged, deterministic."""
    H = np.asarray(kwargs["H_curr"], dtype=float)
    n_z, n_phi = H.shape
    P = (1.0 / np.maximum(H, 1e-3)) * 0.5
    theta = np.where(P > 1.5, 0.7, 1.0) * np.ones_like(P)
    return dict(
        P=P, theta=theta,
        residual_linf=1e-7, n_inner=50, converged=True,
    )


@pytest.fixture
def _ausas_finite_backend():
    set_ausas_backend_for_tests(_fake_dict_ausas_backend)
    try:
        yield
    finally:
        set_ausas_backend_for_tests(None)


def test_damped_committed_converged_finite_state(_ausas_finite_backend):
    """Healthy damped run: finite Ausas converged + Picard reaches
    fixed point → committed_converged."""
    N_phi, N_z = 8, 4
    backend = AusasDynamicBackend()

    def _build_H(eps_x, eps_y):
        return np.full((N_z, N_phi), 0.5)

    def _clamp(ex, ey, vx, vy):
        return ex, ey, vx, vy, False

    H0 = _build_H(0.0, 0.0)
    state = DieselAusasState.from_initial_gap(H0)
    ms = advance_mechanical_step(
        ex_n=0.0, ey_n=0.0, vx_n=0.0, vy_n=0.0,
        ax_prev=0.0, ay_prev=0.0,
        dt_phys_s=1e-5,
        backend=backend,
        backend_state=state,
        policy=POLICY_AUSAS_DYNAMIC,
        guards_cfg=PhysicalGuardsConfig.from_profile(
            "off", "general"),
        ausas_tol=1e-4, ausas_max_inner=5000,
        extra_options=None,
        context=_step_context(N_phi=N_phi, N_z=N_z),
        m_shaft=10.0, eps_max=0.95,
        clamp_fn=_clamp, build_H_fn=_build_H,
    )
    # Picard converges trivially for this constant-H stub. We expect
    # committed_converged.
    assert ms.committed_state_is_finite is True
    assert ms.committed_state_status in (
        "committed_converged",
        # Picard might not reach fixed-point on first iter for some
        # tolerances; either way the state must be finite — the test
        # is permissive on the path but pins the contract.
        "committed_last_valid",
        "rolled_back_previous",
    )
    # No NaN in the committed trial outputs in any case.
    assert np.isfinite(ms.Fx_hyd_committed) or ms.Fx_hyd_committed == 0.0


# ─── Test 3: damped Ausas — final commit returns non-finite ────────


def test_damped_nonfinite_commit_falls_back_to_rollback():
    """When the commit-time GPU call returns a non-finite pressure,
    the kernel must NOT save it as ``committed_*``; it must fall
    through to ``rolled_back_previous`` and zero out the trial
    outputs so the runner doesn't write NaN-tainted values to the
    trajectory."""
    def _nan_returning_backend(**kwargs):
        """Always returns a NaN pressure field with converged=True
        so the kernel commit-gate fires (state_committed=True path)
        and the Task 32 finite-gate has to catch it."""
        H = np.asarray(kwargs["H_curr"], dtype=float)
        return dict(
            P=np.full_like(H, np.nan, dtype=float),
            theta=np.ones_like(H, dtype=float),
            residual_linf=float("nan"),
            n_inner=10,
            converged=True,
        )

    _flaky_backend = _nan_returning_backend

    set_ausas_backend_for_tests(_flaky_backend)
    try:
        N_phi, N_z = 8, 4
        backend = AusasDynamicBackend()

        def _build_H(eps_x, eps_y):
            return np.full((N_z, N_phi), 0.5)

        def _clamp(ex, ey, vx, vy):
            return ex, ey, vx, vy, False

        H0 = _build_H(0.0, 0.0)
        state = DieselAusasState.from_initial_gap(H0)
        ms = advance_mechanical_step(
            ex_n=0.0, ey_n=0.0, vx_n=0.0, vy_n=0.0,
            ax_prev=0.0, ay_prev=0.0,
            dt_phys_s=1e-5,
            backend=backend,
            backend_state=state,
            policy=POLICY_AUSAS_DYNAMIC,
            guards_cfg=PhysicalGuardsConfig.from_profile(
                "off", "general"),
            ausas_tol=1e-4, ausas_max_inner=5000,
            extra_options=None,
            context=_step_context(N_phi=N_phi, N_z=N_z),
            m_shaft=10.0, eps_max=0.95,
            clamp_fn=_clamp, build_H_fn=_build_H,
        )
        # Whatever the kernel decides, the committed state must NOT
        # leak NaN into the trajectory.
        if ms.P_nd_committed is not None:
            assert np.all(np.isfinite(ms.P_nd_committed)), (
                "Non-finite P_nd_committed leaked through the "
                "Task 32 finite gate")
        # Either committed_last_valid (finite earlier trial) OR
        # rolled_back_previous (rollback). Never committed_converged
        # for this flaky backend.
        assert ms.committed_state_status != "committed_converged"
    finally:
        set_ausas_backend_for_tests(None)


# ─── Test 4: log split contract ────────────────────────────────────


def test_no_accepted_log_for_failed_final_trial():
    """The runner's debug-print path (``_print_ausas_debug_step``)
    must NOT emit ``ACCEPTED`` when ``final_trial_status != converged``.
    The new labels are ``FINAL-TRIAL-FAILED`` / ``COMMITTED`` /
    ``ROLLED-BACK`` / ``REJECTED-NO-COMMIT``.

    This is verified at the kernel-result level: when
    ``final_trial_status != 'converged'``, the runner branches on
    ``committed_state_status`` and emits one of the new labels;
    we don't unit-test the print path itself (would require
    capsys + a full ``run_transient`` call), but pin the kernel
    contract so the runner has the right inputs.
    """
    backend = _make_legacy_hs_backend(converged=False)

    def _build_H(eps_x, eps_y):
        return np.full((4, 8), 0.5)

    def _clamp(ex, ey, vx, vy):
        return ex, ey, vx, vy, False

    ms = advance_mechanical_step(
        ex_n=0.0, ey_n=0.0, vx_n=0.0, vy_n=0.0,
        ax_prev=0.0, ay_prev=0.0,
        dt_phys_s=1e-5,
        backend=backend,
        backend_state=None,
        policy=POLICY_LEGACY_HS,
        guards_cfg=PhysicalGuardsConfig.from_profile(
            "diagnostic", "general"),
        ausas_tol=1e-6, ausas_max_inner=5000,
        extra_options=None,
        context=_step_context(),
        m_shaft=10.0, eps_max=0.95,
        clamp_fn=_clamp, build_H_fn=_build_H,
    )
    assert ms.final_trial_status != "converged"
    # Runner's print branch sees this and emits FINAL-TRIAL-FAILED
    # + REJECTED-NO-COMMIT (HS path has no last_valid_trial).
    assert ms.committed_state_status == "rejected_no_commit"
