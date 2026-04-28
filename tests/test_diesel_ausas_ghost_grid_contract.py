"""Stage J followup-2 Step 12 (Gate 4) — adapter ghost-grid contract.

Pipeline-side contract test: how
:func:`models.diesel_ausas_adapter.ausas_one_step_with_state` hands
the diesel ``(N_z, N_phi)`` physical arrays to the Ausas backend.

Per expert review the adapter MUST forward the four arrays
(``H_curr`` / ``H_prev`` / ``P_prev`` / ``theta_prev``) in
**ghost-padded** ``(N_z, N_phi + 2)`` form, with the seam wrap::

    padded[:, 0]   = physical[:, -1]      # left ghost = last column
    padded[:, -1]  = physical[:, 0]       # right ghost = first column

The backend output (also padded) MUST be unpadded before the
adapter returns / commits, so ``DieselAusasState`` and
``DieselAusasStepResult.P_nd`` / ``.theta`` stay on the physical
shape ``(N_z, N_phi)``.

Six assertions:

1. ``pad_phi_for_ausas`` / ``unpad_phi_from_ausas`` round-trip
   the seam wrap correctly (left ghost = last physical column,
   right ghost = first physical column).
2. live ``ausas_one_step_with_state`` forwards padded arrays to
   the backend (shape ``(N_z, N_phi+2)``).
3. ``d_phi`` stays at the physical spacing ``2π / N_phi`` —
   never ``2π / (N_phi+2)``.
4. ``periodic_phi=True`` / ``periodic_z=False`` plus the four
   Z-boundary kwargs (``p_bc_z0`` / ``p_bc_zL`` / ``theta_bc_z0``
   / ``theta_bc_zL``) reach the backend explicitly.
5. The adapter unpads the backend's padded output before
   handing it back.
6. ``DieselAusasState`` arrays (``P``, ``theta``, ``H_prev``)
   stay on the unpadded physical shape after a commit.

The fixture is a tiny ``N_z=4``, ``N_phi=6`` grid with an
intentionally non-symmetric scalar field (``field[i, j] =
100*i + j``) — symmetric padding bugs that swap the two seam
columns would not be caught by a uniform-field test.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.diesel_ausas_adapter import (
    DieselAusasState,
    ausas_one_step_with_state,
    pad_phi_for_ausas,
    set_ausas_backend_for_tests,
    unpad_phi_from_ausas,
)


def _field(n_z: int, n_phi: int, offset: float = 0.0) -> np.ndarray:
    """Non-symmetric scalar field. ``field[i, j] = offset + 100*i + j``
    — distinct values per cell so a swap of the two seam columns is
    caught bit-for-bit."""
    i = np.arange(n_z, dtype=float)[:, None]
    j = np.arange(n_phi, dtype=float)[None, :]
    return offset + 100.0 * i + j


def _assert_phi_ghost_packed(padded: np.ndarray,
                              physical: np.ndarray) -> None:
    assert padded.shape == (physical.shape[0],
                             physical.shape[1] + 2), (
        f"padded shape {padded.shape} expected "
        f"{(physical.shape[0], physical.shape[1] + 2)}")
    np.testing.assert_array_equal(padded[:, 1:-1], physical)
    # Left ghost = last physical column (phi = -dphi).
    np.testing.assert_array_equal(padded[:, 0], physical[:, -1])
    # Right ghost = first physical column (phi = 2π).
    np.testing.assert_array_equal(padded[:, -1], physical[:, 0])


# ─── 1. Helper round-trip ─────────────────────────────────────────


def test_pad_phi_for_ausas_seam_wrap():
    """Pure helper contract: the two ghost columns mirror the
    seam (left = last physical, right = first physical), and
    ``unpad`` reverses ``pad`` exactly."""
    n_z, n_phi = 4, 6
    field = _field(n_z, n_phi, offset=0.0)
    padded = pad_phi_for_ausas(field)
    _assert_phi_ghost_packed(padded, field)
    # Round-trip.
    recovered = unpad_phi_from_ausas(padded)
    np.testing.assert_array_equal(recovered, field)


# ─── 2. Live adapter contract ─────────────────────────────────────


def test_ausas_adapter_phi_ghost_grid_contract():
    """Live ``ausas_one_step_with_state`` MUST hand the backend
    ghost-padded arrays AND explicit periodic / Z-boundary
    kwargs; backend output (padded) MUST be unpadded before
    commit; ``DieselAusasState`` MUST stay on the physical shape.

    Stage J integration regression note (Bug 3, 2026-04 expert
    review): assertions are made on ``calls[0]`` AFTER the call
    completes (rather than inside ``fake_ausas_backend``) so a
    shape mismatch surfaces as a direct failure on the kwargs
    field rather than as ``out.P_nd is None`` (the adapter
    swallows backend exceptions and reports the step as
    solver-failed).
    """
    n_z, n_phi = 4, 6

    H_prev = _field(n_z, n_phi, offset=1.0)
    H_curr = _field(n_z, n_phi, offset=10.0)
    P_prev = _field(n_z, n_phi, offset=1000.0)
    theta_prev = np.clip(
        _field(n_z, n_phi, offset=0.0) / 1000.0, 0.0, 1.0)

    state = DieselAusasState(
        P=P_prev.copy(),
        theta=theta_prev.copy(),
        H_prev=H_prev.copy(),
    )

    calls = []

    def fake_ausas_backend(**kwargs):
        """Capture-only stub. Returns a result that matches the
        shape of the H_curr the adapter passed in — so the test
        fails on the *contract* check below, not on a
        shape-mismatch inside the backend."""
        calls.append(kwargs)
        H_in = np.asarray(kwargs["H_curr"])
        P_out = H_in + 10000.0
        theta_out = np.ones_like(P_out) * 0.75
        # If the input was padded, mirror the seam wrap so the
        # output is consistent (real backend behaviour).
        if P_out.shape[1] >= 3:
            P_out[:, 0] = P_out[:, -2]
            P_out[:, -1] = P_out[:, 1]
            theta_out[:, 0] = theta_out[:, -2]
            theta_out[:, -1] = theta_out[:, 1]
        return P_out, theta_out, 1.0e-9, 3

    set_ausas_backend_for_tests(fake_ausas_backend)
    try:
        out = ausas_one_step_with_state(
            state, H_curr=H_curr, dt_s=1.0e-4, omega_shaft=200.0,
            d_phi=2.0 * np.pi / n_phi, d_Z=2.0 / (n_z - 1),
            R=0.1, L=0.08,
            extra_options={
                "tol": 1.0e-6, "max_inner": 50,
                "periodic_phi": True, "periodic_z": False,
                "p_bc_z0": 0.0, "p_bc_zL": 0.0,
                "theta_bc_z0": 1.0, "theta_bc_zL": 1.0,
            },
            commit=True,
        )
    finally:
        set_ausas_backend_for_tests(None)

    assert len(calls) == 1, (
        f"backend not invoked exactly once: {len(calls)} calls")
    ckwargs = calls[0]

    # (2) Padded shape on input — the four arrays the live path
    # ships to the backend. Failure here means the adapter is
    # forwarding unpadded (N_z, N_phi) directly and relying on
    # solver-side ``_pack_ghosts``.
    _assert_phi_ghost_packed(ckwargs["H_curr"], H_curr)
    _assert_phi_ghost_packed(ckwargs["H_prev"], H_prev)
    _assert_phi_ghost_packed(ckwargs["P_prev"], P_prev)
    _assert_phi_ghost_packed(ckwargs["theta_prev"], theta_prev)

    # (3) d_phi is the PHYSICAL spacing — not divided by the
    # padded width.
    assert ckwargs["d_phi"] == pytest.approx(2.0 * np.pi / n_phi)

    # (4) Explicit periodic / Z-boundary kwargs reach the backend.
    assert ckwargs.get("periodic_phi", True) is True
    assert ckwargs.get("periodic_z", False) is False
    assert ckwargs.get("p_bc_z0", 0.0) == pytest.approx(0.0)
    assert ckwargs.get("p_bc_zL", 0.0) == pytest.approx(0.0)
    assert ckwargs.get("theta_bc_z0", 1.0) == pytest.approx(1.0)
    assert ckwargs.get("theta_bc_zL", 1.0) == pytest.approx(1.0)

    # (5) Adapter unpads the backend's padded output.
    assert out.P_nd is not None and out.theta is not None, (
        f"adapter returned None for P_nd / theta; "
        f"reason={out.reason!r}")
    assert out.P_nd.shape == (n_z, n_phi)
    assert out.theta.shape == (n_z, n_phi)
    np.testing.assert_array_equal(out.P_nd, H_curr + 10000.0)
    np.testing.assert_array_equal(out.theta,
                                    np.ones_like(H_curr) * 0.75)

    # (6) Committed state stays on the physical shape.
    assert state.P.shape == (n_z, n_phi)
    assert state.theta.shape == (n_z, n_phi)
    assert state.H_prev.shape == (n_z, n_phi)
    np.testing.assert_array_equal(state.P, out.P_nd)
    np.testing.assert_array_equal(state.theta, out.theta)
    np.testing.assert_array_equal(state.H_prev, H_curr)
    assert state.step_index == 1
