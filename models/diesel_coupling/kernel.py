"""Stage J followup-2 — backend-agnostic mechanical-step kernel.

One mechanical step (one accepted Verlet update) lives entirely
inside :func:`advance_mechanical_step`. The Verlet OUTER loop
remains in ``run_transient`` — only the body of one step (currently
the ``for k in range(N_SUB)`` corrector + the post-clamp commit)
moves here.

Two policies are dispatched on the same kernel call:

* ``POLICY_LEGACY_HS`` — bit-for-bit equivalent to the existing
  3-pass corrector. No line search, no relax shrinking, guards in
  diagnostic mode (warn-only). Locked in by Gate 1.
* ``POLICY_AUSAS_DYNAMIC`` — damped implicit film coupling. Up
  to ``policy.max_mech_inner`` candidate evaluations; relax shrinks
  on rejection; guards in hard mode; no commit on clamped step.

Selection is by the policy ``name`` field, not by backend name —
the same kernel handles both via different relax / guard /
line-search paths.

Step 2 — skeleton only. Step 3 implements ``LegacyVerletPolicy``
path (bit-for-bit move). Step 6 adds ``DampedImplicitFilmPolicy``
skeleton. Step 7-9 wire the guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from .backends import PressureBackend, PressureSolveResult, StepContext
from .guards import (
    GuardOutcome,
    PhysicalGuardsConfig,
    RejectionReason,
)
from .policies import CouplingPolicy


# ─── Per-trial record (debug + summary) ────────────────────────────


@dataclass
class TrialRecord:
    """One pressure-solve attempt within a single mechanical step.

    Captured for the debug printer (Stage J followup-1
    ``--debug-first-steps N``) and for the per-step npz array
    ``trials_per_step`` (Gate 3 schema).
    """
    inner: int
    relax_used: float
    eps_x_cand: float
    eps_y_cand: float
    pressure_result: PressureSolveResult
    outcome_solver: GuardOutcome
    outcome_physical: GuardOutcome
    outcome_mechanical: GuardOutcome
    accepted: bool


# ─── Kernel return value ───────────────────────────────────────────


@dataclass
class MechanicalStepResult:
    """Outcome of one mechanical step.

    On ``accepted=True``:
        new shaft state ``(eps_x_new, eps_y_new, vx_new, vy_new,
        ax_new, ay_new)`` and committed pressure / theta / film
        thickness for downstream diagnostics.
    On ``accepted=False``:
        the runner treats the step as solver-failed for envelope
        statistics; the previous accepted shaft state advances
        only mechanically (Verlet predict, no force) — same
        contract as the legacy 3-pass corrector when the SOR solver
        fails on every k.

    Ausas-state mutation is reported via ``state_committed``: the
    runner asserts ``state_committed → accepted AND not step_clamped
    AND backend.stateful``.
    """
    accepted: bool
    eps_x_new: float
    eps_y_new: float
    vx_new: float
    vy_new: float
    ax_new: float
    ay_new: float

    H_committed: Optional[np.ndarray]
    P_nd_committed: Optional[np.ndarray]
    theta_committed: Optional[np.ndarray]
    Fx_hyd_committed: float
    Fy_hyd_committed: float

    residual: float
    n_inner: int
    n_trials: int
    mech_relax_min_seen: float

    rejection_reason: RejectionReason
    rejection_detail: str
    trial_log: List[TrialRecord]

    state_committed: bool
    step_clamped: bool


# ─── Public kernel entry point ─────────────────────────────────────


def advance_mechanical_step(
    *,
    # Mechanical state at start of step
    ex_n: float,
    ey_n: float,
    vx_n: float,
    vy_n: float,
    ax_prev: float,
    ay_prev: float,
    # Step driving
    dt_phys_s: float,
    # Backend / policy / guards
    backend: PressureBackend,
    backend_state: Optional[Any],
    policy: CouplingPolicy,
    guards_cfg: PhysicalGuardsConfig,
    # Solver knobs (forwarded to the backend's ``extra_options``)
    ausas_tol: float,
    ausas_max_inner: int,
    extra_options: Optional[dict],
    # Geometric / textural / kinematic context
    context: StepContext,
    # Mechanical scaffolding owned by the runner
    m_shaft: float,
    eps_max: float,
    clamp_fn: Callable[
        [float, float, float, float],
        # returns (ex_clamped, ey_clamped, vx_clamped, vy_clamped, clamped_bool)
        Any,
    ],
    build_H_fn: Callable[
        [float, float],
        # returns H array shaped like context.Phi_mesh
        np.ndarray,
    ],
) -> MechanicalStepResult:
    """One mechanical step.

    Half-Sommerfeld (``policy.name == "legacy_verlet"``,
    ``max_mech_inner=3``, no line-search, guards diagnostic):
        Bit-for-bit equivalent to the existing
        ``for k in range(N_SUB):`` corrector body in
        ``run_transient``. Operation order preserved (build H →
        squeeze_to_api_params → solve_reynolds → forces → Verlet
        update → next k → final clamp). This is what Gate 1 locks.

    Ausas dynamic (``policy.name == "damped_implicit_film"``):
        Outer loop runs up to ``policy.max_mech_inner`` trial
        evaluations. Each trial:
            1. Build H_curr from blended candidate ε.
            2. ``backend.solve_trial(commit=False)``.
            3. ``check_solver_validity`` — HARD gate.
            4. ``check_physical_guards`` — HARD/diagnostic per cfg.
            5. ``check_mechanical_candidate`` — Δε bounds.
            6. On any HARD reject: shrink ``mech_relax`` by 0.5,
               re-blend candidate, retry. If ``mech_relax`` falls
               below ``policy.mech_relax_min`` — return
               ``accepted=False`` with the first rejection reason.
        On accept of a non-clamped final candidate, call
        ``backend.solve_trial(commit=True)`` so the stateful backend
        commits ``(P_nd, theta, H_prev)`` to ``backend_state``. If
        the final candidate is clamped, do NOT commit (per
        ``policy.commit_on_clamp == False``).

    Step 2 — skeleton only. Step 3 implements the legacy path;
    Step 6 + 7 + 8 + 9 implement the damped path layer by layer.
    """
    ...
