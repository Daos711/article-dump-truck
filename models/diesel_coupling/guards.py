"""Stage J followup-2 — physical / solver / mechanical guards.

Three guard layers, all gated on ``CouplingPolicy.enable_physical_guards``
and the ``physical_guards_mode`` setting:

* :func:`check_solver_validity` — HARD gate, all backends, always
  enforced. ``n_inner == max_inner`` is **never** considered
  converged (doc 1 §1.2). Theta checks skipped for backends that
  don't return theta (half-Sommerfeld).

* :func:`check_physical_guards` — pressure-amplitude / force-ratio
  / sign-runaway / cavitation-runaway. Active in ``"hard"``,
  warn-only in ``"diagnostic"``, silent in ``"off"``. Smoke vs
  general thresholds via :class:`PhysicalGuardsConfig.from_profile`.

* :func:`check_mechanical_candidate` — bounds the per-inner and
  per-step ε increment so the explicit Verlet corrector cannot
  hand the solver a candidate that activates a fictitious squeeze
  pulse (doc 1 §1.3 / §4.3). Active only when
  ``policy.max_delta_eps_inner`` / ``max_delta_eps_step`` are set.

Thresholds are taken verbatim from doc 1 §3.4 (``AusasPhysicalGuard``
dataclass). All numeric constants are module-level so the brief's
``--ausas-p-hard-mpa`` / ``--ausas-force-ratio-smoke`` etc. CLI
flags can override them via :meth:`PhysicalGuardsConfig.from_profile`.

Step 2 — skeleton only. Step 7 implements solver-validity body;
Step 8 implements physical-guard body; Step 9 implements
mechanical-candidate body. The kernel (Step 9) consumes the
``GuardOutcome`` / ``RejectionReason`` to drive line-search / commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import numpy as np


# ─── Thresholds (doc 1 §3.4 verbatim) ──────────────────────────────


P_DIM_HARD_MAX_PA: float = 1.0e9       # 1 GPa  (general profile cap)
P_DIM_SMOKE_MAX_PA: float = 2.0e8      # 200 MPa (smoke F=0.3 cap)
F_RATIO_HARD_MAX: float = 100.0        # general |F_hyd|/|F_ext|
F_RATIO_SMOKE_MAX: float = 25.0        # smoke |F_hyd|/|F_ext|
SAME_DIR_RUNAWAY_DOT_NORM: float = 0.90
SAME_DIR_RUNAWAY_F_RATIO: float = 10.0
CAV_FRAC_RUNAWAY: float = 0.98

P_NEGATIVE_TOL_ND: float = 1e-12       # P_nd.min() >= -tol
THETA_FLOOR: float = -1e-10            # theta.min() >= floor
THETA_CEIL: float = 1.0 + 1e-10        # theta.max() <= ceil
F_EXT_FLOOR_N: float = 1.0             # avoid div-by-zero on F_ratio


GuardsMode = Literal["off", "diagnostic", "hard"]
GuardsProfile = Literal["general", "smoke"]


@dataclass(frozen=True)
class PhysicalGuardsConfig:
    """Resolved per-run guard configuration.

    Built once by the runner from CLI flags
    (``--guards-profile`` / ``--ausas-physical-guards`` /
    ``--ausas-p-hard-mpa`` / ...). The kernel consumes this
    dataclass on each candidate; ``mode`` decides whether to
    REJECT (``hard``) or WARN (``diagnostic``) or do nothing
    (``off``).
    """
    mode: GuardsMode
    profile: GuardsProfile
    p_dim_max_pa: float
    f_ratio_max: float
    same_dir_runaway_dot_norm: float
    same_dir_runaway_f_ratio: float
    cav_frac_runaway: float
    early_phi_deg: float
    p_negative_tol_nd: float
    theta_floor: float
    theta_ceil: float
    f_ext_floor_n: float

    @classmethod
    def from_profile(
        cls,
        mode: GuardsMode,
        profile: GuardsProfile,
        *,
        p_dim_hard_pa: Optional[float] = None,
        p_dim_smoke_pa: Optional[float] = None,
        f_ratio_hard: Optional[float] = None,
        f_ratio_smoke: Optional[float] = None,
        early_phi_deg: float = 120.0,
    ) -> "PhysicalGuardsConfig":
        """Build a resolved config from profile + optional overrides.

        Step 2 — skeleton only. Step 8 implements the body.
        """
        ...


# ─── Rejection taxonomy ────────────────────────────────────────────


class RejectionReason(str, Enum):
    """Single-source taxonomy for trial / commit rejections.

    Used by:
        * the kernel to decide line-search shrinking vs abort
        * ``MechanicalStepResult.rejection_reason`` for per-step
          npz array
        * the summary writer's per-config rejection counters
          (Gate 3 block)
    """
    NONE = "none"

    # Solver-validity (HARD gate, always-on for all backends).
    SOLVER_NONFINITE = "solver_nonfinite"
    SOLVER_NEG_PRESSURE = "solver_negative_pressure"
    SOLVER_THETA_OUT_OF_RANGE = "solver_theta_out_of_range"
    SOLVER_RESIDUAL = "solver_residual_above_tol"
    SOLVER_BUDGET = "solver_n_inner_at_max"

    # Physical guards (active only when mode='hard').
    PHYSICAL_PRESSURE_GPA = "physical_pressure_above_dim_max"
    PHYSICAL_FORCE_RATIO = "physical_force_ratio_above_max"
    PHYSICAL_SAME_DIR_RUNAWAY = "physical_force_aligned_with_load"
    PHYSICAL_CAV_RUNAWAY = "physical_cav_frac_runaway"

    # Mechanical candidate guards (active only when policy provides
    # max_delta_eps_inner / max_delta_eps_step).
    MECHANICAL_DELTA_EPS_INNER = "mechanical_delta_eps_inner_exceeded"
    MECHANICAL_DELTA_EPS_STEP = "mechanical_delta_eps_step_exceeded"
    MECHANICAL_EPS_MAX = "mechanical_eps_at_clamp"


@dataclass
class GuardOutcome:
    """Single guard's verdict on one trial candidate.

    ``accept=True`` is the only path that lets the kernel proceed
    to the next gate. ``detail`` is human-readable for the debug
    logger and the summary writer's first-reject-reason slot.
    """
    accept: bool
    reason: RejectionReason
    detail: str = ""


# ─── Guard functions (skeleton signatures) ─────────────────────────


def check_solver_validity(
    *,
    backend_name: str,
    P_nd: Optional[np.ndarray],
    theta: Optional[np.ndarray],
    residual: float,
    n_inner: int,
    converged: bool,
    ausas_tol: float,
    ausas_max_inner: int,
) -> GuardOutcome:
    """HARD gate — always enforced (doc 1 §3.1).

    Composition:
        * ``np.isfinite(residual)``
        * ``residual <= ausas_tol``  (only when ``converged is False``;
          if the backend explicitly reports ``converged=True``, trust
          it on the residual front)
        * ``n_inner < ausas_max_inner``  (NEVER accept budget-exhaust)
        * ``np.isfinite(P_nd).all()``
        * ``P_nd.min() >= -P_NEGATIVE_TOL_ND``
        * (theta-bearing backends) ``np.isfinite(theta).all()``
        * (theta-bearing backends) ``theta`` in ``[THETA_FLOOR, THETA_CEIL]``

    Half-Sommerfeld: ``theta is None`` → theta checks skipped;
    ``residual``/``n_inner`` come from the SOR solver.

    Step 2 — skeleton only. Step 7 implements the body.
    """
    ...


def check_physical_guards(
    *,
    P_nd: np.ndarray,
    p_scale: float,
    Fx_hyd: float,
    Fy_hyd: float,
    Fx_ext: float,
    Fy_ext: float,
    theta: Optional[np.ndarray],
    phi_deg: float,
    cfg: PhysicalGuardsConfig,
) -> GuardOutcome:
    """Active in ``mode='hard'``; emits warning in ``'diagnostic'``;
    silent in ``'off'``.

    Composition (doc 1 §3.2 + §4.2):
        * ``max(P_nd * p_scale) <= cfg.p_dim_max_pa``
        * ``|F_hyd| / max(|F_ext|, cfg.f_ext_floor_n) <= cfg.f_ratio_max``
        * NOT (``dot_norm > cfg.same_dir_runaway_dot_norm`` AND
          ``F_ratio > cfg.same_dir_runaway_f_ratio``) — same-direction
          runaway gate
        * ``cav_frac <= cfg.cav_frac_runaway``  (theta-bearing only)
        * ``theta`` in [floor, ceil]  (theta-bearing only)

    Smoke profile additionally tightens ``p_dim_max_pa`` to 200 MPa
    and ``f_ratio_max`` to 25.0; applies only on the early-cycle
    window ``phi_deg < cfg.early_phi_deg``.

    Step 2 — skeleton only. Step 8 implements the body.
    """
    ...


def check_mechanical_candidate(
    *,
    eps_x_curr: float,
    eps_y_curr: float,
    eps_x_cand: float,
    eps_y_cand: float,
    eps_x_step_start: float,
    eps_y_step_start: float,
    eps_max: float,
    max_delta_eps_inner: Optional[float],
    max_delta_eps_step: Optional[float],
) -> GuardOutcome:
    """Active only when ``policy.max_delta_eps_inner`` /
    ``max_delta_eps_step`` are set (i.e. damped policy).

    Composition (doc 1 §3.3 + §4.3):
        * ``|Δε_inner|  = hypot(eps_x_cand - eps_x_curr,
                                 eps_y_cand - eps_y_curr)
                         <= max_delta_eps_inner``
        * ``|Δε_step|   = hypot(eps_x_cand - eps_x_step_start,
                                 eps_y_cand - eps_y_step_start)
                         <= max_delta_eps_step``
        * ``hypot(eps_x_cand, eps_y_cand) < eps_max``

    On violation the **kernel** must shrink ``mech_relax`` and
    re-blend the candidate, NOT push the violating candidate to
    the solver.

    Step 2 — skeleton only. Step 9 implements the body.
    """
    ...
