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

        Profile choice picks the active ``p_dim_max_pa`` and
        ``f_ratio_max`` limits:

        * ``general`` → ``P_DIM_HARD_MAX_PA``, ``F_RATIO_HARD_MAX``
        * ``smoke``   → ``P_DIM_SMOKE_MAX_PA``, ``F_RATIO_SMOKE_MAX``

        Each of the four numeric thresholds can be overridden via the
        matching ``p_dim_hard_pa`` / ``p_dim_smoke_pa`` /
        ``f_ratio_hard`` / ``f_ratio_smoke`` kwargs (CLI flags surface
        them in Step 10). Other thresholds (``same_dir_runaway_*``,
        ``cav_frac_runaway``, theta tolerances, ε-floor) come from the
        module-level constants verbatim from doc 1 §3.4.
        """
        p_hard = (float(p_dim_hard_pa) if p_dim_hard_pa is not None
                  else P_DIM_HARD_MAX_PA)
        p_smoke = (float(p_dim_smoke_pa) if p_dim_smoke_pa is not None
                   else P_DIM_SMOKE_MAX_PA)
        fr_hard = (float(f_ratio_hard) if f_ratio_hard is not None
                   else F_RATIO_HARD_MAX)
        fr_smoke = (float(f_ratio_smoke) if f_ratio_smoke is not None
                    else F_RATIO_SMOKE_MAX)
        if profile == "smoke":
            p_dim_max = p_smoke
            f_ratio_max = fr_smoke
        else:
            p_dim_max = p_hard
            f_ratio_max = fr_hard
        return cls(
            mode=mode,
            profile=profile,
            p_dim_max_pa=p_dim_max,
            f_ratio_max=f_ratio_max,
            same_dir_runaway_dot_norm=SAME_DIR_RUNAWAY_DOT_NORM,
            same_dir_runaway_f_ratio=SAME_DIR_RUNAWAY_F_RATIO,
            cav_frac_runaway=CAV_FRAC_RUNAWAY,
            early_phi_deg=float(early_phi_deg),
            p_negative_tol_nd=P_NEGATIVE_TOL_ND,
            theta_floor=THETA_FLOOR,
            theta_ceil=THETA_CEIL,
            f_ext_floor_n=F_EXT_FLOOR_N,
        )


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

    # Stage J fu-2 Task 27 — Picard fixed-point coupling reasons.
    # Distinct from ``SOLVER_RESIDUAL`` so failure_classifier can
    # bucket Picard-non-convergence as ``picard_noncontractive``
    # rather than mislabelling it as a solver issue. Emitted by the
    # damped path when ``solve_ok=True`` (Ausas converged on the
    # last trial) but the Picard map didn't reach a fixed point
    # within ``max_mech_inner`` iterations.
    PICARD_NOT_CONVERGED = "damped_picard_not_converged"


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
    """HARD gate — Stage J fu-2 §3.1.

    Composition (priority order — first failing check wins):

    1. ``P_nd is None``                           → SOLVER_NONFINITE
    2. ``P_nd`` contains NaN / Inf                → SOLVER_NONFINITE
    3. ``P_nd.min() < -P_NEGATIVE_TOL_ND``        → SOLVER_NEG_PRESSURE
    4. (theta-bearing only) ``theta`` non-finite  → SOLVER_NONFINITE
    5. (theta-bearing only) theta out of [floor, ceil]
                                                  → SOLVER_THETA_OUT_OF_RANGE
    6. ``n_inner >= ausas_max_inner``             → SOLVER_BUDGET
       — doc 1 §1.2: "n_inner == max_inner means
       the loop exhausted the budget. It must not become
       converged=True even if the last sampled residual is small."
    7. ``residual > ausas_tol`` (only if finite)  → SOLVER_RESIDUAL
       — non-finite residual on a converged backend is allowed
       (legacy SOR doesn't surface a residual).

    Half-Sommerfeld (``theta is None``) skips checks 4-5; the
    residual check still runs but a NaN residual + converged=True
    is permitted. Backends that DO surface residual (Ausas) get
    the residual check enforced when finite.
    """
    if P_nd is None:
        return GuardOutcome(
            False, RejectionReason.SOLVER_NONFINITE,
            f"{backend_name}: P_nd is None")
    P_arr = np.asarray(P_nd)
    if not np.all(np.isfinite(P_arr)):
        return GuardOutcome(
            False, RejectionReason.SOLVER_NONFINITE,
            f"{backend_name}: P_nd has NaN/Inf")
    if float(P_arr.min()) < -P_NEGATIVE_TOL_ND:
        return GuardOutcome(
            False, RejectionReason.SOLVER_NEG_PRESSURE,
            f"{backend_name}: min(P_nd)={float(P_arr.min()):.3e} "
            f"< -{P_NEGATIVE_TOL_ND:.1e}")
    if theta is not None:
        theta_arr = np.asarray(theta)
        if not np.all(np.isfinite(theta_arr)):
            return GuardOutcome(
                False, RejectionReason.SOLVER_NONFINITE,
                f"{backend_name}: theta has NaN/Inf")
        if (float(theta_arr.min()) < THETA_FLOOR
                or float(theta_arr.max()) > THETA_CEIL):
            return GuardOutcome(
                False, RejectionReason.SOLVER_THETA_OUT_OF_RANGE,
                f"{backend_name}: theta=["
                f"{float(theta_arr.min()):.3e}, "
                f"{float(theta_arr.max()):.3e}] outside "
                f"[{THETA_FLOOR:.1e}, {THETA_CEIL:.3f}]")
    if int(n_inner) >= int(ausas_max_inner):
        return GuardOutcome(
            False, RejectionReason.SOLVER_BUDGET,
            f"{backend_name}: n_inner={int(n_inner)} >= "
            f"max_inner={int(ausas_max_inner)} — budget exhausted, "
            "doc 1 §1.2 forbids accepting this as converged")
    if np.isfinite(residual):
        if float(residual) > float(ausas_tol):
            return GuardOutcome(
                False, RejectionReason.SOLVER_RESIDUAL,
                f"{backend_name}: residual={float(residual):.3e} > "
                f"tol={float(ausas_tol):.3e}")
    else:
        # Non-finite residual: trust the backend's ``converged`` flag.
        # Legacy half-Sommerfeld SOR doesn't surface a residual; the
        # solver's own convergence check fires when ok=True.
        if not bool(converged):
            return GuardOutcome(
                False, RejectionReason.SOLVER_NONFINITE,
                f"{backend_name}: residual is non-finite and "
                "converged=False")
    return GuardOutcome(True, RejectionReason.NONE, "ok")


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
    """Stage J fu-2 §3.2 / doc 1 §4.2 — pressure / force / sign /
    cavitation HARD guards.

    Returns a verdict regardless of ``cfg.mode`` so the kernel can
    populate ``TrialRecord.outcome_physical`` consistently. The
    KERNEL decides what to do with the verdict:

    * ``mode == "hard"``       → reject the trial (break Picard loop).
    * ``mode == "diagnostic"`` → emit a warning, accept the trial.
    * ``mode == "off"``        → no checks, returned verdict is
      always ``GuardOutcome(True, NONE, "off")``.

    Composition (priority order — first failing check wins):

    1. Pressure-amplitude cap: ``max(P_nd · p_scale) <= cfg.p_dim_max_pa``
       — caps GPa-runaway from squeeze-driven Verlet jumps. The
       ``smoke`` profile additionally applies a 200 MPa cap on the
       early-cycle window ``phi_deg < cfg.early_phi_deg``;
       ``general`` profile uses 1 GPa.
    2. Force-ratio cap: ``|F_hyd| / max(|F_ext|, F_FLOOR_N) <=
       cfg.f_ratio_max``. Smoke: 25; general: 100.
    3. Same-direction runaway: rejects when the hydrodynamic force
       points ALONG the external load (positive cosine) AND its
       magnitude is many times the external load. This is the
       signature of a film blow-up that drives the journal further
       into the wall instead of resisting.
    4. Cavitation runaway: ``mean(theta < 1) <= cfg.cav_frac_runaway``
       — catches global cavitation collapse (typically > 0.98).
    """
    if cfg.mode == "off":
        return GuardOutcome(True, RejectionReason.NONE, "off")

    # Pressure cap.
    if P_nd is not None:
        P_arr = np.asarray(P_nd)
        if P_arr.size > 0:
            p_dim_max = float(np.max(P_arr)) * float(p_scale)
            cap = float(cfg.p_dim_max_pa)
            in_smoke_window = (
                cfg.profile == "smoke"
                and float(phi_deg) < cfg.early_phi_deg
            )
            label = ("smoke-window" if in_smoke_window
                     else cfg.profile)
            if p_dim_max > cap:
                return GuardOutcome(
                    False,
                    RejectionReason.PHYSICAL_PRESSURE_GPA,
                    f"max(P_dim)={p_dim_max:.3e}Pa > cap="
                    f"{cap:.3e}Pa ({label} profile, "
                    f"phi={phi_deg:.1f}°)")

    # Force-ratio + same-direction runaway.
    F_hyd_mag = float(np.hypot(Fx_hyd, Fy_hyd))
    F_ext_mag = float(np.hypot(Fx_ext, Fy_ext))
    if (np.isfinite(F_hyd_mag) and np.isfinite(F_ext_mag)
            and F_ext_mag > float(cfg.f_ext_floor_n)):
        F_ratio = F_hyd_mag / F_ext_mag
        if F_ratio > float(cfg.f_ratio_max):
            return GuardOutcome(
                False, RejectionReason.PHYSICAL_FORCE_RATIO,
                f"|F_hyd|/|F_ext|={F_ratio:.2f} > cap="
                f"{cfg.f_ratio_max:.2f} (|F_hyd|="
                f"{F_hyd_mag/1e3:.2f}kN, |F_ext|="
                f"{F_ext_mag/1e3:.2f}kN)")
        if F_hyd_mag > 0.0:
            dot_norm = ((float(Fx_hyd) * float(Fx_ext)
                         + float(Fy_hyd) * float(Fy_ext))
                        / (F_hyd_mag * F_ext_mag))
            if (dot_norm > float(cfg.same_dir_runaway_dot_norm)
                    and F_ratio
                    > float(cfg.same_dir_runaway_f_ratio)):
                return GuardOutcome(
                    False, RejectionReason.PHYSICAL_SAME_DIR_RUNAWAY,
                    f"dot_norm={dot_norm:+.3f} > "
                    f"{cfg.same_dir_runaway_dot_norm:.2f} AND "
                    f"F_ratio={F_ratio:.2f} > "
                    f"{cfg.same_dir_runaway_f_ratio:.2f}")

    # Cavitation runaway (theta-bearing backends only).
    if theta is not None:
        theta_arr = np.asarray(theta)
        if theta_arr.size > 0:
            cav_frac = float(np.mean(theta_arr < 1.0))
            if cav_frac > float(cfg.cav_frac_runaway):
                return GuardOutcome(
                    False, RejectionReason.PHYSICAL_CAV_RUNAWAY,
                    f"cav_frac={cav_frac:.3f} > cap="
                    f"{cfg.cav_frac_runaway:.3f}")

    return GuardOutcome(True, RejectionReason.NONE, "ok")


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
    """Stage J fu-2 §3.3 / doc 1 §4.3 — mechanical candidate guards.

    Three priority-ordered checks:

    1. ``|Δε_inner| <= max_delta_eps_inner`` — bounds the per-trial
       jump from the LAST accepted (or anchor) candidate to the new
       blended candidate. Catches squeeze-driven Verlet runaway
       within one mechanical step.
    2. ``|Δε_step| <= max_delta_eps_step`` — bounds the cumulative
       drift from the start of the mechanical step to the current
       candidate. Catches a slow accumulation that individually
       passes the inner bound but globally exceeds the step
       envelope.
    3. ``|ε| < eps_max`` — the candidate must NOT have been pushed
       to (or past) the absolute eccentricity clamp. Hitting
       ``eps_max`` means a Verlet jump put the journal on the
       wall; the kernel should shrink relax and re-blend rather
       than commit a clamped candidate.

    Both Δε bounds are skipped when their corresponding policy
    field is ``None`` (legacy_verlet uses neither). The inputs are
    expected in NON-DIMENSIONAL ε units (i.e., ``ex / c``).

    On violation the **kernel** shrinks ``mech_relax`` and
    re-blends; the violating candidate is NEVER pushed to the
    pressure backend. Guard-only — does not touch state.
    """
    if max_delta_eps_inner is not None:
        delta_inner = float(np.hypot(
            float(eps_x_cand) - float(eps_x_curr),
            float(eps_y_cand) - float(eps_y_curr),
        ))
        if delta_inner > float(max_delta_eps_inner):
            return GuardOutcome(
                False,
                RejectionReason.MECHANICAL_DELTA_EPS_INNER,
                f"|Δε_inner|={delta_inner:.4f} > "
                f"cap={float(max_delta_eps_inner):.4f}",
            )
    if max_delta_eps_step is not None:
        delta_step = float(np.hypot(
            float(eps_x_cand) - float(eps_x_step_start),
            float(eps_y_cand) - float(eps_y_step_start),
        ))
        if delta_step > float(max_delta_eps_step):
            return GuardOutcome(
                False,
                RejectionReason.MECHANICAL_DELTA_EPS_STEP,
                f"|Δε_step|={delta_step:.4f} > "
                f"cap={float(max_delta_eps_step):.4f}",
            )
    eps_mag = float(np.hypot(float(eps_x_cand), float(eps_y_cand)))
    if eps_mag >= float(eps_max):
        return GuardOutcome(
            False,
            RejectionReason.MECHANICAL_EPS_MAX,
            f"|ε|={eps_mag:.4f} >= eps_max={float(eps_max):.4f} "
            "— candidate pushed to clamp; shrink relax instead",
        )
    return GuardOutcome(True, RejectionReason.NONE, "ok")
