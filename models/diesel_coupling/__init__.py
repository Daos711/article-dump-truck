"""Stage J followup-2 — backend-agnostic mechanical coupling layer.

Public API re-exports::

    PressureBackend, PressureSolveResult, StepContext
    HalfSommerfeldBackend, AusasDynamicBackend
    CouplingPolicy, POLICY_LEGACY_HS, POLICY_AUSAS_DYNAMIC
    select_policy, resolve_policy, resolve_policy_overrides
    PhysicalGuardsConfig, GuardsMode, GuardsProfile, RejectionReason
    GuardOutcome
    check_solver_validity, check_physical_guards, check_mechanical_candidate
    advance_mechanical_step, MechanicalStepResult, TrialRecord

The runner (``models.diesel_transient.run_transient``) imports
``advance_mechanical_step`` and constructs the appropriate
``PressureBackend`` from the ``cavitation`` kwarg; it never branches
on backend name. This is what makes future PV (Stage K) trivial:
PV-on-half-Sommerfeld stays stateless and selects the legacy policy;
PV-on-Ausas is stateful and selects the damped policy. No new
``if use_*_dynamic:`` branches in the runner.

This package is **skeleton-only** at the post-Step-2 commit
(empty bodies marked ``...``). Step 3 moves the existing Verlet
substep loop body into the kernel as ``LegacyVerletPolicy.run`` —
the runner-side wiring lands in Step 4. Until Step 7+ no guards
are active; until Step 9 no mechanical line-search is wired.
"""

from .backends import (
    AusasDynamicBackend,
    HalfSommerfeldBackend,
    PressureBackend,
    PressureSolveResult,
    StepContext,
)
from .failure_classifier import (
    BucketCounts,
    FAILURE_BUCKETS,
    StepDiagnosticRow,
    aggregate_buckets,
    classify_failure,
)
from .guards import (
    GuardOutcome,
    GuardsMode,
    GuardsProfile,
    PhysicalGuardsConfig,
    RejectionReason,
    check_mechanical_candidate,
    check_physical_guards,
    check_solver_validity,
)
from .kernel import (
    MechanicalStepResult,
    TrialRecord,
    advance_mechanical_step,
)
from .policies import (
    CouplingPolicy,
    POLICY_AUSAS_DYNAMIC,
    POLICY_LEGACY_HS,
    resolve_policy,
    resolve_policy_overrides,
    select_policy,
)

__all__ = [
    "AusasDynamicBackend",
    "BucketCounts",
    "CouplingPolicy",
    "FAILURE_BUCKETS",
    "GuardOutcome",
    "GuardsMode",
    "GuardsProfile",
    "HalfSommerfeldBackend",
    "MechanicalStepResult",
    "PhysicalGuardsConfig",
    "POLICY_AUSAS_DYNAMIC",
    "POLICY_LEGACY_HS",
    "PressureBackend",
    "PressureSolveResult",
    "RejectionReason",
    "StepContext",
    "StepDiagnosticRow",
    "TrialRecord",
    "advance_mechanical_step",
    "aggregate_buckets",
    "check_mechanical_candidate",
    "check_physical_guards",
    "check_solver_validity",
    "classify_failure",
    "resolve_policy",
    "resolve_policy_overrides",
    "select_policy",
]
