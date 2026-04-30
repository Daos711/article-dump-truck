"""Stage J fu-2 — failure-bucket classifier (Task 14).

Per-step classification of why a transient step failed. Replaces
the legacy ``Solver: failed=N`` aggregate with one of:

* ``ok`` — step succeeded, no failure to classify;
* ``solver_budget`` — Ausas hit ``max_inner`` without converging;
* ``solver_residual`` — Ausas returned ``residual > tol`` while
  reporting ``converged=False``;
* ``solver_nonfinite`` — non-finite ``p_max`` or ``theta_max``;
* ``picard_not_converged`` — coupling Picard exhausted
  ``max_mech_inner`` without reaching the fixed point;
* ``picard_relax_floor`` — Picard sat at the ``mech_relax_min``
  floor (contractivity detector hit bottom);
* ``reject_at_anchor`` — ``rejection_reason`` carries the
  ``reject_at_anchor_*`` tag (kernel aborted line-search at the
  anchor itself);
* ``mechanical_guard`` — mechanical-candidate guard fired;
* ``physical_guard`` — physical-pressure / theta-range / force-
  balance guard fired;
* ``unknown`` — step is_failure but none of the above rules
  matched (sentinel — should not happen in practice; flag in
  summary so the rule-set can be extended).

The function is pure (no I/O, no GPU) so it can be exercised by
synthetic-step unit tests without spinning up a transient run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


# ─── Per-step input ────────────────────────────────────────────────


@dataclass
class StepDiagnosticRow:
    """One step worth of post-run diagnostic numbers.

    All fields come from the npz arrays saved by the transient
    runner. The classifier consumes this struct rather than a
    dict so the field names are pinned in one place.
    """
    is_failure: bool
    rejection_reason: str
    # Ausas-side
    ausas_n_inner: int
    ausas_residual: float
    ausas_converged: bool
    p_max: float
    theta_max: float
    # Coupling-side
    fp_converged: bool
    n_trials: int
    mech_relax_min_seen: float
    # Run-level scalars (constant per run; threaded in for the
    # comparison thresholds)
    ausas_tol: float
    ausas_max_inner: int
    max_mech_inner: int
    mech_relax_min: float
    # Stage J fu-2 Task 27 — explicit nonfinite signals from the
    # gpu-reynolds Task 12 dict-API. Optional: defaults preserve
    # legacy archive compatibility (``failure_kind=""`` and
    # ``nonfinite_count=0`` map to "no info" → fall through to the
    # numerical-residual rules).
    failure_kind: str = ""
    nonfinite_count: int = 0
    # Force-balance non-finite signal — set by the runner when
    # ``F_hyd_x`` or ``F_hyd_y`` are NaN/Inf despite
    # ``ausas_converged=True``. Pure pass-through here.
    force_nonfinite: bool = False


# ─── Classifier ────────────────────────────────────────────────────


# Public bucket names. Order matters for tie-breaks in
# ``classify_failure``: solver issues take priority over Picard,
# Picard over rejection-reason tags, rejection-reason tags over
# the catch-all ``unknown``.
FAILURE_BUCKETS: List[str] = [
    "ok",
    # Stage J fu-2 Task 27 — nonfinite ALWAYS wins over the other
    # solver buckets so a NaN residual / state / failure_kind
    # signal isn't shadowed by a finite-but-above-tol rule. Order
    # in this list is documentation; ``classify_failure`` enforces
    # the priority programmatically.
    "solver_nonfinite",
    "solver_budget",
    "solver_residual",
    # Stage J fu-2 Task 27 — Picard non-convergence is its own
    # bucket. Previously emitted as ``picard_not_converged`` from
    # the ``SOLVER_RESIDUAL`` rejection-reason text; now driven by
    # the dedicated ``RejectionReason.PICARD_NOT_CONVERGED`` enum
    # and renamed to ``picard_noncontractive`` to match the spec
    # vocabulary. The legacy alias ``picard_not_converged`` is
    # NOT preserved as a separate bucket — every Picard fail
    # routes through ``picard_noncontractive`` regardless of
    # whether the rejection-reason text or the structural
    # fp_converged-fallback drove the dispatch.
    "picard_noncontractive",
    "picard_relax_floor",
    "reject_at_anchor",
    "mechanical_guard",
    "physical_guard",
    "unknown",
]


def classify_failure(step: StepDiagnosticRow) -> str:
    """Return the failure bucket for one step.

    Stage J fu-2 Task 27 priority — *nonfinite always wins*. The
    runner saves ``pmax=NaN`` on every failed step regardless of
    cause, but a NaN ``residual`` / ``failure_kind=nonfinite_state``
    / ``failure_kind=invalid_input`` are first-class signals from
    the Ausas one-step solver that the GPU returned a non-finite
    state — they MUST not be misclassified as
    ``solver_residual`` (residual above tol) or
    ``solver_budget`` (budget exhausted but residual finite).

    Rule order (priority, top to bottom):

    1. ``is_failure=False`` → ``ok``.
    2. **Nonfinite first** — any of:
       a. ``failure_kind in {nonfinite_state, invalid_input}``
          (post-Task-12 explicit signal from gpu-reynolds);
       b. ``nonfinite_count > 0``;
       c. ``residual`` non-finite (NaN / Inf);
       d. ``p_max`` or ``theta_max`` non-finite;
       e. ``force_nonfinite=True`` (runner detected NaN F_hyd
          even though Ausas returned converged);
       → ``solver_nonfinite``.
    3. Canonical kernel rejection-reason tags
       (``RejectionReason`` enum strings):
       a. ``solver_n_inner_at_max``               → ``solver_budget``;
       b. ``solver_residual_above_tol``           → ``solver_residual``;
       c. ``solver_nonfinite`` / ``solver_negative_pressure`` /
          ``solver_theta_out_of_range``           → ``solver_nonfinite``;
       d. ``damped_picard_not_converged ...``     → ``picard_noncontractive``;
       e. ``reject_at_anchor_*``                  → ``reject_at_anchor``;
       f. ``mechanical_*``                        → ``mechanical_guard``;
       g. ``physical_*``                          → ``physical_guard``.
    4. Numerical fallbacks (apply only when steps 2-3 didn't fire):
       a. ``ausas_n_inner >= ausas_max_inner`` AND finite residual
          AND not converged                       → ``solver_budget``;
       b. finite ``ausas_residual > ausas_tol`` AND not converged
                                                  → ``solver_residual``;
       c. ``not fp_converged`` AND
          ``n_trials >= max_mech_inner``          → ``picard_noncontractive``;
       d. ``mech_relax_min_seen`` at the floor
          (within 1%)                             → ``picard_relax_floor``.
    5. None matched → ``unknown``.
    """
    if not step.is_failure:
        return "ok"

    # 2 — Nonfinite-first. Any of these signals indicates the
    # solver's state itself is pathological; bucketing it as a
    # finite-residual failure would hide the real problem from
    # the operator.
    if step.failure_kind in ("nonfinite_state", "invalid_input"):
        return "solver_nonfinite"
    if int(step.nonfinite_count) > 0:
        return "solver_nonfinite"
    if not _is_finite(step.ausas_residual):
        return "solver_nonfinite"
    if (not _is_finite(step.p_max)
            or not _is_finite(step.theta_max)):
        return "solver_nonfinite"
    if bool(step.force_nonfinite):
        return "solver_nonfinite"

    rr = str(step.rejection_reason or "")

    # 3 — canonical rejection-reason tags. Use ``in`` / ``startswith``
    # because the kernel may decorate the bucket name with a detail
    # (e.g. ``reject_at_anchor_solver_residual: residual too large``).
    if "solver_n_inner_at_max" in rr:
        return "solver_budget"
    if "solver_residual_above_tol" in rr:
        return "solver_residual"
    if ("solver_nonfinite" in rr
            or "solver_negative_pressure" in rr
            or "solver_theta_out_of_range" in rr):
        return "solver_nonfinite"
    if "damped_picard_not_converged" in rr:
        return "picard_noncontractive"
    if rr.startswith("reject_at_anchor_"):
        return "reject_at_anchor"
    # ``RejectionReason`` mechanical-guard string values
    # (see guards.py):
    #   mechanical_delta_eps_inner / _step / mechanical_eps_max_wall
    if rr.startswith("mechanical_"):
        return "mechanical_guard"
    if rr.startswith("physical_"):
        return "physical_guard"

    # 4 — numerical fallbacks for archived or partially-saved runs
    # where the rejection-reason text isn't available. ``ausas_tol``
    # / ``ausas_max_inner`` come from the run's ``ausas_options``.
    converged = bool(step.ausas_converged)
    residual_over_tol = (
        _is_finite(step.ausas_residual)
        and step.ausas_residual > step.ausas_tol
    )
    if not converged and residual_over_tol:
        if int(step.ausas_n_inner) >= int(step.ausas_max_inner):
            return "solver_budget"
        return "solver_residual"
    if (not bool(step.fp_converged)
            and int(step.n_trials) >= int(step.max_mech_inner)):
        return "picard_noncontractive"
    # Floor check uses 1% tolerance: shrink halves to 2× the
    # floor in one step, so anything within 1% of the floor
    # means the contractivity detector pinned the relax there.
    if (_is_finite(step.mech_relax_min_seen)
            and step.mech_relax_min_seen <= step.mech_relax_min * 1.01):
        return "picard_relax_floor"

    return "unknown"


def _is_finite(x: float) -> bool:
    """Robust finite-check — works on NaN, +/-inf, plain floats,
    and numpy scalars without importing math.isfinite."""
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


# ─── Aggregation helpers ───────────────────────────────────────────


def aggregate_buckets(
    rows: Iterable[StepDiagnosticRow],
) -> "BucketCounts":
    """Run ``classify_failure`` over all rows and return per-bucket
    counts plus the total.
    """
    counts = {b: 0 for b in FAILURE_BUCKETS}
    n_total = 0
    for r in rows:
        counts[classify_failure(r)] += 1
        n_total += 1
    return BucketCounts(
        per_bucket=counts,
        n_total=n_total,
    )


@dataclass
class BucketCounts:
    """Counts per bucket (incl. ``ok``) plus the total number of
    rows aggregated. ``n_failures`` and ``frac_*`` are derived."""
    per_bucket: dict
    n_total: int

    @property
    def n_failures(self) -> int:
        return self.n_total - int(self.per_bucket.get("ok", 0))

    def frac(self, bucket: str) -> float:
        if self.n_total <= 0:
            return 0.0
        return float(self.per_bucket.get(bucket, 0)) / float(self.n_total)

    def frac_of_failures(self, bucket: str) -> float:
        if self.n_failures <= 0:
            return 0.0
        return (float(self.per_bucket.get(bucket, 0))
                / float(self.n_failures))

    def dominant_failure_bucket(self) -> Optional[str]:
        """Bucket with the highest failure count (excluding ``ok``).
        Returns ``None`` if there were no failures."""
        if self.n_failures <= 0:
            return None
        candidates = {
            b: c for b, c in self.per_bucket.items()
            if b != "ok" and c > 0
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get)


__all__ = [
    "FAILURE_BUCKETS",
    "StepDiagnosticRow",
    "BucketCounts",
    "classify_failure",
    "aggregate_buckets",
]
