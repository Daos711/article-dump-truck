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


# ─── Classifier ────────────────────────────────────────────────────


# Public bucket names. Order matters for tie-breaks in
# ``classify_failure``: solver issues take priority over Picard,
# Picard over rejection-reason tags, rejection-reason tags over
# the catch-all ``unknown``.
FAILURE_BUCKETS: List[str] = [
    "ok",
    "solver_budget",
    "solver_residual",
    "solver_nonfinite",
    "picard_not_converged",
    "picard_relax_floor",
    "reject_at_anchor",
    "mechanical_guard",
    "physical_guard",
    "unknown",
]


def classify_failure(step: StepDiagnosticRow) -> str:
    """Return the failure bucket for one step.

    Rule order:

    1. ``is_failure=False`` → ``ok``.
    2. Solver-side first (the runner only reaches the coupling
       Picard if the trial returned a finite, plausibly converged
       pressure; classifying solver failures first means a
       solver-budget step doesn't get mislabelled as
       ``picard_not_converged``).
       a. non-finite p_max / theta_max → ``solver_nonfinite``;
       b. ``ausas_n_inner >= ausas_max_inner`` AND
          ``ausas_residual > ausas_tol`` AND not converged
          → ``solver_budget``;
       c. ``ausas_residual > ausas_tol`` AND not converged
          → ``solver_residual`` (residual didn't reach tolerance,
          but the budget wasn't exhausted — typically a
          divergence / numerical-stall).
    3. Coupling Picard:
       a. ``rejection_reason`` carries the
          ``damped_picard_not_converged`` text → that;
       b. structural fallback: ``not fp_converged`` AND
          ``n_trials >= max_mech_inner`` → ``picard_not_converged``;
       c. Picard sat at the ``mech_relax_min`` floor (within 1%):
          → ``picard_relax_floor``.
    4. Rejection-reason tag dispatch:
       a. starts with ``reject_at_anchor_`` → ``reject_at_anchor``;
       b. equals one of the mechanical-guard buckets
          (``mechanical_delta_eps_inner`` / ``..._step`` /
          ``mechanical_eps_max_wall``) → ``mechanical_guard``;
       c. starts with ``physical_`` → ``physical_guard``.
    5. None matched → ``unknown``.
    """
    if not step.is_failure:
        return "ok"

    # 2a — non-finite output is the most-specific solver failure.
    if (not _is_finite(step.p_max)
            or not _is_finite(step.theta_max)):
        return "solver_nonfinite"

    # 2b/c — solver residual didn't make it.
    converged = bool(step.ausas_converged)
    residual_over_tol = (
        _is_finite(step.ausas_residual)
        and step.ausas_residual > step.ausas_tol
    )
    if not converged and residual_over_tol:
        if int(step.ausas_n_inner) >= int(step.ausas_max_inner):
            return "solver_budget"
        return "solver_residual"

    # 3 — Picard fixed-point coupling.
    rr = str(step.rejection_reason or "")
    if "damped_picard_not_converged" in rr:
        return "picard_not_converged"
    if (not bool(step.fp_converged)
            and int(step.n_trials) >= int(step.max_mech_inner)):
        return "picard_not_converged"
    # Floor check uses 1% tolerance: shrink halves to 2× the
    # floor in one step, so anything within 1% of the floor
    # means the contractivity detector pinned the relax there.
    if (_is_finite(step.mech_relax_min_seen)
            and step.mech_relax_min_seen <= step.mech_relax_min * 1.01):
        return "picard_relax_floor"

    # 4 — rejection-reason tag dispatch.
    if rr.startswith("reject_at_anchor_"):
        return "reject_at_anchor"
    # ``RejectionReason`` enum string values (see guards.py):
    #   mechanical_delta_eps_inner / _step / mechanical_eps_max_wall
    if rr.startswith("mechanical_"):
        return "mechanical_guard"
    if rr.startswith("physical_"):
        return "physical_guard"

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
