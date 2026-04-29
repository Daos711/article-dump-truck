"""Stage J fu-2 Task 14 — failure classifier contract.

Synthetic ``StepDiagnosticRow`` per bucket; one round-trip test
through ``aggregate_buckets`` to pin the totals / dominant-bucket
selection. The classifier is pure (no GPU, no I/O) so these tests
are fast and don't need fixture data.
"""
from __future__ import annotations

import math

import pytest

from models.diesel_coupling import (
    BucketCounts,
    FAILURE_BUCKETS,
    StepDiagnosticRow,
    aggregate_buckets,
    classify_failure,
)


# ─── Builder ───────────────────────────────────────────────────────


def _make_step(
    *,
    is_failure: bool = True,
    rejection_reason: str = "",
    ausas_n_inner: int = 100,
    ausas_residual: float = 1e-7,
    ausas_converged: bool = True,
    p_max: float = 1.0,
    theta_max: float = 1.0,
    fp_converged: bool = True,
    n_trials: int = 1,
    mech_relax_min_seen: float = 0.25,
    ausas_tol: float = 1e-4,
    ausas_max_inner: int = 5000,
    max_mech_inner: int = 64,
    mech_relax_min: float = 0.015625,
) -> StepDiagnosticRow:
    """Defaults describe a healthy converged step. Tests override
    only the fields relevant to the bucket they exercise."""
    return StepDiagnosticRow(
        is_failure=is_failure,
        rejection_reason=rejection_reason,
        ausas_n_inner=ausas_n_inner,
        ausas_residual=ausas_residual,
        ausas_converged=ausas_converged,
        p_max=p_max,
        theta_max=theta_max,
        fp_converged=fp_converged,
        n_trials=n_trials,
        mech_relax_min_seen=mech_relax_min_seen,
        ausas_tol=ausas_tol,
        ausas_max_inner=ausas_max_inner,
        max_mech_inner=max_mech_inner,
        mech_relax_min=mech_relax_min,
    )


# ─── Per-bucket tests ──────────────────────────────────────────────


def test_classifier_ok():
    """Healthy converged step is ``ok`` regardless of other fields."""
    s = _make_step(is_failure=False)
    assert classify_failure(s) == "ok"


def test_classifier_solver_budget():
    """Ausas hit ``max_inner`` AND residual > tol AND not converged."""
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=1e-3, ausas_tol=1e-4,
        ausas_converged=False,
    )
    assert classify_failure(s) == "solver_budget"


def test_classifier_solver_residual():
    """Residual > tol and not converged, but ``n_inner`` short of
    the budget — solver stalled / diverged before exhausting time."""
    s = _make_step(
        ausas_n_inner=2500, ausas_max_inner=5000,
        ausas_residual=5e-3, ausas_tol=1e-4,
        ausas_converged=False,
    )
    assert classify_failure(s) == "solver_residual"


def test_classifier_solver_nonfinite_pressure():
    """Non-finite ``p_max`` short-circuits to ``solver_nonfinite``."""
    s = _make_step(p_max=float("nan"))
    assert classify_failure(s) == "solver_nonfinite"


def test_classifier_solver_nonfinite_theta():
    """Non-finite ``theta_max`` also lands here."""
    s = _make_step(theta_max=float("inf"))
    assert classify_failure(s) == "solver_nonfinite"


def test_classifier_picard_not_converged_via_reason_string():
    """The kernel emits ``damped_picard_not_converged after ...``;
    that prefix dispatches to ``picard_not_converged`` even when
    the structural counters look fine."""
    s = _make_step(
        rejection_reason=(
            "damped_picard_not_converged after 64/64 trials; "
            "min_relax=0.06250; last_delta_eps=2.345e-3"),
        fp_converged=False,
        n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.0625,
    )
    assert classify_failure(s) == "picard_not_converged"


def test_classifier_picard_not_converged_structural_fallback():
    """If the rejection_reason text is empty (e.g. partial-result
    rows from before the runner wrote it), the classifier still
    picks ``picard_not_converged`` from ``fp_converged=False`` AND
    ``n_trials >= max_mech_inner``."""
    s = _make_step(
        rejection_reason="",
        fp_converged=False,
        n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.5,  # not at floor
    )
    assert classify_failure(s) == "picard_not_converged"


def test_classifier_picard_relax_floor():
    """Picard sat at the contractivity-detector floor — relax
    ground down to ``mech_relax_min`` (within 1%)."""
    s = _make_step(
        fp_converged=False,
        n_trials=24, max_mech_inner=64,
        mech_relax_min_seen=0.015625,
        mech_relax_min=0.015625,
    )
    assert classify_failure(s) == "picard_relax_floor"


def test_classifier_reject_at_anchor():
    """``reject_at_anchor_<bucket>`` rejection-reason tag → its own
    bucket (Step-9 fixup-1 fast-abort path)."""
    s = _make_step(
        rejection_reason=(
            "reject_at_anchor_solver_residual: residual too large"),
        fp_converged=False,
        n_trials=1, max_mech_inner=64,
    )
    assert classify_failure(s) == "reject_at_anchor"


def test_classifier_mechanical_guard():
    """Mechanical-candidate guard reasons (``mechanical_*``) bucket
    to ``mechanical_guard``."""
    for reason in (
        "mechanical_delta_eps_inner",
        "mechanical_delta_eps_step",
        "mechanical_eps_max_wall",
    ):
        s = _make_step(
            rejection_reason=reason,
            fp_converged=False,
            n_trials=8, max_mech_inner=64,
        )
        assert classify_failure(s) == "mechanical_guard", reason


def test_classifier_physical_guard():
    """Physical-guard reasons (``physical_*``) bucket to
    ``physical_guard``."""
    for reason in (
        "physical_pressure_above_dim_max",
        "physical_theta_out_of_range",
        "physical_force_balance_sign_flip",
    ):
        s = _make_step(
            rejection_reason=reason,
            fp_converged=False,
            n_trials=8, max_mech_inner=64,
        )
        assert classify_failure(s) == "physical_guard", reason


def test_classifier_unknown_sentinel():
    """A failure with no recognisable signature goes to ``unknown``
    — the rule-set should be extended, not the data hidden."""
    s = _make_step(
        is_failure=True,
        rejection_reason="some_new_failure_we_haven't_seen_yet",
        # Solver-side looks healthy (converged=True, residual<tol,
        # finite outputs), so no solver_* bucket fires.
        ausas_converged=True, ausas_residual=1e-7,
        # Picard-side also looks healthy: fp_converged=True; relax
        # nowhere near the floor.
        fp_converged=True, n_trials=4, max_mech_inner=64,
        mech_relax_min_seen=0.25,
    )
    assert classify_failure(s) == "unknown"


# ─── Priority / tie-break tests ────────────────────────────────────


def test_solver_budget_wins_over_picard():
    """If both Ausas budget exhausted AND Picard fp_converged=False,
    the classifier reports the underlying solver issue rather than
    the symptom on the Picard side. That's what the rule order in
    the docstring promises (solver before Picard before tags)."""
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=1e-3, ausas_tol=1e-4,
        ausas_converged=False,
        # Picard side ALSO looks bad — but the solver issue is
        # the root cause.
        rejection_reason=(
            "damped_picard_not_converged after 64/64 trials; "
            "min_relax=0.0156; last_delta_eps=1e-2"),
        fp_converged=False, n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.015625,
        mech_relax_min=0.015625,
    )
    assert classify_failure(s) == "solver_budget"


def test_solver_nonfinite_wins_over_solver_budget():
    """Non-finite outputs are reported even if budget was also
    exhausted — non-finite is more specific."""
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=1e-3, ausas_tol=1e-4,
        ausas_converged=False,
        p_max=float("nan"),
    )
    assert classify_failure(s) == "solver_nonfinite"


# ─── Aggregator ────────────────────────────────────────────────────


def test_aggregate_buckets_counts_and_dominant():
    """``aggregate_buckets`` runs the classifier over a sequence
    and returns counts, totals, and the dominant failure bucket."""
    rows = [
        # 3 ok
        _make_step(is_failure=False),
        _make_step(is_failure=False),
        _make_step(is_failure=False),
        # 5 solver_budget
        *[_make_step(
            ausas_n_inner=5000, ausas_max_inner=5000,
            ausas_residual=1e-3, ausas_tol=1e-4,
            ausas_converged=False,
        ) for _ in range(5)],
        # 2 picard_not_converged
        _make_step(
            rejection_reason="damped_picard_not_converged after 64/64",
            fp_converged=False, n_trials=64, max_mech_inner=64,
        ),
        _make_step(
            rejection_reason="damped_picard_not_converged after 64/64",
            fp_converged=False, n_trials=64, max_mech_inner=64,
        ),
    ]
    bc = aggregate_buckets(rows)
    assert isinstance(bc, BucketCounts)
    assert bc.n_total == 10
    assert bc.n_failures == 7
    assert bc.per_bucket["ok"] == 3
    assert bc.per_bucket["solver_budget"] == 5
    assert bc.per_bucket["picard_not_converged"] == 2
    assert bc.dominant_failure_bucket() == "solver_budget"
    # frac semantics — over total vs. over failures.
    assert bc.frac("solver_budget") == pytest.approx(0.5)
    assert bc.frac_of_failures("solver_budget") == pytest.approx(5.0 / 7.0)


def test_aggregate_no_failures_returns_none_dominant():
    rows = [_make_step(is_failure=False) for _ in range(4)]
    bc = aggregate_buckets(rows)
    assert bc.n_failures == 0
    assert bc.dominant_failure_bucket() is None
    assert bc.frac_of_failures("solver_budget") == 0.0


# ─── Public-API surface ────────────────────────────────────────────


def test_failure_buckets_listed():
    """``FAILURE_BUCKETS`` is the exhaustive enumeration. If a new
    bucket is added, this test highlights it for the documentation
    update."""
    assert "ok" in FAILURE_BUCKETS
    assert "solver_budget" in FAILURE_BUCKETS
    assert "solver_residual" in FAILURE_BUCKETS
    assert "solver_nonfinite" in FAILURE_BUCKETS
    assert "picard_not_converged" in FAILURE_BUCKETS
    assert "picard_relax_floor" in FAILURE_BUCKETS
    assert "reject_at_anchor" in FAILURE_BUCKETS
    assert "mechanical_guard" in FAILURE_BUCKETS
    assert "physical_guard" in FAILURE_BUCKETS
    assert "unknown" in FAILURE_BUCKETS
    # No silent duplicates.
    assert len(set(FAILURE_BUCKETS)) == len(FAILURE_BUCKETS)
