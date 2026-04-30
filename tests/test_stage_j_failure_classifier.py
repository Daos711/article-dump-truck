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
    failure_kind: str = "",
    nonfinite_count: int = 0,
    force_nonfinite: bool = False,
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
        failure_kind=failure_kind,
        nonfinite_count=nonfinite_count,
        force_nonfinite=force_nonfinite,
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


def test_classifier_picard_noncontractive_via_reason_string():
    """The kernel emits ``damped_picard_not_converged after ...``;
    that prefix dispatches to ``picard_noncontractive`` even when
    the structural counters look fine."""
    s = _make_step(
        rejection_reason=(
            "damped_picard_not_converged after 64/64 trials; "
            "min_relax=0.06250; last_delta_eps=2.345e-3"),
        fp_converged=False,
        n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.0625,
    )
    assert classify_failure(s) == "picard_noncontractive"


def test_classifier_picard_noncontractive_structural_fallback():
    """If the rejection_reason text is empty (e.g. partial-result
    rows from before the runner wrote it), the classifier still
    picks ``picard_noncontractive`` from ``fp_converged=False`` AND
    ``n_trials >= max_mech_inner``."""
    s = _make_step(
        rejection_reason="",
        fp_converged=False,
        n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.5,  # not at floor
    )
    assert classify_failure(s) == "picard_noncontractive"


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


def test_explicit_picard_reason_wins_over_numerical_budget():
    """The kernel's emitted ``damped_picard_not_converged`` rejection
    reason is the canonical signal — even when numerical signals
    look like solver_budget. The kernel knows whether the budget
    exhaustion drove the Picard failure or the Picard failure was
    independent; we trust its tag."""
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=1e-3, ausas_tol=1e-4,
        ausas_converged=False,
        rejection_reason=(
            "damped_picard_not_converged after 64/64 trials; "
            "min_relax=0.0156; last_delta_eps=1e-2"),
        fp_converged=False, n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.015625,
        mech_relax_min=0.015625,
    )
    assert classify_failure(s) == "picard_noncontractive"


def test_numerical_solver_budget_wins_over_picard_when_no_reason():
    """Without an explicit rejection_reason, numerical signals
    drive the classification. Budget exhaustion (n_inner == max
    AND residual > tol) wins over Picard's fp_converged=False
    via fallback rule order."""
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=1e-3, ausas_tol=1e-4,
        ausas_converged=False,
        rejection_reason="",  # no explicit tag
        fp_converged=False, n_trials=64, max_mech_inner=64,
        mech_relax_min_seen=0.015625,
        mech_relax_min=0.015625,
    )
    assert classify_failure(s) == "solver_budget"


def test_nonfinite_residual_wins_over_solver_budget_reason():
    """Task 27 — when the rejection reason says ``solver_n_inner_at_max``
    (budget exhausted) BUT ``ausas_residual`` is NaN, the underlying
    state IS non-finite — the budget exhaustion is a symptom, not
    the root cause. Classifier must report ``solver_nonfinite`` so
    the operator looks for the real problem (NaN in P/theta/state)
    rather than just bumping ``--ausas-max-inner``.

    This is the contract that the B'-result diagnose under-reported
    after Task 14 fixup-2 (which routed everything with
    ``solver_n_inner_at_max`` tag straight to ``solver_budget``,
    hiding the NaN-residual cases).
    """
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=float("nan"),     # nonfinite — the key signal
        ausas_tol=1e-4,
        ausas_converged=False,
        rejection_reason="solver_n_inner_at_max",
        p_max=float("nan"),
        theta_max=float("nan"),
    )
    assert classify_failure(s) == "solver_nonfinite"


def test_finite_solver_budget_with_finite_residual_routes_to_budget():
    """Mirror image — when residual IS finite (just above tol) AND
    ``n_inner == max``, that's a genuine budget exhaustion: the
    solver did real work, just couldn't reach tol. ``solver_budget``
    wins. Pmax / theta_max stay finite by default so the
    nonfinite-first rule doesn't fire."""
    s = _make_step(
        ausas_n_inner=5000, ausas_max_inner=5000,
        ausas_residual=1e-3,             # finite, > tol
        ausas_tol=1e-4,
        ausas_converged=False,
        rejection_reason="solver_n_inner_at_max",
    )
    assert classify_failure(s) == "solver_budget"


# ─── Task 27 new signals (gpu-reynolds Task 12 dict-API) ───────────


def test_classifier_nonfinite_via_failure_kind_nonfinite_state():
    """``failure_kind='nonfinite_state'`` from gpu-reynolds is the
    most specific signal — bucket immediately, even if everything
    else looks healthy. The solver explicitly told us the state
    went pathological."""
    s = _make_step(
        failure_kind="nonfinite_state",
        # All other fields finite — only failure_kind drives this.
        ausas_residual=1e-7, ausas_converged=True,
        p_max=1.0, theta_max=1.0,
        rejection_reason="",
    )
    assert classify_failure(s) == "solver_nonfinite"


def test_classifier_nonfinite_via_failure_kind_invalid_input():
    """``failure_kind='invalid_input'`` likewise — the solver
    refused to run because its input was already broken."""
    s = _make_step(failure_kind="invalid_input")
    assert classify_failure(s) == "solver_nonfinite"


def test_classifier_nonfinite_via_nonfinite_count():
    """``nonfinite_count > 0`` — solver-side hard-stop reported
    a non-finite cell count. Routes to nonfinite even without
    explicit failure_kind."""
    s = _make_step(nonfinite_count=12, failure_kind="")
    assert classify_failure(s) == "solver_nonfinite"


def test_classifier_nonfinite_via_residual_nan():
    """The B'-bug regression. ``ausas_residual=NaN`` triggers
    nonfinite EVEN WHEN ``rejection_reason`` says budget
    or residual."""
    for reason in (
        "solver_n_inner_at_max",
        "solver_residual_above_tol",
        "",
    ):
        s = _make_step(
            ausas_residual=float("nan"),
            rejection_reason=reason,
        )
        assert classify_failure(s) == "solver_nonfinite", reason


def test_classifier_nonfinite_via_force_nonfinite():
    """Runner-detected NaN F_hyd despite Ausas converged=True
    (Task 32 commit-semantics surface this case via the
    ``force_nonfinite`` flag — solver returned finite P but the
    integrated force is non-finite, e.g. from a pathological
    grid integration)."""
    s = _make_step(
        force_nonfinite=True,
        # Solver-side healthy.
        ausas_residual=1e-7, ausas_converged=True,
        p_max=1.0, theta_max=1.0,
    )
    assert classify_failure(s) == "solver_nonfinite"


def test_numerical_nonfinite_fallback_when_no_reason(tmp_path=None):
    """If pmax/theta_max non-finite AND no explicit reason, fall
    back to ``solver_nonfinite`` (best-effort signal for archived
    runs)."""
    s = _make_step(
        rejection_reason="",
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
        # 2 picard_noncontractive
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
    assert bc.per_bucket["picard_noncontractive"] == 2
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
    assert "picard_noncontractive" in FAILURE_BUCKETS
    assert "picard_relax_floor" in FAILURE_BUCKETS
    assert "reject_at_anchor" in FAILURE_BUCKETS
    assert "mechanical_guard" in FAILURE_BUCKETS
    assert "physical_guard" in FAILURE_BUCKETS
    assert "unknown" in FAILURE_BUCKETS
    # No silent duplicates.
    assert len(set(FAILURE_BUCKETS)) == len(FAILURE_BUCKETS)
