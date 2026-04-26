"""Stage Diesel Transient ClampAccounting Fix — diagnostics-only
contract tests.

Two cleanup items the patch addresses without touching physics:

1. The per-step clamp counter previously printed *event* counts
   (predictor + N_SUB substeps + final = up to ~3 events per step)
   labelled as "шагов" (steps). The patch keeps the existing
   ``contact_clamp`` step-mask semantics and adds a separate
   ``contact_clamp_event_count`` int array surfaced in the
   ``run_transient`` results dict.

2. ``classify_global_status`` keyed only on per-config envelope
   ``status``, missing the case where a config completed all steps
   but failed the applicability gate (e.g. valid_no_clamp_frac <
   0.85). The patch keys on ``applicable`` so the global header
   agrees with the per-config envelope block. ``per_config_status_line``
   gains a ``full / outside applicability gate`` category.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pytest

import models.diesel_transient as dt
from models.diesel_transient import CONFIGS, EnvelopeAbortConfig
from scripts.run_diesel_thd_transient import (
    classify_global_status,
    per_config_status_line,
    _build_envelope_records_from_results,
)


# ─── 1. classify_global_status: applicable, not status ─────────────

def test_classify_global_status_uses_applicable_not_status():
    """Two configs both ran to completion (status="ok") but one
    failed the applicability gate — the global header must downgrade
    to ``partial_production_result``, not stay at ``production_result``.
    """
    recs = [
        {"status": "ok", "applicable": True},
        {"status": "ok", "applicable": False},
    ]
    assert classify_global_status(recs) == "partial_production_result"


def test_classify_global_status_all_applicable_is_production():
    """All configs applicable — production_result."""
    recs = [
        {"status": "ok", "applicable": True},
        {"status": "ok", "applicable": True},
    ]
    assert classify_global_status(recs) == "production_result"


def test_classify_global_status_all_aborted_is_aborted():
    """All configs aborted — aborted_outside_envelope (independent of
    any leftover ``applicable`` field — aborted always means not
    applicable in the global header)."""
    recs = [
        {"status": "aborted_outside_envelope", "applicable": False},
        {"status": "aborted_outside_envelope", "applicable": False},
    ]
    assert classify_global_status(recs) == "aborted_outside_envelope"


# ─── 2. per_config_status_line: outside applicability gate ─────────

def test_per_config_status_line_outside_gate():
    """status="ok" + applicable=False — completed but envelope gate
    not met — must report the new ``outside applicability gate``
    category (do NOT silently mark as ``near-edge``)."""
    rec = {"status": "ok", "applicable": False}
    # Even with frac=1.0 the applicability gate veto wins.
    assert (per_config_status_line(rec, 1.0)
            == "full / outside applicability gate")
    # And with a lower frac (still completed) — same answer.
    assert (per_config_status_line(rec, 0.65)
            == "full / outside applicability gate")


def test_per_config_status_line_full_applicable():
    """status="ok" + applicable=True + frac>=0.95 — clean
    production case."""
    rec = {"status": "ok", "applicable": True}
    assert per_config_status_line(rec, 1.0) == "full / applicable"
    assert per_config_status_line(rec, 0.96) == "full / applicable"


def test_per_config_status_line_aborted_overrides():
    """status="aborted_outside_envelope" must override every other
    consideration — even applicable=True (defensive default) reads
    as aborted."""
    rec = {"status": "aborted_outside_envelope", "applicable": True}
    assert (per_config_status_line(rec, 1.0)
            == "aborted_outside_envelope")
    rec2 = {"status": "aborted_outside_envelope"}  # legacy shape
    assert (per_config_status_line(rec2, 0.50)
            == "aborted_outside_envelope")


# ─── 3. clamp event count vs step-mask ─────────────────────────────

def _stub_solve_reynolds_ok():
    """Successful solver returning a small finite non-dim P so the
    runner exercises the real ε / clamp accounting path."""
    def fake(H, d_phi, d_Z, R, L, **kw):
        P = np.full(H.shape, 1.0)
        return (P, 1e-6, 100, True)
    return fake


def _run_short(*, force_clamp: bool):
    """Run a tiny transient with the real ``run_transient`` but a
    mocked solver. When ``force_clamp`` is True, monkey-patch the
    ``_clamp`` helper so the runner sees a clamp event on the
    predictor + every substep + the final step (max events). When
    False, the helper is the identity (no clamp ever fires)."""
    fake = _stub_solve_reynolds_ok()
    target_solve = dt.solve_reynolds
    target_clamp_fn = None
    monkey_clamp_calls = {"n": 0}
    try:
        dt.solve_reynolds = fake

        # We can't directly substitute the closure ``_clamp`` (it's
        # built inside ``run_transient``). Instead, manipulate
        # ``params.eps_max``: dropping it below the initial ε forces
        # every clamp call to trigger; raising it well above any
        # achievable ε keeps the clamp inactive.
        from config import diesel_params as p
        original_eps_max = float(p.eps_max)
        if force_clamp:
            p.eps_max = 1e-6   # any nonzero ε triggers clamp
        else:
            p.eps_max = 1e12   # never reached even with mocked solver
        try:
            return dt.run_transient(
                F_max=200_000.0,
                configs=[CONFIGS[0]],   # smooth + mineral, fastest
                n_grid=20,
                n_cycles=1,
                d_phi_base_deg=20.0,
                d_phi_peak_deg=10.0,
                envelope_abort=EnvelopeAbortConfig.disabled(),
            )
        finally:
            p.eps_max = original_eps_max
    finally:
        dt.solve_reynolds = target_solve


def test_clamp_event_count_separate_from_step_mask():
    """When ε_max is below the initial ε, every step must trigger
    multiple clamp events (predictor + substep(s) + final), so the
    per-step event count exceeds the boolean step-mask."""
    res = _run_short(force_clamp=True)
    cc = np.asarray(res["contact_clamp"])         # bool step-mask
    ev = np.asarray(res["contact_clamp_event_count"])  # int per-step
    assert cc.shape == ev.shape
    assert ev.dtype.kind in ("i", "u")
    # Every step must be flagged in the step-mask.
    assert bool(cc.all())
    # And at least one step must carry strictly more than one clamp
    # event (proving events vs step-mask is a real distinction).
    assert int(ev.max()) > 1
    # Sum of events must equal or exceed sum of step-mask trues.
    assert int(ev.sum()) >= int(cc.sum())


def test_clamp_event_count_zero_when_no_clamp():
    """With ε_max well above any reachable ε, no clamp ever fires —
    the event count must be all zeros and the step-mask all False."""
    res = _run_short(force_clamp=False)
    cc = np.asarray(res["contact_clamp"])
    ev = np.asarray(res["contact_clamp_event_count"])
    assert ev.dtype.kind in ("i", "u")
    assert int(ev.sum()) == 0
    assert int(cc.sum()) == 0


def test_results_contains_clamp_event_count_field():
    """Public-API contract: the ``run_transient`` results dict must
    surface ``contact_clamp_event_count`` with shape (N_configs,
    N_steps) and an integer dtype."""
    res = _run_short(force_clamp=False)
    assert "contact_clamp_event_count" in res
    ev = np.asarray(res["contact_clamp_event_count"])
    cc = np.asarray(res["contact_clamp"])
    assert ev.shape == cc.shape
    assert ev.ndim == 2
    assert ev.shape[0] == len(res["configs"])
    assert ev.dtype.kind in ("i", "u")


# ─── 4. Envelope-records adapter surfaces ``applicable`` ───────────

def test_build_envelope_records_carries_applicable():
    """Sanity check that the adapter passes ``applicable`` straight
    through so ``classify_global_status`` keys on the right field.
    A run-results dict where one config is applicable=False (but
    not aborted) must produce ``partial_production_result``."""
    fake_results = {
        "configs": [
            {"label": "smooth"}, {"label": "textured"},
        ],
        "aborted": np.array([False, False]),
        "applicable": np.array([True, False]),
    }
    recs = _build_envelope_records_from_results(fake_results)
    assert recs[0]["applicable"] is True
    assert recs[1]["applicable"] is False
    assert classify_global_status(recs) == "partial_production_result"
