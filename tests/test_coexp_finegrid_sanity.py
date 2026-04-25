"""Contract tests for fine-grid sanity wrapper (coexp_v1.2).

Покрывают только pure logic — top-2 picker, decision_signal,
build_summary схема. Не запускают реальные solve.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_coexp_finegrid_sanity import (  # noqa: E402
    SCHEMA_FINEGRID,
    pick_top2_candidates,
    compute_decision_signal,
    build_summary,
    LOW_LOAD_WY,
)


# ── Fixtures ───────────────────────────────────────────────────────

def _diag_C(results_by_wy):
    return dict(
        schema_version="coexp_v1.1",
        diagnostic_id="C",
        results_by_wy=results_by_wy,
    )


def _pair(profile_hash, family, h_r=None, useful=False, **extras):
    base = dict(
        profile_hash=profile_hash, family=family,
        params={"A2": 0.05, "phi2_deg": 10.0},
        h_r=h_r, p_r=0.99, f_r=1.005, c_d=0.001,
        useful=bool(useful),
        smooth_accepted=True, textured_accepted=True,
        eps_smooth=0.30, eps_textured=0.29,
    )
    base.update(extras)
    return base


# ── T1: top-2 picker ───────────────────────────────────────────────

def test_top2_extraction_picks_best_per_wy():
    """Distinct top per Wy: должны попасть оба (без коллизии хэшей)."""
    diag = _diag_C({
        "0.050": [
            _pair("aaaa1111", "two_lobe", h_r=1.005),
            _pair("d6afa927", "mixed_harmonic", h_r=1.023, useful=True),
            _pair("bbbb2222", "local_bump", h_r=0.995),
        ],
        "0.100": [
            _pair("73d19cfd", "local_bump", h_r=1.014, useful=True),
            _pair("d6afa927", "mixed_harmonic", h_r=1.011),
            _pair("cccc3333", "two_lobe", h_r=1.002),
        ],
        "0.250": [
            _pair("zzzz9999", "two_lobe", h_r=0.99),
        ],
    })
    chosen = pick_top2_candidates(diag, low_load_wy=[0.05, 0.10])
    assert len(chosen) == 2
    # First slot: best @ Wy=0.05 → d6afa927 (h_r=1.023)
    assert chosen[0]["profile_hash"] == "d6afa927"
    assert chosen[0]["Wy_share"] == pytest.approx(0.05)
    assert chosen[0]["h_r_coarse"] == pytest.approx(1.023)
    assert chosen[0]["useful_coarse"] is True
    # Second slot: best @ Wy=0.10 → 73d19cfd (h_r=1.014)
    assert chosen[1]["profile_hash"] == "73d19cfd"
    assert chosen[1]["Wy_share"] == pytest.approx(0.10)


def test_top2_extraction_handles_hash_collision():
    """Если top@0.05 и top@0.10 — один и тот же hash, второй слот
    становится runner-up для Wy=0.10."""
    diag = _diag_C({
        "0.050": [
            _pair("d6afa927", "mixed_harmonic", h_r=1.025, useful=True),
            _pair("73d19cfd", "local_bump", h_r=1.015, useful=True),
        ],
        "0.100": [
            _pair("d6afa927", "mixed_harmonic", h_r=1.020, useful=True),
            _pair("73d19cfd", "local_bump", h_r=1.014, useful=True),
        ],
    })
    chosen = pick_top2_candidates(diag, low_load_wy=[0.05, 0.10])
    assert len(chosen) == 2
    assert chosen[0]["profile_hash"] == "d6afa927"
    assert chosen[0]["Wy_share"] == pytest.approx(0.05)
    # Slot 2 must skip d6afa927 (already chosen) → 73d19cfd
    assert chosen[1]["profile_hash"] == "73d19cfd"
    assert chosen[1]["Wy_share"] == pytest.approx(0.10)


def test_top2_extraction_skips_pairs_without_h_r():
    """Кандидаты с h_r=None должны игнорироваться."""
    diag = _diag_C({
        "0.050": [
            _pair("aaaa1111", "two_lobe", h_r=None),
            _pair("bbbb2222", "local_bump", h_r=1.010, useful=True),
        ],
        "0.100": [],
    })
    chosen = pick_top2_candidates(diag, low_load_wy=[0.05, 0.10])
    assert len(chosen) == 1
    assert chosen[0]["profile_hash"] == "bbbb2222"


# ── T2: decision_signal three branches ─────────────────────────────

def test_decision_signal_proceed():
    """ALL useful_fine=True AND ALL h_r_fine ≥ 1.015 → PROCEED_TO_V2."""
    cands = [
        dict(h_r_fine=1.025, useful_fine=True),
        dict(h_r_fine=1.018, useful_fine=True),
    ]
    assert compute_decision_signal(cands) == "PROCEED_TO_V2"


def test_decision_signal_investigate_nr():
    """ALL h_r_fine in [1.005, 1.015) → INVESTIGATE_NR."""
    cands = [
        dict(h_r_fine=1.010, useful_fine=True),
        dict(h_r_fine=1.008, useful_fine=True),
    ]
    assert compute_decision_signal(cands) == "INVESTIGATE_NR"


def test_decision_signal_investigate_when_useful_but_no_margin():
    """Even mixed: один useful с margin, другой useful без margin →
    INVESTIGATE (т.к. requires ALL >= 1.015)."""
    cands = [
        dict(h_r_fine=1.020, useful_fine=True),
        dict(h_r_fine=1.010, useful_fine=True),
    ]
    assert compute_decision_signal(cands) == "INVESTIGATE_NR"


def test_decision_signal_close_on_collapse():
    """Любой h_r_fine < 1.0 → CLOSE_COEXP (опрокинулся)."""
    cands = [
        dict(h_r_fine=1.020, useful_fine=True),
        dict(h_r_fine=0.985, useful_fine=False),
    ]
    assert compute_decision_signal(cands) == "CLOSE_COEXP"


def test_decision_signal_close_on_missing_h_r():
    """h_r_fine=None (например, NR не сошёлся для ratios) → CLOSE."""
    cands = [
        dict(h_r_fine=None, useful_fine=False),
        dict(h_r_fine=1.020, useful_fine=True),
    ]
    assert compute_decision_signal(cands) == "CLOSE_COEXP"


def test_decision_signal_close_on_below_useful_gate():
    """Если хоть один h_r ниже useful gate (<1.005) но >=1.0 — fall-
    through в CLOSE (mixed unhealthy)."""
    cands = [
        dict(h_r_fine=1.020, useful_fine=True),
        dict(h_r_fine=1.001, useful_fine=False),
    ]
    assert compute_decision_signal(cands) == "CLOSE_COEXP"


def test_decision_signal_empty():
    """Пустой список → CLOSE."""
    assert compute_decision_signal([]) == "CLOSE_COEXP"


# ── T3: summary schema ────────────────────────────────────────────

def test_finegrid_sanity_summary_schema():
    diag = _diag_C({})
    cf = [
        dict(profile_hash="d6afa927", family="mixed_harmonic",
             Wy_share=0.05,
             coarse_grid={"N_phi": 800, "N_Z": 200},
             fine_grid={"N_phi": 1600, "N_Z": 400},
             h_r_coarse=1.023, p_r_coarse=0.98,
             f_r_coarse=1.01, c_d_coarse=0.001,
             useful_coarse=True,
             h_r_fine=1.018, p_r_fine=0.99,
             f_r_fine=1.01, c_d_fine=0.001,
             useful_fine=True,
             smooth_accepted_fine=True, textured_accepted_fine=True,
             rel_residual_smooth_fine=4e-3,
             rel_residual_textured_fine=4e-3),
        dict(profile_hash="73d19cfd", family="local_bump",
             Wy_share=0.10,
             coarse_grid={"N_phi": 800, "N_Z": 200},
             fine_grid={"N_phi": 1600, "N_Z": 400},
             h_r_coarse=1.014, p_r_coarse=0.99,
             f_r_coarse=1.01, c_d_coarse=0.001,
             useful_coarse=True,
             h_r_fine=1.020, p_r_fine=0.99,
             f_r_fine=1.01, c_d_fine=0.001,
             useful_fine=True,
             smooth_accepted_fine=True, textured_accepted_fine=True,
             rel_residual_smooth_fine=3e-3,
             rel_residual_textured_fine=3e-3),
    ]
    doc = build_summary(diag, cf, base_run="coexp_2026_04_16",
                         from_diag="coexp_2026_04_16",
                         tol_accept=0.01, max_iter=200)
    # Required top-level keys
    assert doc["schema_version"] == SCHEMA_FINEGRID
    assert doc["schema_version"] == "coexp_v1.2"
    assert doc["diagnostic_id"] == "finegrid_sanity"
    assert doc["base_run"] == "coexp_2026_04_16"
    assert doc["from_diagnostic_C"] == "coexp_2026_04_16"
    assert doc["tol_accept_effective"] == pytest.approx(0.01)
    assert doc["max_iter_nr"] == 200
    assert "created_utc" in doc
    assert isinstance(doc["candidates_tested"], list)
    assert len(doc["candidates_tested"]) == 2
    # Required summary block
    s = doc["summary"]
    for k in ("n_candidates", "n_useful_fine",
              "n_useful_coarse_but_not_fine",
              "max_h_r_fine", "decision_signal"):
        assert k in s, f"missing summary.{k}"
    assert s["n_candidates"] == 2
    assert s["n_useful_fine"] == 2
    assert s["n_useful_coarse_but_not_fine"] == 0
    assert s["max_h_r_fine"] == pytest.approx(1.020)
    # Both >= 1.015 + useful → PROCEED
    assert s["decision_signal"] == "PROCEED_TO_V2"
    # JSON-roundtrip-safe (no datetime objects, no numpy scalars)
    blob = json.dumps(doc)
    again = json.loads(blob)
    assert again["schema_version"] == SCHEMA_FINEGRID


def test_summary_counts_coarse_only_useful():
    """useful_coarse=True но useful_fine=False → счётчик
    n_useful_coarse_but_not_fine."""
    diag = _diag_C({})
    cf = [
        dict(profile_hash="aaa", family="x", Wy_share=0.05,
             coarse_grid={}, fine_grid={},
             h_r_coarse=1.020, useful_coarse=True,
             h_r_fine=0.99, useful_fine=False),
        dict(profile_hash="bbb", family="y", Wy_share=0.10,
             coarse_grid={}, fine_grid={},
             h_r_coarse=1.015, useful_coarse=True,
             h_r_fine=1.020, useful_fine=True),
    ]
    doc = build_summary(diag, cf, base_run="r",
                         from_diag="r",
                         tol_accept=0.01, max_iter=200)
    assert doc["summary"]["n_useful_coarse_but_not_fine"] == 1
    # Один h_r_fine < 1.0 → CLOSE
    assert doc["summary"]["decision_signal"] == "CLOSE_COEXP"
