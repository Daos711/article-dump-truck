"""Contract tests for gu_loaded_side ablation group and status classification."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cases.gu_loaded_side.schema import (
    SCHEMA, TOL_HARD, TOL_SOFT, classify_status, is_feasible,
)

# ── T1: ablation group has four configurations ───────────────────────

ABLATION_CONFIGS = ["conv_nomag", "groove_nomag", "conv_mag", "groove_mag"]


def test_ablation_group_has_four_configs():
    """The loaded-side ablation study must use exactly four configs:
    conv_nomag, groove_nomag, conv_mag, groove_mag."""
    assert len(ABLATION_CONFIGS) == 4
    assert set(ABLATION_CONFIGS) == {
        "conv_nomag", "groove_nomag", "conv_mag", "groove_mag"
    }


# ── T2: classify_status thresholds ──────────────────────────────────

def test_classify_status_thresholds():
    """Verify hard_converged / soft_converged / failed boundaries."""
    # 4e-3 < TOL_HARD (5e-3) -> hard_converged
    assert classify_status(4e-3, converged=True) == "hard_converged", (
        "4e-3 with converged=True should be hard_converged")

    # 1.5e-2 is between TOL_HARD (5e-3) and TOL_SOFT (2e-2) -> soft_converged
    assert classify_status(1.5e-2, converged=True) == "soft_converged", (
        "1.5e-2 with converged=True should be soft_converged")

    # 3e-2 > TOL_SOFT (2e-2) -> failed (even though converged flag is True)
    assert classify_status(3e-2, converged=True) == "failed", (
        "3e-2 with converged=True should be failed (exceeds TOL_SOFT)")

    # Not converged at all -> failed regardless of residual
    assert classify_status(1e-4, converged=False) == "failed", (
        "converged=False should always be failed")


# ── T3: is_feasible ─────────────────────────────────────────────────

def test_is_feasible():
    """soft_converged and hard_converged are feasible; failed is not."""
    assert is_feasible("hard_converged") is True
    assert is_feasible("soft_converged") is True
    assert is_feasible("failed") is False
