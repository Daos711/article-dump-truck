"""Contract tests for gu_loaded_side working geometry and schema."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cases.gu_loaded_side.common import (
    D, L, LD_RATIO, R, c, N_g, EPS_REF, LOADCASE_NAMES,
)
from cases.gu_loaded_side.schema import SCHEMA


def test_working_geometry_uses_ld_040():
    """LD_RATIO must be 0.40 and L == D * 0.40."""
    assert LD_RATIO == 0.40, f"LD_RATIO={LD_RATIO}, expected 0.40"
    assert abs(L - D * 0.40) < 1e-12, (
        f"L={L} != D*0.40={D * 0.40}")


def test_anchor_loadcases_consistent():
    """LOADCASE_NAMES has 3 entries matching EPS_REF."""
    assert len(LOADCASE_NAMES) == 3, (
        f"expected 3 load-case names, got {len(LOADCASE_NAMES)}")
    assert len(EPS_REF) == 3, (
        f"expected 3 EPS_REF entries, got {len(EPS_REF)}")
    assert len(LOADCASE_NAMES) == len(EPS_REF), (
        "LOADCASE_NAMES / EPS_REF length mismatch")


def test_schema_is_gu_loaded_side_v1():
    """Schema constant must be 'gu_loaded_side_v1'."""
    assert SCHEMA == "gu_loaded_side_v1", f"SCHEMA={SCHEMA!r}"
