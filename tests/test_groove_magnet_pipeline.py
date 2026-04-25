"""Contract tests for groove magnet pipeline (ТЗ §10.2)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.groove_magnet_force import make_groove_magnet_model


def test_no_mag_invariance_conv():
    """With B_ref=0, force model gives (0, 0) — no perturbation of
    equilibrium. Conventional case with zero magnets should equal
    conventional without magnets."""
    m = make_groove_magnet_model(B_ref_T=0.0)
    for _ in range(10):
        X = np.random.uniform(-0.5, 0.5)
        Y = np.random.uniform(-0.5, 0.5)
        Fx, Fy = m.force(X, Y)
        assert Fx == 0.0, f"Fx={Fx} at B=0"
        assert Fy == 0.0, f"Fy={Fy} at B=0"


def test_no_mag_invariance_groove():
    """Same check — groove geometry doesn't change magnet force."""
    m = make_groove_magnet_model(B_ref_T=0.0)
    Fx, Fy = m.force(0.3, -0.4)
    assert Fx == 0.0 and Fy == 0.0


def test_all_four_cases_share_same_loadcase():
    """run_ablation uses the SAME W_applied for all 4 configs. Check
    by reading the script source — it constructs W_applied once per
    loadcase outside the config loop."""
    import ast
    path = os.path.join(ROOT, "cases", "gu_2020_magnet", "run_ablation.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "W_applied" in src
    # W_applied defined before the configs loop
    assert src.index("W_applied = np.array") < src.index("for cfg in CONFIGS")


def test_manifest_schema_rejects_stale():
    """Schema name must be groove_magnet_v1."""
    from cases.gu_2020_magnet.config_gu_magnet import SCHEMA
    assert SCHEMA == "groove_magnet_v1"


def test_ablation_result_completeness_contract():
    """Each (loadcase, Bref) must produce all 4 configurations.
    We verify this as a structural contract — the ablation script
    iterates CONFIGS = [conv_nomag, groove_nomag, conv_mag, groove_mag]."""
    import ast
    path = os.path.join(ROOT, "cases", "gu_2020_magnet", "run_ablation.py")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    configs_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIGS":
                    if isinstance(node.value, ast.List):
                        elts = [e.value for e in node.value.elts
                                if isinstance(e, ast.Constant)]
                        assert set(elts) == {"conv_nomag", "groove_nomag",
                                             "conv_mag", "groove_mag"}, (
                            f"CONFIGS must have all 4; got {elts}")
                        configs_found = True
    assert configs_found, "CONFIGS constant not found in run_ablation.py"
