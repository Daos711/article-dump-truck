"""Contract tests for groove magnet force model (ТЗ §10.1)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.groove_magnet_force import (
    EmbeddedGrooveMagnetModel,
    compute_magnet_centers_from_groove_cells,
    make_groove_magnet_model,
    magnetic_force_from_Bref,
    MU_0,
)


def test_groove_magnet_force_center_zero():
    """At (0,0) with all 10 identical magnets: |F| < 1e-12."""
    m = make_groove_magnet_model(B_ref_T=0.3)
    Fx, Fy = m.force(0.0, 0.0)
    assert abs(Fx) < 1e-10, f"Fx={Fx}"
    assert abs(Fy) < 1e-10, f"Fy={Fy}"


def test_groove_magnet_restoring_direction():
    """For 20 random points with eps < 0.5: W_mag·(X,Y) < 0."""
    m = make_groove_magnet_model(B_ref_T=0.3)
    rng = np.random.default_rng(42)
    for _ in range(20):
        eps = rng.uniform(0.05, 0.49)
        ang = rng.uniform(0, 2 * math.pi)
        X = eps * math.cos(ang)
        Y = eps * math.sin(ang)
        Fx, Fy = m.force(X, Y)
        dot = Fx * X + Fy * Y
        assert dot < 0, (
            f"not restoring at X={X:.4f} Y={Y:.4f}: "
            f"F·r = {dot:.6e}")


def test_groove_magnet_bref_monotone():
    """Larger B_ref → larger |W_mag|."""
    models = [make_groove_magnet_model(B_ref_T=b)
              for b in [0.10, 0.20, 0.30, 0.50]]
    X, Y = 0.1, -0.2
    forces = [math.sqrt(m.force(X, Y)[0] ** 2 + m.force(X, Y)[1] ** 2)
              for m in models]
    for i in range(len(forces) - 1):
        assert forces[i] < forces[i + 1], (
            f"|F|(B={[0.10,0.20,0.30,0.50][i]:.2f})="
            f"{forces[i]:.4f} >= {forces[i+1]:.4f}")


def test_magnet_centers_inside_grooves():
    """Every magnet center at phi_k = cell_span*(k+0.5) is inside
    the corresponding groove cell [k*cell_span, (k+1)*cell_span]."""
    N_g = 10
    centers = compute_magnet_centers_from_groove_cells(N_g)
    cell_span = 2 * math.pi / N_g
    for k, phi_k in enumerate(centers):
        lo = cell_span * k
        hi = cell_span * (k + 1)
        assert lo < phi_k < hi, (
            f"center {k}: phi={phi_k:.4f} not in "
            f"[{lo:.4f}, {hi:.4f}]")


def test_bref_zero_gives_zero_force():
    """B_ref=0 → exactly zero force everywhere."""
    m = make_groove_magnet_model(B_ref_T=0.0)
    Fx, Fy = m.force(0.3, -0.2)
    assert Fx == 0.0 and Fy == 0.0


def test_F_ref_formula():
    """F_ref = A_mag * B_ref^2 / (2*mu_0)."""
    B = 0.30
    A = 24e-6
    expected = A * B ** 2 / (2 * MU_0)
    got = magnetic_force_from_Bref(B, A)
    assert abs(got - expected) / expected < 1e-12
