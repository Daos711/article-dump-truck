"""Shared constants for v2 central-feed pipeline."""
from __future__ import annotations
from typing import Dict, Any

D = 0.054
R = D / 2.0
LD_RATIO = 0.40
L = D * LD_RATIO
c = 40e-6
n_rpm = 2000
eta = 0.018
sigma = 0.0

w_g = 0.004
N_BRANCH_DEFAULT = 3
BELT_HALF_FRACS = [0.10, 0.15, 0.20]
D_G_LEVELS = [10e-6, 25e-6]
D_G_DIAGNOSTIC = 50e-6

P_SUPPLY_BAR = [0, 2, 5]
ALPHA_PV_TARGET = 18e-9  # Barus coefficient from oil_properties

EPS_REF = [0.2, 0.5, 0.8]
LOADCASE_NAMES = ["L20", "L50", "L80"]

GRID_CONFIRM = (1200, 400)
MAX_ITER_NR = 400
STEP_CAP = 0.05
EPS_MAX = 0.90

HS_WARMUP_ITER = 200_000
HS_WARMUP_TOL = 1e-5


def ps_solve(ps_fn, H, d_phi, d_Z, R_val, L_val):
    return ps_fn(H, d_phi, d_Z, R_val, L_val,
                 tol=1e-6, max_iter=10_000_000,
                 hs_warmup_iter=HS_WARMUP_ITER,
                 hs_warmup_tol=HS_WARMUP_TOL)
