"""Shared working geometry + groove params for gu_loaded_side_v1.

All stages import from here — single source of truth for D, L, c, etc.
"""
from __future__ import annotations

import math
from typing import Dict, Any

# ── Working geometry (frozen from L/D sweep) ────────────────────
D = 0.054          # shaft diameter (m)
R = D / 2.0
LD_RATIO = 0.40
L = D * LD_RATIO   # = 21.6 mm
c = 40e-6          # radial clearance (m)
n_rpm = 2000
eta = 0.018        # Pa·s
sigma = 0.0        # surface roughness (m)

# ── Groove geometry (Gu validated, same mm) ─────────────────────
w_g = 0.004        # groove width (m)
L_g = 0.012        # groove length (m)
d_g = 50e-6        # groove depth (m)
beta_deg = 30.0
N_g = 10

# ── Magnet defaults ────────────────────────────────────────────
A_MAG_M2 = 4e-3 * 6e-3   # 24 mm²
N_MAG_EXP = 3.0
G_REG_M = 10e-6
T_COVER_M = 0.0
D_SUB_M = d_g             # subsurface depth = groove depth

# ── Load cases ──────────────────────────────────────────────────
EPS_REF = [0.2, 0.5, 0.8]
LOADCASE_NAMES = ["L20", "L50", "L80"]

# ── Solver ──────────────────────────────────────────────────────
GRID_CONFIRM = (1200, 400)
GRID_COARSE = (800, 268)

MAX_ITER_NR = 400
STEP_CAP = 0.05
EPS_MAX = 0.90

# ── B_ref sweep (Stages C/D) ───────────────────────────────────
BREF_SWEEP = [0.3, 0.5, 0.8, 1.0]

# ── Partial groove defaults (Stage B) ──────────────────────────
N_ACTIVE_LIST = [3, 5, 7]
SHIFT_CELLS_LIST = [-1, 0, 1]


def working_geometry_dict() -> Dict[str, Any]:
    return dict(D_mm=D * 1e3, L_mm=L * 1e3, LD_ratio=LD_RATIO,
                c_um=c * 1e6, n_rpm=n_rpm, eta_Pa_s=eta,
                sigma_um=sigma * 1e6)


def groove_geometry_dict() -> Dict[str, Any]:
    return dict(w_g_mm=w_g * 1e3, L_g_mm=L_g * 1e3,
                d_g_um=d_g * 1e6, beta_deg=beta_deg, N_g=N_g)
