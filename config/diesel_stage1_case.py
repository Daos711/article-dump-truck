"""Stage I — synthetic main bearing case for quasi-static diesel cycle.

Fallback synthetic case study. NOT a real Cummins KTA19.
"""
import math
import numpy as np

# ── Bearing ─────────────────────────────────────────────────────
D = 0.110          # shaft diameter (m)
R = D / 2.0
L = 0.055          # bush width (m), L/D = 0.50
c = 70e-6          # radial clearance (m)
n_rpm = 2000       # crankshaft speed

# ── Oil (mineral, Tin=105°C) ───────────────────────────────────
OIL_NAME = "mineral_105C"
T_IN_C = 105.0
ETA_COLD = 0.010    # Pa·s at 105°C (iso-viscous for I-A)
ALPHA_PV = None     # no PV for I-A

# ── Feed ───────────────────────────────────────────────────────
FEED_VARIANT = "point_unloaded"
FEED_PHI_HALF_DEG = 5.0
FEED_Z_BELT_HALF = 0.15
P_SUPPLY_PA = 2e5   # 2 bar

# ── Texture ────────────────────────────────────────────────────
TEX_VARIANT = "half_herringbone_ramped"
TEX_CHIRALITY = "pump_to_edge"
TEX_N_BRANCH = 10
TEX_BETA_DEG = 45.0
TEX_DG_M = 15e-6
TEX_TAPER = 0.6
TEX_BELT_HALF = 0.15
TEX_RAMP_FRAC = 0.15

# ── Grid ───────────────────────────────────────────────────────
GRID_COARSE = (400, 120)
GRID_CONFIRM = (800, 240)

# ── Solver ─────────────────────────────────────────────────────
MAX_ITER_NR = 400
STEP_CAP = 0.05
EPS_MAX = 0.92
TOL_HARD = 5e-3
TOL_SOFT = 2e-2

# ── Cycle ──────────────────────────────────────────────────────
CYCLE_N_COARSE = 36
CYCLE_N_FINE = 72

# ── Placement ─────────────────────────────────────────────────
PLACEMENT_MODE = "phase90_seed"
PHI_REF_DEG = 90.0   # reference crank angle for placement
