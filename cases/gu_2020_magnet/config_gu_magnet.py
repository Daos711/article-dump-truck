"""Configuration for groove+magnet ablation on Gu 2020 geometry."""
from cases.gu_2020.config_gu import D, R, L, c, n, eta, sigma
from cases.gu_2020.config_gu import w_g, L_g, d_g, beta_deg, N_g

# Magnet
A_MAG_M2 = 4e-3 * 6e-3        # 4 mm × 6 mm = 24 mm²
N_MAG = 3.0                    # force exponent
G_REG_M = 10e-6                # gap regularizer (m)
T_COVER_M = 0.0                # cover thickness (m), optimistic

BREF_SWEEP_T = [0.10, 0.20, 0.30, 0.50]

# Load cases: from conventional aligned at ε = 0.2, 0.5, 0.8
LOAD_CASE_EPS = [0.2, 0.5, 0.8]
LOAD_CASE_NAMES = ["L20", "L50", "L80"]

# Solver
TOL_ACCEPT = 5e-3
STEP_CAP = 0.10
EPS_MAX = 0.90
MAX_ITER_NR = 120

GRID = (800, 200)

SCHEMA = "groove_magnet_v1"
