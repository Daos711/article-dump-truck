"""Shared constants for v3 central-feed geometry-first branch."""
D = 0.054
R = D / 2.0
LD_RATIO = 0.40
L = D * LD_RATIO
c = 40e-6
n_rpm = 2000
eta = 0.018
sigma = 0.0
w_g = 0.004

EPS_REF = [0.2, 0.5, 0.8]
LOADCASE_NAMES = ["L20", "L50", "L80"]

GRID_MAIN = (1200, 400)
MAX_ITER_NR = 400
STEP_CAP = 0.05
EPS_MAX = 0.90
HS_WARMUP_ITER = 200_000
HS_WARMUP_TOL = 1e-5
