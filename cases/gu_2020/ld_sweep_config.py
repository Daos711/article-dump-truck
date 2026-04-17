"""L/D sweep configuration (herringbone_ld_v1)."""
import math

SCHEMA = "herringbone_ld_v1"

# ── Base bearing (Gu 2020, D fixed) ──────────────────────────────
D = 0.054
R = D / 2.0
c = 40e-6
n_rpm = 2000
eta = 0.018
sigma = 0.0

# ── Groove same-mm (validated Gu design) ─────────────────────────
w_g = 0.004
L_g = 0.012
d_g = 50e-6
beta_deg = 30.0
N_g = 10

# ── Sweep ────────────────────────────────────────────────────────
LD_RATIOS = [0.30, 0.40, 0.50, 0.60]
EPS_LIST = [0.2, 0.5, 0.8]

# Anchor: exact Gu L=16 mm (L/D ≈ 0.2963)
ANCHOR_L_MM = 16.0
ANCHOR_LD_ACTUAL = ANCHOR_L_MM / (D * 1e3)

# ── Grid scaling ─────────────────────────────────────────────────
GRID_FAMILIES = {
    "coarse":  {"N_phi": 800,  "N_Z_base": 200,  "LD_base": 0.30},
    "confirm": {"N_phi": 1200, "N_Z_base": 300,  "LD_base": 0.30},
    "fine":    {"N_phi": 1600, "N_Z_base": 400,  "LD_base": 0.30},
}


def scale_NZ(N_Z_base: int, LD_base: float, LD_target: float) -> int:
    """Round to nearest multiple of 4."""
    raw = N_Z_base * LD_target / LD_base
    return max(4, int(round(raw / 4.0)) * 4)


def get_grid(grid_name: str, LD_target: float):
    fam = GRID_FAMILIES[grid_name]
    N_phi = fam["N_phi"]
    N_Z = scale_NZ(fam["N_Z_base"], fam["LD_base"], LD_target)
    return N_phi, N_Z
