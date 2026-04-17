"""Gu (Xiang) 2020 — Table 1 bearing + groove parameters.

Used for herringbone groove validation. These are DIMENSIONAL values
from the paper; nondimensional conversion is in texture_geometry.py.
"""

# ── Bearing ────────────────────────────────────────────────────────
D = 0.054                  # shaft diameter (m)
R = D / 2.0                # shaft radius (m)
L = 0.016                  # bush width (m)
c = 40e-6                  # radial clearance (m)
n = 2000                   # speed (rpm)
sigma = 0.0                # surface roughness (m) — not given in Gu

# ── Oil (approx from Gu context, mineral, ~50 °C) ─────────────────
eta = 0.018                # dynamic viscosity (Pa·s)

# ── Groove ─────────────────────────────────────────────────────────
w_g = 0.004                # groove width (m)
L_g = 0.012                # groove length (m)
d_g = 50e-6                # groove depth (m)
beta_deg = 30.0            # herringbone angle (deg)
N_g = 10                   # number of grooves

# ── Eccentricity sweep ─────────────────────────────────────────────
EPS_VALIDATION = [0.2, 0.5, 0.8]
EPS_DENSE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# ── Grids ──────────────────────────────────────────────────────────
GRIDS = {
    "coarse":  (800, 200),
    "confirm": (1200, 300),
    "fine":    (1600, 400),
}
