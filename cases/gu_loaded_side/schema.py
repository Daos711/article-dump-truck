"""Schema constants + acceptance helpers for gu_loaded_side_v1."""
from __future__ import annotations

SCHEMA = "gu_loaded_side_v1"

TOL_HARD = 5e-3
TOL_SOFT = 5e-2


def classify_status(rel_residual: float, converged: bool) -> str:
    if not converged:
        return "failed"
    if rel_residual <= TOL_HARD:
        return "hard_converged"
    if rel_residual <= TOL_SOFT:
        return "soft_converged"
    return "failed"


def is_feasible(status: str) -> bool:
    return status in ("hard_converged", "soft_converged")
