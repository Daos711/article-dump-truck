"""Schema for v4 geometry rescue gate."""
import os

SCHEMA = "gu_loaded_side_v4_geometry_rescue"
TOL_HARD = 5e-3
TOL_SOFT = 2e-2

PROTECTED_LO_DEG = 105.0
PROTECTED_HI_DEG = 175.0


def resolve_stage_dir(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".json"):
        return os.path.dirname(path)
    return path


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
