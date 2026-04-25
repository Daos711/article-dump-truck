"""Stage THD-0: pipeline-side thermal coupling for the quasistatic diesel runner.

Provides:
  * ``ThermalConfig`` — dataclass holding mixing/convergence parameters.
  * ``build_oil_walther`` — Walther fit from existing oil dict
    (``eta_pump`` at 50 °C, ``eta_diesel`` at 105 °C, ``rho``).
  * ``viscosity_at_T_C`` — Pa·s at a given crank-angle effective temperature.
  * ``global_static_step`` — wraps the solver's ``global_static_target_C``;
    one call performs a single static energy-balance update, no relax.

Solver functions are imported from ``reynolds_solver.thermal``. If the
solver package is not importable in the current environment (e.g. a CI
sandbox) the module still imports cleanly, but any call into the helpers
raises a ``RuntimeError`` describing which thermal API call is missing.
This keeps the rest of the pipeline (and unit tests not gated on solver
availability) usable.

THD outer loop is owned by ``models/diesel_quasistatic.run_diesel_analysis``
(per Stage THD-0 scope: this module only provides the small helpers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# Lazy import — keep module importable even when the solver package isn't
# installed. Pipeline code paths that actually need thermal will raise
# explicitly via ``_require_solver_thermal``.
try:
    from reynolds_solver.thermal import (  # type: ignore
        fit_walther_two_point as _fit_walther_two_point,
        mu_at_T_C as _mu_at_T_C,
        global_static_target_C as _global_static_target_C,
    )
    SOLVER_THERMAL_AVAILABLE = True
except ImportError:  # pragma: no cover — env-dependent
    _fit_walther_two_point = None
    _mu_at_T_C = None
    _global_static_target_C = None
    SOLVER_THERMAL_AVAILABLE = False


def _require_solver_thermal(symbol: str) -> None:
    if not SOLVER_THERMAL_AVAILABLE:
        raise RuntimeError(
            f"reynolds_solver.thermal.{symbol} is not available; install "
            f"the solver package or run in an environment with the THD "
            f"API."
        )


@dataclass(frozen=True)
class ThermalConfig:
    """Stage THD-0 / global_static thermal mixing config.

    ``mode == "off"`` preserves the legacy isothermal behaviour (no
    fixed-point iteration; viscosity = ``oil["eta_diesel"]``).
    ``mode == "global_static"`` runs a per-angle fixed-point iteration
    around ``global_static_target_C``, see Section 3 of the patch spec.
    """
    mode: str = "off"
    T_in_C: float = 105.0
    gamma_mix: float = 0.7
    cp_J_kgK: float = 2000.0
    mdot_floor_kg_s: float = 1e-4
    tol_T_C: float = 0.5
    max_outer: int = 5
    underrelax_T: float = 0.6

    def is_off(self) -> bool:
        return str(self.mode).lower() == "off"

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            mode=self.mode, T_in_C=self.T_in_C, gamma_mix=self.gamma_mix,
            cp_J_kgK=self.cp_J_kgK, mdot_floor_kg_s=self.mdot_floor_kg_s,
            tol_T_C=self.tol_T_C, max_outer=self.max_outer,
            underrelax_T=self.underrelax_T,
        )


# ─── Oil model from existing oil dictionaries (Section 2) ──────────

def build_oil_walther(oil: Dict[str, Any]) -> Any:
    """Fit Walther viscosity-temperature law from an existing oil dict.

    The reference points are:
      * 50 °C  ↔ ``oil["eta_pump"]``  (Pa·s)
      * 105 °C ↔ ``oil["eta_diesel"]`` (Pa·s)

    Density ``oil["rho"]`` is used to convert dynamic viscosity (Pa·s)
    to kinematic viscosity (cSt = mm²/s) for the solver fit:
        ν[cSt] = η[Pa·s] / ρ[kg/m³] * 1e6

    Returns
    -------
    fit
        Whatever object ``reynolds_solver.thermal.fit_walther_two_point``
        returns; treat it as opaque and pass it back to
        ``viscosity_at_T_C``.
    """
    _require_solver_thermal("fit_walther_two_point")
    rho = float(oil["rho"])
    nu_50_cSt = float(oil["eta_pump"]) / rho * 1e6
    nu_105_cSt = float(oil["eta_diesel"]) / rho * 1e6
    return _fit_walther_two_point(
        T1_C=50.0, nu1_cSt=nu_50_cSt,
        T2_C=105.0, nu2_cSt=nu_105_cSt,
    )


def viscosity_at_T_C(walther_fit: Any, T_C: float) -> float:
    """Dynamic viscosity (Pa·s) at ``T_C`` from a fitted Walther model.

    Direct passthrough to ``reynolds_solver.thermal.mu_at_T_C``.
    """
    _require_solver_thermal("mu_at_T_C")
    return float(_mu_at_T_C(walther_fit, T_C))


# ─── Single global_static step (Section 3 step 7) ──────────────────

def global_static_step(
    *,
    T_in_C: float,
    P_loss_W: float,
    mdot_kg_s: float,
    cp_J_kgK: float,
    gamma: float,
) -> float:
    """One static energy-balance update.

    T_target = T_in + gamma * P_loss / (mdot * cp)

    Direct passthrough to ``reynolds_solver.thermal.global_static_target_C``;
    we keep the wrapper so the rest of the pipeline depends on a stable
    pipeline-side name and so the import is centralised in one place.
    """
    _require_solver_thermal("global_static_target_C")
    return float(_global_static_target_C(
        T_in_C=float(T_in_C),
        P_loss_W=float(P_loss_W),
        mdot_kg_s=float(mdot_kg_s),
        cp_J_kgK=float(cp_J_kgK),
        gamma=float(gamma),
    ))


__all__ = [
    "SOLVER_THERMAL_AVAILABLE",
    "ThermalConfig",
    "build_oil_walther",
    "viscosity_at_T_C",
    "global_static_step",
]
