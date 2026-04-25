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

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional


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

def build_oil_walther(
    oil: Dict[str, Any],
    *,
    cp_J_kgK: Optional[float] = None,
    gamma_mix: Optional[float] = None,
) -> Any:
    """Fit Walther viscosity-temperature law from an existing oil dict.

    The reference points are:
      * 50 °C  ↔ ``oil["eta_pump"]``  (Pa·s)
      * 105 °C ↔ ``oil["eta_diesel"]`` (Pa·s)

    Density ``oil["rho"]`` is used to convert dynamic viscosity (Pa·s)
    to kinematic viscosity (cSt = mm²/s) for the solver fit:
        ν[cSt] = η[Pa·s] / ρ[kg/m³] * 1e6

    The solver's ``fit_walther_two_point`` returns an ``OilModel`` that
    *also* carries ``rho_kg_m3`` / ``cp_J_kgK`` / ``alpha_pv_base`` /
    ``gamma_mix`` for downstream use by ``mu_at_T_C`` and other thermal
    helpers. To make the round-trip
    ``mu_at_T_C(105 °C) == oil["eta_diesel"]`` exact, the actual oil
    density must be embedded in the OilModel — otherwise the solver
    falls back to its own default and the dynamic-viscosity conversion
    drifts. We therefore pass ``rho_kg_m3``, ``alpha_pv_base`` and
    optionally ``cp_J_kgK`` / ``gamma_mix`` (sourced from
    ``ThermalConfig`` when available) as kwargs. If the installed
    ``fit_walther_two_point`` doesn't accept these, we fall back to the
    bare two-point fit.

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

    base = dict(
        T1_C=50.0, nu1_cSt=nu_50_cSt,
        T2_C=105.0, nu2_cSt=nu_105_cSt,
    )
    extras: Dict[str, Any] = dict(rho_kg_m3=rho)
    alpha_pv = oil.get("alpha_pv")
    if alpha_pv is not None:
        extras["alpha_pv_base"] = float(alpha_pv)
    if cp_J_kgK is not None:
        extras["cp_J_kgK"] = float(cp_J_kgK)
    if gamma_mix is not None:
        extras["gamma_mix"] = float(gamma_mix)

    # Try the modern OilModel signature first; fall back kwarg-by-kwarg
    # if the installed solver is older / takes a narrower signature.
    fit: Any = None
    last_err: Optional[Exception] = None
    for trial in (
        {**base, **extras},
        {**base, "rho_kg_m3": rho},
        base,
    ):
        try:
            fit = _fit_walther_two_point(**trial)
            break
        except TypeError as e:
            last_err = e
            continue
    if fit is None:
        raise last_err  # type: ignore[misc]

    # The solver's ``fit_walther_two_point`` may silently drop kwargs
    # it doesn't accept and populate the returned OilModel with its
    # built-in defaults (e.g. rho_kg_m3=860). That makes the
    # mu_at_T_C(105) == eta_diesel round-trip drift by exactly
    # rho_default / oil["rho"]. Override the populated dataclass fields
    # so the OilModel matches the actual oil dictionary.
    overrides: Dict[str, Any] = {}
    for fld_name, value in (("rho_kg_m3", rho),
                              ("alpha_pv_base", alpha_pv),
                              ("cp_J_kgK", cp_J_kgK),
                              ("gamma_mix", gamma_mix)):
        if value is None:
            continue
        if dataclasses.is_dataclass(fit) and hasattr(fit, fld_name):
            current = getattr(fit, fld_name)
            if abs(float(current) - float(value)) > 1e-12:
                overrides[fld_name] = float(value)
    if overrides and dataclasses.is_dataclass(fit):
        try:
            fit = dataclasses.replace(fit, **overrides)
        except (TypeError, ValueError):
            # Frozen-replace failed (e.g. unknown field) — leave as-is.
            pass
    return fit


def viscosity_at_T_C(walther_fit: Any, T_C: float) -> float:
    """Dynamic viscosity (Pa·s) at ``T_C`` from a fitted Walther model.

    Argument order in the solver's ``mu_at_T_C`` is ``(T_C, model)`` —
    we keep the pipeline-side wrapper as ``(walther_fit, T_C)`` so the
    rest of the pipeline reads naturally and centralises the bridging.
    """
    _require_solver_thermal("mu_at_T_C")
    try:
        return float(_mu_at_T_C(T_C, walther_fit))
    except TypeError:
        # Some older versions of the solver may swap the order; try the
        # alternative before giving up.
        return float(_mu_at_T_C(walther_fit, T_C))


# ─── Single global_static step (Section 3 step 7) ──────────────────

def global_static_step(
    *,
    T_in_C: float,
    P_loss_W: float,
    mdot_kg_s: float,
    cp_J_kgK: float,
    gamma: float,
    model: Any = None,
) -> float:
    """One static energy-balance update.

    T_target = T_in + gamma * P_loss / (mdot * cp)

    Wraps ``reynolds_solver.thermal.global_static_target_C``. The solver
    may take either an explicit ``cp_J_kgK`` / ``gamma`` pair or an
    ``OilModel`` (which already carries those scalars). We try the
    explicit-args form first (matches the patch spec); if the installed
    solver insists on ``model=`` we fall back to that with the supplied
    OilModel.
    """
    _require_solver_thermal("global_static_target_C")
    try:
        return float(_global_static_target_C(
            T_in_C=float(T_in_C),
            P_loss_W=float(P_loss_W),
            mdot_kg_s=float(mdot_kg_s),
            cp_J_kgK=float(cp_J_kgK),
            gamma=float(gamma),
        ))
    except TypeError:
        if model is None:
            raise
        return float(_global_static_target_C(
            T_in_C=float(T_in_C),
            P_loss_W=float(P_loss_W),
            mdot_kg_s=float(mdot_kg_s),
            model=model,
        ))


__all__ = [
    "SOLVER_THERMAL_AVAILABLE",
    "ThermalConfig",
    "build_oil_walther",
    "viscosity_at_T_C",
    "global_static_step",
]
