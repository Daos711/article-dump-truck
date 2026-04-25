"""Квазистационарный анализ подшипника ДВС карьерного самосвала.

Методологическая оговорка: для каждого угла поворота коленвала φ решается
СТАЦИОНАРНАЯ задача Рейнольдса при эквивалентной внешней нагрузке F_ext(φ).
Squeeze-эффект (∂h/∂t), инерция вала и полная связка Reynolds + equation of
motion НЕ учитываются (квазистационарная аппроксимация).

Stage THD-0 (global_static) extension
-------------------------------------
``run_diesel_analysis(thermal=ThermalConfig(...))`` adds an outer per-angle
fixed-point iteration around the solver's ``global_static_target_C`` law:

    P_loss   = F_tr * U,   U = omega * R
    mdot     = max(rho * |Q|, mdot_floor)
    T_target = T_in + gamma * P_loss / (mdot * cp)

Convergence is on |T_new - T_guess| < tol_T_C with under-relaxation.

``thermal=None`` or ``thermal.mode == "off"`` keeps the legacy isothermal
behaviour exactly (eta = oil["eta_diesel"]); the new result arrays are
filled with consistent values so downstream code can read them
unconditionally.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# Stage THD-0B sanity thresholds. Above these the SOR has visibly
# diverged but still returns *finite* floats (e.g. delta=1.0 at
# 50 000 iterations on textured peaks). Treat that as a solver failure
# so build_load_table / find_epsilon_for_load do not poison the chain.
_W_SANITY_LIMIT_N = 1.0e10     # 10 GN — 4 orders of magnitude above
                                #         any physical bearing load
_P_SANITY_LIMIT_Pa = 1.0e10    # 10 GPa — 1 order of magnitude above
                                #         peak hydrodynamic pressure
_SOR_WARN_FRAGMENT = "SOR не сошёлся"  # exact warning emitted by
                                          # bearing_model on non-conv.

from models.bearing_model import (
    setup_grid, setup_texture, make_H, solve_and_compute,
    DEFAULT_CLOSURE, DEFAULT_CAVITATION,
)
from models.thermal_coupling import (
    ThermalConfig,
    build_oil_walther,
    viscosity_at_T_C,
    global_static_step,
)
from config import diesel_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

N_GRID = 100                           # 100 для трендов, 300 для финала
PHI_CRANK = np.linspace(0, 720, 36)   # шаг 20° для трендов, 360 для финала

CONFIGS = [
    {"label": "Гладкий + минеральное", "textured": False,
     "oil": MINERAL_OIL, "color": "blue", "ls": "-"},
    {"label": "Гладкий + рапсовое", "textured": False,
     "oil": RAPESEED_OIL, "color": "blue", "ls": "--"},
    {"label": "Текстура + минеральное", "textured": True,
     "oil": MINERAL_OIL, "color": "red", "ls": "-"},
    {"label": "Текстура + рапсовое", "textured": True,
     "oil": RAPESEED_OIL, "color": "red", "ls": "--"},
]


# Public registry mapping the short labels exposed in the CLI to entries
# in ``CONFIGS``. Lets the entry script accept e.g.
# ``--configs mineral_smooth,mineral_textured`` without typing Russian
# labels on the command line.
CONFIG_KEYS: Dict[str, int] = {
    "mineral_smooth":   0,
    "rapeseed_smooth":  1,
    "mineral_textured": 2,
    "rapeseed_textured": 3,
}


def load_diesel(phi_deg, F_max=None):
    """Упрощённая индикаторная диаграмма 4-тактного ДВС.

    phi_deg : 0..720°. Пик при φ ≈ 370° (после ВМТ рабочего хода).
    """
    if F_max is None:
        F_max = params.F_max
    F_gas = F_max * np.exp(-((phi_deg - 370) ** 2) / (2 * 30 ** 2))
    F_inertia = 0.1 * F_max * np.cos(2 * np.deg2rad(phi_deg))
    F_total = F_gas + F_inertia + 0.05 * F_max
    return np.maximum(F_total, 20000.0)


def _solver_result_is_sane(F: float, P: np.ndarray) -> bool:
    """Reject results that look mathematically valid but are physically
    nonsense.

    Checks:
      * ``F`` and every entry of ``P`` are finite;
      * ``|F| < _W_SANITY_LIMIT_N`` (10 GN);
      * ``max|P| < _P_SANITY_LIMIT_Pa`` (10 GPa).

    SOR can return arrays with values ~1e72 when divergent (the textured
    BelAZ peak triggers this). Such results pass ``np.isfinite`` but are
    clearly garbage; pretending they are real poisons the warm-start
    chain and the load matcher.
    """
    if not np.isfinite(F):
        return False
    if abs(float(F)) > _W_SANITY_LIMIT_N:
        return False
    if not np.all(np.isfinite(P)):
        return False
    if float(np.max(np.abs(P))) > _P_SANITY_LIMIT_Pa:
        return False
    return True


def _solve_and_check(
    H, d_phi, d_Z, R_, L_, eta_eff, n_, c_,
    phi_1D, Z_1D, Phi_mesh, *, P_init=None,
    closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION,
    closure_kw_retry: Optional[Dict[str, Any]] = None,
    allow_cold_retry: bool = True,
):
    """Wrap solve_and_compute with SOR-warning capture + sanity check.

    Returns ``(P, F, mu, Q, h_min, p_max, F_tr, ok, reason)`` where ``ok``
    is False when the SOR warned non-convergence OR the sanity check
    rejected the magnitudes. On failure callers should reset their
    ``P_init`` warm-start to ``None`` and mark the angle as
    ``solver_failed``.

    Stage Texture Stability Diesel: if the first attempt fails AND
    ``allow_cold_retry`` is True AND ``P_init`` was non-None (i.e. we
    were warm-starting), one cold-start retry is attempted with
    ``P_init=None``. If ``closure_kw_retry`` is supplied it is forwarded
    to ``solve_and_compute(closure_kw=...)`` on the retry only — the
    intent is to plug in a more conservative SOR omega when the solver
    API exposes one. ``reason`` carries ``retried_cold_start`` /
    ``retried_with_kw`` annotations so the script can log retry usage.
    """
    def _attempt(*, p_init, closure_kw=None):
        sor_diverged = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                P, F, mu_v, Qv, hm, pm, F_tr, _, _, _ = solve_and_compute(
                    H, d_phi, d_Z, R_, L_, eta_eff, n_, c_,
                    phi_1D, Z_1D, Phi_mesh, P_init=p_init,
                    closure=closure, cavitation=cavitation,
                    closure_kw=closure_kw,
                )
            except TypeError:
                # Older solver builds may not accept closure_kw=None.
                P, F, mu_v, Qv, hm, pm, F_tr, _, _, _ = solve_and_compute(
                    H, d_phi, d_Z, R_, L_, eta_eff, n_, c_,
                    phi_1D, Z_1D, Phi_mesh, P_init=p_init,
                    closure=closure, cavitation=cavitation,
                )
            except Exception as exc:
                return (None, float("nan"), float("nan"), float("nan"),
                        float("nan"), float("nan"), float("nan"), False,
                        f"exception:{type(exc).__name__}:{exc}")
            for w in caught:
                if _SOR_WARN_FRAGMENT in str(w.message):
                    sor_diverged = True
                    break
        if sor_diverged:
            return (P, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), False,
                    "SOR_did_not_converge")
        if not _solver_result_is_sane(F, P):
            F_repr = f"{float(F):.3e}" if np.isfinite(F) else repr(F)
            P_repr = f"max|P|={float(np.max(np.abs(P))):.3e}"
            return (P, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), False,
                    f"non_physical:F={F_repr},{P_repr}")
        return (P, float(F), float(mu_v), float(Qv), float(hm),
                float(pm), float(F_tr), True, "ok")

    res = _attempt(p_init=P_init, closure_kw=None)
    if res[7]:
        return res

    # Stage Texture Stability Diesel — conservative retry. Only retry if
    # the first attempt was warm-started (cold start can't get worse by
    # restarting cold). Never reuse pressure from the failed solve.
    first_reason = res[8]
    if not allow_cold_retry or P_init is None:
        return res

    res2 = _attempt(p_init=None, closure_kw=closure_kw_retry)
    if res2[7]:
        # Annotate the success so callers can count retries.
        annotated = list(res2)
        tag = "retried_cold_start"
        if closure_kw_retry:
            tag = "retried_with_kw"
        annotated[8] = tag
        return tuple(annotated)
    # Both attempts failed. Combine reasons in the diagnostic string.
    annotated = list(res2)
    annotated[8] = (
        f"first:{first_reason};retry_cold:{res2[8]}"
    )
    return tuple(annotated)


def _eps_max_hydro(override: Optional[float] = None) -> float:
    """Top-end of the eps grid used by the THD load matcher.

    Stage THD-0B caps this at the physical ``params.eps_max`` (default
    0.95) instead of the historical 0.98 — at sigma=2 µm, eps=0.98 is
    already inside the mixed-lubrication band and the full-film
    bisection there is unphysical. Override is allowed for callers that
    explicitly want a different cap.
    """
    if override is not None:
        return float(override)
    return float(getattr(params, "eps_max", 0.95))


def build_load_table(Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                     oil, textured=False, phi_c=None, Z_c=None,
                     closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION,
                     eps_max_hydro: Optional[float] = None,
                     params_view=None,
                     profile: str = "sqrt"):
    """Построить таблицу W(ε) для быстрой интерполяции.

    Reference viscosity is ``oil["eta_diesel"]``. THD path uses this same
    table — load capacity in isoviscous Reynolds scales linearly with η,
    so a thermal-shifted target is found by scaling
    ``F_target_ref = F_target * eta_ref / eta_guess`` (Section 4 of the
    patch spec).

    Stage THD-0B safeguards (no behaviour change on healthy runs):
      * ε grid is capped at the physical ``eps_max_hydro`` (defaults to
        ``params.eps_max``, typically 0.95) — see ``_eps_max_hydro``.
      * Each W_table entry is checked for finiteness; if the solver
        returns ``NaN``/``Inf`` the entry is recorded as ``NaN`` and the
        warm-start ``P_init`` is reset to ``None`` so the next iteration
        cannot inherit a poisoned pressure field. This prevents the
        runaway SOR cascade observed on textured BelAZ peaks.

    Returns
    -------
    eps_table : (M,)
    W_table   : (M,)  — may contain NaN entries when solver fails;
                          callers must check ``np.isfinite(W_table)``.
    """
    eta = oil["eta_diesel"]
    eps_top = _eps_max_hydro(eps_max_hydro)
    eps_table = np.concatenate([
        np.linspace(0.001, 0.05, 5),
        np.linspace(0.06, eps_top, 15),
    ])
    W_table = np.full(len(eps_table), np.nan, dtype=float)
    P_prev = None
    if params_view is None:
        params_view = params

    n_failed = 0
    n_retried = 0
    for i, eps in enumerate(eps_table):
        H = make_H(eps, Phi_mesh, Z_mesh, params_view,
                   textured=textured, phi_c_flat=phi_c, Z_c_flat=Z_c,
                   profile=profile)
        P, F, _, _, _, _, _, ok, reason = _solve_and_check(
            H, d_phi, d_Z, params.R, params.L, eta, params.n, params.c,
            phi_1D, Z_1D, Phi_mesh, P_init=P_prev,
            closure=closure, cavitation=cavitation,
            allow_cold_retry=textured,  # retry only for textured
        )
        if reason in ("retried_cold_start", "retried_with_kw"):
            n_retried += 1
        if not ok:
            print(f"    [warn] W_table eps={eps:.3f} rejected: {reason}; "
                  f"resetting P_init")
            W_table[i] = np.nan
            P_prev = None
            n_failed += 1
            continue
        W_table[i] = F
        P_prev = P

    finite = np.isfinite(W_table)
    if finite.any():
        msg = (f"    W_table: min={np.nanmin(W_table)/1000:.1f} кН, "
               f"max={np.nanmax(W_table)/1000:.1f} кН, "
               f"finite={int(finite.sum())}/{len(W_table)}")
        if n_retried:
            msg += f", cold_retry_recovered={n_retried}"
        print(msg)
    else:
        print(f"    W_table: ALL ENTRIES FAILED ({n_failed}/{len(W_table)})")
    # Diagnostic — non-monotone finite subset means the bracket
    # localisation will use a sorted view (find_epsilon_for_load already
    # sorts), but flag it so the caller can record the anomaly.
    monotone = True
    if finite.sum() >= 2:
        W_finite = W_table[finite]
        monotone = bool(np.all(np.diff(W_finite) >= 0))
        if not monotone:
            print(f"    [warn] W_table non-monotone after filter")
    return eps_table, W_table, dict(
        n_failed=int(n_failed),
        n_retried_cold=int(n_retried),
        monotone=monotone,
    )


def find_epsilon_for_load(F_target, eps_table, W_table,
                          Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                          oil, textured=False, phi_c=None, Z_c=None,
                          closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION,
                          eta=None,
                          eps_max_hydro: Optional[float] = None,
                          params_view=None,
                          profile: str = "sqrt"):
    """Найти ε для заданной нагрузки: локализация по таблице + 5 шагов бисекции.

    ``eta`` overrides the per-oil isothermal viscosity (Section 4 of the
    THD-0 patch). When given, the bracket localisation uses
    ``F_target_ref = F_target * eta_ref / eta`` against the reference
    W_table, then bisection runs at the actual ``eta`` and converges to
    ``F_hyd ≈ F_target``.

    Stage THD-0B additions:
      * Bracket localisation skips ``NaN`` entries in ``W_table``; if
        fewer than two finite entries remain we report
        ``status = "wtable_failed"`` and return without any further
        solver calls.
      * The high-eps clamp uses ``params.eps_max`` (or an override),
        not a hard-coded 0.98.
      * Each in-loop ``solve_and_compute`` is wrapped against exceptions
        and ``NaN`` returns; on failure we report ``status =
        "solver_failed"`` and break the bisection without poisoning the
        outer THD loop.

    Returns
    -------
    eps_mid, F_hyd, mu, Q, h_min, p_max, F_tr, status
        ``status`` ∈ {"ok", "below_range", "above_range",
                      "wtable_failed", "solver_failed"}.
    """
    eta_ref = oil["eta_diesel"]
    eta_eff = float(eta) if eta is not None else float(eta_ref)
    F_target_ref = F_target * eta_ref / eta_eff
    eps_top = _eps_max_hydro(eps_max_hydro)
    if params_view is None:
        params_view = params

    finite_mask = np.isfinite(W_table)
    n_finite = int(finite_mask.sum())
    if n_finite < 2:
        return (
            float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), "wtable_failed",
        )

    eps_finite = np.asarray(eps_table)[finite_mask]
    W_finite = np.asarray(W_table)[finite_mask]
    # Sort by W just in case the eps grid produced a slightly non-monotonic
    # W_table (rare but possible near texture transitions).
    order = np.argsort(W_finite)
    eps_finite = eps_finite[order]
    W_finite = W_finite[order]

    status = "ok"
    if F_target_ref <= W_finite[0]:
        eps_lo, eps_hi = max(0.001, float(eps_finite[0])), float(eps_finite[1])
        status = "below_range"
    elif F_target_ref >= W_finite[-1]:
        eps_lo, eps_hi = float(eps_finite[-2]), eps_top
        status = "above_range"
    else:
        j = int(np.searchsorted(W_finite, F_target_ref))
        eps_lo, eps_hi = float(eps_finite[j - 1]), float(eps_finite[j])

    eps_mid = 0.5 * (eps_lo + eps_hi)
    F_hyd = mu = Qv = h_min = p_max = F_tr = 0.0

    for _ in range(5):
        eps_mid = 0.5 * (eps_lo + eps_hi)
        # Clamp eps_mid to the hydro cap (relevant in above_range).
        eps_mid = min(eps_mid, eps_top)
        H = make_H(eps_mid, Phi_mesh, Z_mesh, params_view,
                   textured=textured, phi_c_flat=phi_c, Z_c_flat=Z_c,
                   profile=profile)
        _, F_hyd, mu, Qv, h_min, p_max, F_tr, ok, reason = (
            _solve_and_check(
                H, d_phi, d_Z, params.R, params.L, eta_eff,
                params.n, params.c,
                phi_1D, Z_1D, Phi_mesh,
                closure=closure, cavitation=cavitation,
                allow_cold_retry=textured,
            )
        )
        if not ok:
            print(f"    [warn] inner solve eps={eps_mid:.3f} rejected: "
                  f"{reason}")
            return (eps_mid, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"),
                    "solver_failed")
        if F_hyd > F_target:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, status


def _omega_rad_s() -> float:
    return 2.0 * math.pi * params.n / 60.0


def _angle_thd_step(
    F_target: float,
    eps_table: np.ndarray, W_table: np.ndarray,
    Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
    *,
    oil: Dict[str, Any],
    textured: bool,
    phi_c, Z_c,
    closure: str, cavitation: str,
    walther_fit: Any,
    thermal: ThermalConfig,
    T_init_C: Optional[float] = None,
    eps_max_hydro: Optional[float] = None,
    params_view=None,
    profile: str = "sqrt",
):
    """Per-angle THD outer loop. Returns dict with all per-angle scalars.

    The outer loop is the textbook fixed point on
    ``T = T_in + gamma * F_tr * U / (rho |Q| cp)`` with under-relaxation
    and a strict ``max_outer`` cap (Section 3 of the patch spec).

    Stage THD-0B additions:
      * After the loop ends (converged or max_outer hit), ``eta_eff`` is
        recomputed at the final ``T_eff`` so it stays consistent with
        ``T_eff`` even when convergence wasn't reached.
      * ``solver_failed`` / ``wtable_failed`` propagate up and the
        record stays finite for the in-loop scalars while the status
        flag flags the angle as not usable for THD metrics.
    """
    rho = float(oil["rho"])
    cp = thermal.cp_J_kgK
    gamma = thermal.gamma_mix
    T_in = thermal.T_in_C
    U = _omega_rad_s() * params.R

    T_guess = float(T_init_C) if T_init_C is not None else float(T_in)
    energy_converged = False
    last_T_target = float(T_in)
    mdot_floor_hit = False
    outer_iters = 0

    eps_mid = F_hyd = mu = Qv = h_min = p_max = F_tr = 0.0
    eta_eff = float(oil["eta_diesel"])
    status = "ok"

    for it in range(thermal.max_outer):
        outer_iters = it + 1
        eta_eff = viscosity_at_T_C(walther_fit, T_guess)
        eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, status = (
            find_epsilon_for_load(
                F_target, eps_table, W_table,
                Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                oil=oil, textured=textured,
                phi_c=phi_c, Z_c=Z_c,
                closure=closure, cavitation=cavitation,
                eta=eta_eff,
                eps_max_hydro=eps_max_hydro,
                params_view=params_view,
                profile=profile,
            )
        )
        if status in ("wtable_failed", "solver_failed"):
            # Cannot continue the THD outer loop on a broken inner solve.
            break
        P_loss = float(F_tr) * U
        mdot_raw = rho * abs(float(Qv))
        mdot = max(mdot_raw, thermal.mdot_floor_kg_s)
        if mdot_raw < thermal.mdot_floor_kg_s:
            mdot_floor_hit = True
        T_target = global_static_step(
            T_in_C=T_in, P_loss_W=P_loss, mdot_kg_s=mdot,
            cp_J_kgK=cp, gamma=gamma,
            model=walther_fit,
        )
        last_T_target = float(T_target)
        if abs(T_target - T_guess) < thermal.tol_T_C:
            T_guess = T_target
            energy_converged = True
            break
        T_guess = T_guess + thermal.underrelax_T * (T_target - T_guess)

    # Recompute final eta at the final T_eff so eta_eff stays consistent
    # with T_eff regardless of whether the loop converged.
    if status not in ("wtable_failed", "solver_failed"):
        eta_eff = viscosity_at_T_C(walther_fit, T_guess)

    F_target_safe = max(float(F_target), 1.0)
    load_match_ratio = (float(F_hyd) / F_target_safe
                        if np.isfinite(F_hyd) else float("nan"))

    return dict(
        eps=eps_mid, F_hyd=F_hyd, mu=mu, Q=Qv, h_min=h_min,
        p_max=p_max, F_tr=F_tr, status=status,
        T_eff=float(T_guess), T_target=float(last_T_target),
        eta_eff=float(eta_eff),
        P_loss=(float(F_tr) * U if np.isfinite(F_tr) else float("nan")),
        mdot=(max(rho * abs(float(Qv)), thermal.mdot_floor_kg_s)
              if np.isfinite(Qv) else float("nan")),
        outer=int(outer_iters),
        energy_converged=bool(energy_converged),
        mdot_floor_hit=bool(mdot_floor_hit),
        load_match_ratio=float(load_match_ratio),
    )


def _is_valid_fullfilm(*, status: str, F_hyd: float, F_tr: float,
                         Qv: float, load_match_ratio: float) -> bool:
    """Stage THD-0B valid_fullfilm rule (Section 3 of the patch).

    A node counts toward THD headline metrics only when the inner solve
    succeeded, every scalar is finite, AND the matched load is within
    5 percent of the requested load.
    """
    if status != "ok":
        return False
    for v in (F_hyd, F_tr, Qv, load_match_ratio):
        if not np.isfinite(v):
            return False
    return load_match_ratio >= 0.95


class _ParamsView:
    """Lightweight view over ``config.diesel_params`` with selected
    overrides — used to inject CLI-driven texture-depth / profile
    diagnostics without touching the persistent module-level params.

    The view forwards every attribute access to the underlying module
    except the explicitly overridden ones.
    """
    __slots__ = ("_base", "_overrides")

    def __init__(self, base, **overrides):
        self._base = base
        self._overrides = dict(overrides)

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def run_diesel_analysis(closure=DEFAULT_CLOSURE,
                          cavitation=DEFAULT_CAVITATION,
                          *,
                          thermal: Optional[ThermalConfig] = None,
                          phi_crank=None,
                          n_phi_grid=None,
                          n_z_grid=None,
                          configs=None,
                          F_max: Optional[float] = None,
                          eps_max_hydro: Optional[float] = None,
                          texture_h_p_m: Optional[float] = None,
                          texture_profile: Optional[str] = None):
    """Выполнить квазистационарный расчёт ДВС для всех конфигураций.

    Parameters
    ----------
    thermal : ThermalConfig | None
        ``None`` or ``mode == "off"`` — legacy isothermal behaviour with
        η = ``oil["eta_diesel"]``. ``mode == "global_static"`` runs the
        per-angle outer fixed-point loop described above.
    phi_crank, n_phi_grid, n_z_grid, configs
        Optional overrides for the module-level defaults; ``None`` keeps
        the existing behaviour (PHI_CRANK / N_GRID / CONFIGS).
    F_max : float | None
        Override for the load-cycle peak (N). ``None`` uses
        ``params.F_max`` (production BelAZ, 850 kN).
    eps_max_hydro : float | None
        High-eps cap for the hydrodynamic load matcher. ``None`` uses
        ``params.eps_max`` (default 0.95). Replaces the historical
        hard-coded 0.98.

    Returns
    -------
    results : dict
        Legacy keys (``epsilon``, ``hmin``, ``f``, ``pmax``, ``F_hyd``,
        ``configs``, ``phi_crank``, ``F_ext``) plus THD arrays
        (``T_eff``, ``T_target``, ``eta_eff``, ``P_loss``, ``Q``,
        ``mdot``, ``F_tr``, ``thermal_outer``, ``thermal_converged``,
        ``thermal_energy_converged``, ``mdot_floor_hit``) plus the
        Stage THD-0B diagnostics (``load_status``, ``load_match_ratio``,
        ``valid_fullfilm``, ``W_table_max``, ``W_table_finite``,
        ``F_max_used``) and ``thermal_config``.

        ``thermal_converged`` is now redefined as
        ``thermal_energy_converged AND valid_fullfilm`` so the headline
        flag only fires when the angle is *both* energy-converged *and*
        load-matched on a healthy full-film inner solve. The original
        energy-only flag is exposed separately as
        ``thermal_energy_converged``.
    """
    if thermal is None:
        thermal = ThermalConfig(mode="off")
    is_off = thermal.is_off()

    phi_crank = (np.asarray(phi_crank, dtype=float)
                  if phi_crank is not None else PHI_CRANK)
    cfg_list = list(configs) if configs is not None else CONFIGS
    Np = int(n_phi_grid) if n_phi_grid is not None else N_GRID
    Nz = int(n_z_grid) if n_z_grid is not None else Np
    eps_top = _eps_max_hydro(eps_max_hydro)
    F_max_used = float(F_max) if F_max is not None else float(params.F_max)
    profile = (texture_profile or "sqrt").strip().lower()
    if profile == "current":
        profile = "sqrt"
    if profile not in ("sqrt", "smoothcap"):
        raise ValueError(f"unknown texture_profile {profile!r}")

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(Np, Nz)
    # Build texture-params view if overridden so setup_texture uses the
    # base layout (untouched) but make_H sees the new depth via H_p.
    if texture_h_p_m is not None:
        params_view = _ParamsView(params, h_p=float(texture_h_p_m))
    else:
        params_view = params
    phi_c, Z_c = setup_texture(params)

    F_ext = load_diesel(phi_crank, F_max=F_max_used)
    n_phi = len(phi_crank)
    n_cfg = len(cfg_list)

    eps_arr = np.full((n_cfg, n_phi), np.nan)
    hmin_arr = np.full((n_cfg, n_phi), np.nan)
    f_arr = np.full((n_cfg, n_phi), np.nan)
    pmax_arr = np.full((n_cfg, n_phi), np.nan)
    F_hyd_arr = np.full((n_cfg, n_phi), np.nan)

    T_eff_arr = np.full((n_cfg, n_phi), np.nan)
    T_target_arr = np.full((n_cfg, n_phi), np.nan)
    eta_eff_arr = np.full((n_cfg, n_phi), np.nan)
    P_loss_arr = np.full((n_cfg, n_phi), np.nan)
    Q_arr = np.full((n_cfg, n_phi), np.nan)
    mdot_arr = np.full((n_cfg, n_phi), np.nan)
    F_tr_arr = np.full((n_cfg, n_phi), np.nan)
    outer_arr = np.zeros((n_cfg, n_phi), dtype=np.int32)
    energy_conv_arr = np.zeros((n_cfg, n_phi), dtype=bool)
    converged_arr = np.zeros((n_cfg, n_phi), dtype=bool)
    mdot_floor_hit_arr = np.zeros((n_cfg, n_phi), dtype=bool)
    load_status_arr = np.full((n_cfg, n_phi), "ok", dtype="<U16")
    load_match_arr = np.full((n_cfg, n_phi), np.nan)
    valid_fullfilm_arr = np.zeros((n_cfg, n_phi), dtype=bool)

    W_table_max = np.full(n_cfg, np.nan)
    W_table_finite = np.zeros(n_cfg, dtype=bool)
    W_table_n_failed = np.zeros(n_cfg, dtype=np.int32)
    W_table_n_retried_cold = np.zeros(n_cfg, dtype=np.int32)
    W_table_monotone = np.ones(n_cfg, dtype=bool)
    n_above_arr = np.zeros(n_cfg, dtype=np.int32)
    n_below_arr = np.zeros(n_cfg, dtype=np.int32)
    n_solver_failed_arr = np.zeros(n_cfg, dtype=np.int32)
    load_coverable_arr = np.full(n_cfg, n_phi, dtype=np.int32)
    solver_success_on_coverable = np.full(n_cfg, np.nan)

    U = _omega_rad_s() * params.R

    for ic, cfg in enumerate(cfg_list):
        print(f"  [{ic+1}/{n_cfg}] {cfg['label']}...")

        # Шаг 1: таблица W(ε) — построена один раз на reference η.
        print("    Построение таблицы W(ε)...")
        eps_table, W_table, wtable_info = build_load_table(
            Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
            oil=cfg["oil"], textured=cfg["textured"],
            phi_c=phi_c, Z_c=Z_c,
            closure=closure, cavitation=cavitation,
            eps_max_hydro=eps_top,
            params_view=params_view,
            profile=profile,
        )
        W_table_n_failed[ic] = int(wtable_info.get("n_failed", 0))
        W_table_n_retried_cold[ic] = int(wtable_info.get("n_retried_cold", 0))
        W_table_monotone[ic] = bool(wtable_info.get("monotone", True))
        finite_mask = np.isfinite(W_table)
        n_finite = int(finite_mask.sum())
        if n_finite >= 1:
            W_table_max[ic] = float(np.nanmax(W_table))
        W_table_finite[ic] = (n_finite >= 3)

        print(f"    F_ext:   min={F_ext.min()/1000:.1f} кН, "
              f"max={F_ext.max()/1000:.1f} кН")

        if not W_table_finite[ic]:
            # Section 4 of THD-0B: skip THD loop entirely on a broken
            # W_table. Mark every angle as wtable_failed so the caller
            # can report it separately.
            print(f"    [warn] W_table has only {n_finite} finite "
                  f"entries — config marked wtable_failed, no "
                  f"per-angle solve attempted")
            load_status_arr[ic, :] = "wtable_failed"
            T_eff_arr[ic, :] = thermal.T_in_C
            eta_eff_arr[ic, :] = (cfg["oil"]["eta_diesel"]
                                   if is_off else np.nan)
            continue

        # Walther fit (only when THD is active — keeps mode='off' free of
        # solver thermal dependency). Pass cp/gamma into the OilModel so
        # the solver's static balance step uses the same numbers as
        # ThermalConfig.
        if is_off:
            walther_fit = None
        else:
            walther_fit = build_oil_walther(
                cfg["oil"],
                cp_J_kgK=thermal.cp_J_kgK,
                gamma_mix=thermal.gamma_mix,
            )

        # Шаг 2: по каждому углу.
        n_below = n_above = n_solver_failed = 0
        n_energy_unconverged = 0
        for ip, phi_k in enumerate(phi_crank):
            F_target = F_ext[ip]
            if is_off:
                eps_mid, F_hyd, mu, Qv, h_min, p_max, F_tr, st = (
                    find_epsilon_for_load(
                        F_target, eps_table, W_table,
                        Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                        oil=cfg["oil"], textured=cfg["textured"],
                        phi_c=phi_c, Z_c=Z_c,
                        closure=closure, cavitation=cavitation,
                        eps_max_hydro=eps_top,
                        params_view=params_view,
                        profile=profile,
                    )
                )
                eta_eff = float(cfg["oil"]["eta_diesel"])
                T_eff_C = thermal.T_in_C
                T_target_C = thermal.T_in_C
                if np.isfinite(F_tr) and np.isfinite(Qv):
                    P_loss = float(F_tr) * U
                    mdot = max(float(cfg["oil"]["rho"]) * abs(float(Qv)),
                                thermal.mdot_floor_kg_s)
                    mdot_floor_hit = (mdot <= thermal.mdot_floor_kg_s + 1e-30)
                else:
                    P_loss = float("nan")
                    mdot = float("nan")
                    mdot_floor_hit = False
                outer_iters = 0
                energy_conv = True
            else:
                rec = _angle_thd_step(
                    F_target, eps_table, W_table,
                    Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, d_Z,
                    oil=cfg["oil"], textured=cfg["textured"],
                    phi_c=phi_c, Z_c=Z_c,
                    closure=closure, cavitation=cavitation,
                    walther_fit=walther_fit,
                    thermal=thermal,
                    T_init_C=None,  # cold-start each angle, no thermal memory
                    eps_max_hydro=eps_top,
                    params_view=params_view,
                    profile=profile,
                )
                eps_mid = rec["eps"]; F_hyd = rec["F_hyd"]; mu = rec["mu"]
                Qv = rec["Q"]; h_min = rec["h_min"]; p_max = rec["p_max"]
                F_tr = rec["F_tr"]; st = rec["status"]
                eta_eff = rec["eta_eff"]
                T_eff_C = rec["T_eff"]; T_target_C = rec["T_target"]
                P_loss = rec["P_loss"]; mdot = rec["mdot"]
                outer_iters = rec["outer"]
                energy_conv = rec["energy_converged"]
                mdot_floor_hit = rec["mdot_floor_hit"]
                if not energy_conv:
                    n_energy_unconverged += 1

            if st == "below_range":
                n_below += 1
            elif st == "above_range":
                n_above += 1
            elif st == "solver_failed":
                n_solver_failed += 1

            F_target_safe = max(float(F_target), 1.0)
            load_match = (float(F_hyd) / F_target_safe
                           if np.isfinite(F_hyd) else float("nan"))
            valid = _is_valid_fullfilm(
                status=st, F_hyd=F_hyd, F_tr=F_tr, Qv=Qv,
                load_match_ratio=load_match,
            )

            eps_arr[ic, ip] = eps_mid
            hmin_arr[ic, ip] = h_min
            f_arr[ic, ip] = mu
            pmax_arr[ic, ip] = p_max
            F_hyd_arr[ic, ip] = F_hyd
            T_eff_arr[ic, ip] = T_eff_C
            T_target_arr[ic, ip] = T_target_C
            eta_eff_arr[ic, ip] = eta_eff
            P_loss_arr[ic, ip] = P_loss
            Q_arr[ic, ip] = Qv
            mdot_arr[ic, ip] = mdot
            F_tr_arr[ic, ip] = F_tr
            outer_arr[ic, ip] = outer_iters
            energy_conv_arr[ic, ip] = energy_conv
            converged_arr[ic, ip] = bool(energy_conv and valid)
            mdot_floor_hit_arr[ic, ip] = mdot_floor_hit
            load_status_arr[ic, ip] = st
            load_match_arr[ic, ip] = load_match
            valid_fullfilm_arr[ic, ip] = valid

        n_valid = int(np.sum(valid_fullfilm_arr[ic]))
        n_ok_status = int(np.sum(load_status_arr[ic] == "ok"))
        # load-coverable = n_phi - above - below (capacity-feasible angles).
        load_coverable = max(0, n_phi - n_above - n_below)
        load_coverable_arr[ic] = load_coverable
        if load_coverable > 0:
            solver_success_on_coverable[ic] = (
                n_ok_status / load_coverable
            )
        n_above_arr[ic] = n_above
        n_below_arr[ic] = n_below
        n_solver_failed_arr[ic] = n_solver_failed
        succ_pct = (
            f"{solver_success_on_coverable[ic]*100:.0f}%"
            if load_coverable > 0 else "n/a"
        )
        print(f"    Status: ok={n_ok_status} below={n_below} "
              f"above={n_above} solver_failed={n_solver_failed}; "
              f"valid_fullfilm={n_valid}/{n_phi}; "
              f"solver_success_on_coverable={succ_pct} "
              f"({n_ok_status}/{load_coverable})")
        if not is_off:
            valid_mask = valid_fullfilm_arr[ic]
            if valid_mask.any():
                T_v = T_eff_arr[ic][valid_mask]
                T_lo = float(np.min(T_v))
                T_hi = float(np.max(T_v))
                T_mn = float(np.mean(T_v))
                n_floor = int(np.sum(mdot_floor_hit_arr[ic] & valid_mask))
                print(f"    THD valid_fullfilm: T_eff min/mean/max = "
                      f"{T_lo:.1f}/{T_mn:.1f}/{T_hi:.1f} °C, "
                      f"energy_unconverged={n_energy_unconverged}, "
                      f"mdot_floor_hit={n_floor}")
                if T_hi > 180.0:
                    print(f"    [WARN] valid_fullfilm max T_eff "
                          f"= {T_hi:.1f} °C > 180 °C")
            else:
                print(f"    [WARN] no valid_fullfilm angles for "
                      f"{cfg['label']!r} — THD metrics unavailable")

    paired = _compute_paired(cfg_list, valid_fullfilm_arr,
                              T_eff_arr, eta_eff_arr, P_loss_arr,
                              hmin_arr, pmax_arr, eps_arr, F_tr_arr)
    return {
        "phi_crank": phi_crank,
        "F_ext": F_ext,
        "F_max_used": F_max_used,
        "eps_max_hydro": eps_top,
        "texture_h_p_used_m": (float(texture_h_p_m)
                                 if texture_h_p_m is not None
                                 else float(params.h_p)),
        "texture_profile_used": profile,
        "epsilon": eps_arr,
        "hmin": hmin_arr,
        "f": f_arr,
        "pmax": pmax_arr,
        "F_hyd": F_hyd_arr,
        "configs": cfg_list,
        "T_eff": T_eff_arr,
        "T_target": T_target_arr,
        "eta_eff": eta_eff_arr,
        "P_loss": P_loss_arr,
        "Q": Q_arr,
        "mdot": mdot_arr,
        "F_tr": F_tr_arr,
        "thermal_outer": outer_arr,
        "thermal_energy_converged": energy_conv_arr,
        "thermal_converged": converged_arr,
        "mdot_floor_hit": mdot_floor_hit_arr,
        "load_status": load_status_arr,
        "load_match_ratio": load_match_arr,
        "valid_fullfilm": valid_fullfilm_arr,
        "W_table_max": W_table_max,
        "W_table_finite": W_table_finite,
        "W_table_n_failed": W_table_n_failed,
        "W_table_n_retried_cold": W_table_n_retried_cold,
        "W_table_monotone": W_table_monotone,
        "n_above_range": n_above_arr,
        "n_below_range": n_below_arr,
        "n_solver_failed": n_solver_failed_arr,
        "load_coverable_count": load_coverable_arr,
        "solver_success_on_coverable": solver_success_on_coverable,
        "paired_comparison": paired,
        "thermal_config": thermal.to_dict(),
    }


def _compute_paired(cfg_list, valid, T_eff, eta_eff, P_loss,
                     hmin, pmax, eps, F_tr):
    """Stage Texture Stability Diesel: paired smooth-vs-textured stats
    on the **common_valid** mask only (Section 1 of the patch).

    Returns a list of dicts, one per oil_key (mineral / rapeseed) for
    which both a textured and a smooth config exist in ``cfg_list``.
    Stats use textured - smooth (positive ΔP_loss = textured worse).
    """
    by_oil: Dict[str, Dict[str, int]] = {}
    for ic, cfg in enumerate(cfg_list):
        oil_key = (cfg.get("oil") or {}).get("name", "")
        bucket = by_oil.setdefault(oil_key, {})
        if cfg.get("textured"):
            bucket["textured_idx"] = ic
        else:
            bucket["smooth_idx"] = ic
    out = []
    for oil_key, bucket in by_oil.items():
        if "textured_idx" not in bucket or "smooth_idx" not in bucket:
            continue
        i_s = bucket["smooth_idx"]
        i_t = bucket["textured_idx"]
        common = (np.asarray(valid[i_s], dtype=bool)
                   & np.asarray(valid[i_t], dtype=bool))
        n_common = int(common.sum())
        n_phi = int(common.size)
        rec = dict(
            oil_name=oil_key,
            smooth_idx=i_s, smooth_label=cfg_list[i_s]["label"],
            textured_idx=i_t, textured_label=cfg_list[i_t]["label"],
            common_valid_count=n_common,
            common_valid_fraction=(n_common / n_phi if n_phi else 0.0),
            mean_dT_eff=float("nan"),
            mean_dP_loss=float("nan"),
            mean_deta_eff=float("nan"),
            mean_dh_min=float("nan"),
            min_dp_max=float("nan"),
            max_dp_max=float("nan"),
            mean_T_smooth=float("nan"),
            mean_T_textured=float("nan"),
            mean_P_loss_smooth=float("nan"),
            mean_P_loss_textured=float("nan"),
        )
        if n_common > 0:
            rec["mean_dT_eff"] = float(
                np.mean(T_eff[i_t][common] - T_eff[i_s][common]))
            rec["mean_dP_loss"] = float(
                np.mean(P_loss[i_t][common] - P_loss[i_s][common]))
            rec["mean_deta_eff"] = float(
                np.mean(eta_eff[i_t][common] - eta_eff[i_s][common]))
            rec["mean_dh_min"] = float(
                np.mean(hmin[i_t][common] - hmin[i_s][common]))
            dpmax = pmax[i_t][common] - pmax[i_s][common]
            rec["min_dp_max"] = float(np.min(dpmax))
            rec["max_dp_max"] = float(np.max(dpmax))
            rec["mean_T_smooth"] = float(np.mean(T_eff[i_s][common]))
            rec["mean_T_textured"] = float(np.mean(T_eff[i_t][common]))
            rec["mean_P_loss_smooth"] = float(
                np.mean(P_loss[i_s][common]))
            rec["mean_P_loss_textured"] = float(
                np.mean(P_loss[i_t][common]))
        out.append(rec)
    return out
