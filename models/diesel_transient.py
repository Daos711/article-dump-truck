"""Нестационарная модель подшипника ДВС (time-stepping).

Совместное решение уравнения Рейнольдса со squeeze и уравнения движения вала.
Интегратор: Velocity Verlet с sub-iterations.

Stage Diesel Transient THD-0 / global_relax overlay
---------------------------------------------------
``run_transient(thermal=ThermalConfig(...))`` adds a per-step thermal
state on top of the existing transient runner:

  * Walther viscosity-temperature fit per oil from existing
    ``eta_pump`` (50 °C) + ``eta_diesel`` (105 °C).
  * Per-step ``eta_step = mu(T_eff_used)`` and
    ``p_scale_step = 6 * eta_step * omega * (R/c)**2``.
  * After the accepted Verlet sub-iteration of a mechanical step,
    ``T_target`` is computed from ``F_tr * U / (rho |Q| cp)`` via
    ``global_static_step``; ``T_state`` is then advanced as

        mode="off"             T_state = T_in
        mode="global_static"   T_state = T_target
        mode="global_relax"    T_state += alpha * (T_target - T_state)
                                  with alpha = 1 - exp(-dt/tau_th_s)

  * ``thermal=None`` / ``mode="off"`` reproduces the legacy isothermal
    transient bit-for-bit (same eta = oil["eta_diesel"], same single
    p_scale per config); the new THD arrays are filled consistently
    (T_eff_used = T_eff = T_in, eta_eff = oil["eta_diesel"], etc.).

Solver-side retry policy (Stage Texture Stability 2) is reused via
``SolverRetryConfig``: textured-only, omegas (1.70, 1.55), cold-start.
"""
import time as _time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
# squeeze helper location differs between solver layouts. The current
# upstream layout is ``reynolds_solver.dynamic.squeeze``; older builds
# kept a flat ``reynolds_solver.squeeze``; some expose it on the top
# level. Try every known path before raising.
try:
    from reynolds_solver.dynamic.squeeze import squeeze_to_api_params
except ImportError:
    try:
        from reynolds_solver.squeeze import squeeze_to_api_params  # type: ignore
    except ImportError:
        try:
            from reynolds_solver import squeeze_to_api_params  # type: ignore
        except ImportError:
            def squeeze_to_api_params(*_a, **_kw):  # type: ignore
                raise RuntimeError(
                    "reynolds_solver.squeeze_to_api_params not found in "
                    "any known location (tried reynolds_solver.dynamic."
                    "squeeze, reynolds_solver.squeeze, top-level). "
                    "Production transient runs need the squeeze helper; "
                    "pipeline contract tests don't depend on it.")
from models.bearing_model import (
    DEFAULT_CAVITATION, DEFAULT_CLOSURE,
    compute_axial_leakage_m3_s,
    setup_grid, setup_texture,
)
# Reuse the SolverRetryConfig + _omega_tag + sanity check from the
# quasistatic stack (already battle-tested in Stage Texture Stability 2).
from models.diesel_quasistatic import (
    SolverRetryConfig,
    _omega_tag,
    _parse_retry_outcome,
    _solver_result_is_sane,
    _SOR_WARN_FRAGMENT,
    _W_SANITY_LIMIT_N,
    _P_SANITY_LIMIT_Pa,
)
from models.thermal_coupling import (
    ThermalConfig,
    build_oil_walther,
    global_relax_step,
    global_static_step,
    viscosity_at_T_C,
)
from config import diesel_params as params
from config.oil_properties import MINERAL_OIL, RAPESEED_OIL

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

# Public registry mapping CLI-friendly labels to CONFIGS index.
CONFIG_KEYS: Dict[str, int] = {
    "mineral_smooth":   0,
    "rapeseed_smooth":  1,
    "mineral_textured": 2,
    "rapeseed_textured": 3,
}


# ─── Stage Diesel Transient Load-Envelope-0 — abort policy ────────


@dataclass(frozen=True)
class EnvelopeAbortConfig:
    """Run-safety policy for transient configs that fall outside the
    full-film envelope (e.g. production 850 kN, where Verlet locks the
    shaft to ε_max within the first few crank angles).

    The runner stops a config cleanly — without traceback or
    KeyboardInterrupt — when one of the thresholds trips:

    * ``clamp_frac_max``       — fraction of clamped steps so far.
    * ``solver_fail_frac_max`` — fraction of solver_failed steps so far.
    * ``consecutive_invalid_max`` — streak of consecutive
      not-valid_dynamic steps from the most recent step.

    Both fractional checks honour ``warmup_steps`` to avoid aborting
    on the first few unstable Verlet steps.
    """
    enabled: bool = True
    clamp_frac_max: float = 0.30
    solver_fail_frac_max: float = 0.30
    consecutive_invalid_max: int = 30
    save_partial_on_abort: bool = True
    warmup_steps: int = 5

    @classmethod
    def disabled(cls) -> "EnvelopeAbortConfig":
        return cls(enabled=False)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            enabled=self.enabled,
            clamp_frac_max=self.clamp_frac_max,
            solver_fail_frac_max=self.solver_fail_frac_max,
            consecutive_invalid_max=self.consecutive_invalid_max,
            save_partial_on_abort=self.save_partial_on_abort,
            warmup_steps=self.warmup_steps,
        )


# ─── envelope classification (Section 3 of the patch spec) ────────

ENVELOPE_THRESHOLDS = dict(
    solver_success_frac_min=0.95,
    valid_dynamic_frac_min=0.90,
    valid_no_clamp_frac_min=0.85,
    paired_no_clamp_frac_min=0.80,
)


def classify_envelope_per_config(*,
                                    n_completed: int,
                                    solver_success_count: int,
                                    valid_dynamic_count: int,
                                    valid_no_clamp_count: int,
                                    retry_exhausted_count: int,
                                    aborted: bool) -> Tuple[bool, str]:
    """Per-config applicable gate.

    Returns ``(applicable, reason)`` where ``reason`` is "ok" on
    success or a human-readable failure string.
    """
    if aborted:
        return False, "aborted_outside_envelope"
    if n_completed <= 0:
        return False, "no_steps_completed"
    sf = float(solver_success_count) / float(n_completed)
    vd = float(valid_dynamic_count) / float(n_completed)
    vnc = float(valid_no_clamp_count) / float(n_completed)
    th = ENVELOPE_THRESHOLDS
    if sf < th["solver_success_frac_min"]:
        return False, (
            f"solver_success_frac={sf:.2f} < "
            f"{th['solver_success_frac_min']:.2f}")
    if vd < th["valid_dynamic_frac_min"]:
        return False, (
            f"valid_dynamic_frac={vd:.2f} < "
            f"{th['valid_dynamic_frac_min']:.2f}")
    if vnc < th["valid_no_clamp_frac_min"]:
        return False, (
            f"valid_no_clamp_frac={vnc:.2f} < "
            f"{th['valid_no_clamp_frac_min']:.2f}")
    if retry_exhausted_count > 0:
        return False, f"retry_exhausted={retry_exhausted_count} > 0"
    return True, "ok"


def classify_paired_envelope(common_no_clamp_count: int,
                               n_steps_min: int) -> Tuple[bool, str]:
    """Per-paired-pair applicable gate (Section 3)."""
    if n_steps_min <= 0:
        return False, "no_overlap"
    frac = float(common_no_clamp_count) / float(n_steps_min)
    th = ENVELOPE_THRESHOLDS["paired_no_clamp_frac_min"]
    if frac < th:
        return False, (
            f"common_valid_no_clamp_frac={frac:.2f} < {th:.2f}")
    return True, "ok"


def load_diesel(phi_deg, F_max=None):
    """Нагрузка ДВС: Вибе-функция + КШМ-разложение на Fx, Fy."""
    if F_max is None:
        F_max = params.F_max

    phi = np.atleast_1d(np.asarray(phi_deg, dtype=float)) % 720.0

    phi_s = 345.0
    phi_p = 370.0
    m_vibe = 2.0
    k_vibe = 1.2

    x = np.clip((phi - phi_s) / (phi_p - phi_s), 0, None)
    F_vibe = np.where(x > 0,
        (F_max - params.F_base) * x**m_vibe * np.exp(m_vibe / k_vibe * (1 - x**k_vibe)),
        0.0)
    F_total = F_vibe + params.F_base

    phi_rad = np.deg2rad(phi)
    beta = np.arcsin(params.lambda_crank * np.sin(phi_rad))
    Fx = F_total * np.sin(beta)
    Fy = -F_total * np.cos(beta)

    return Fx, Fy


def build_H_2d(eps_x, eps_y, Phi_mesh, Z_mesh, p,
               textured=False, phi_c_flat=None, Z_c_flat=None):
    """Зазор для 2D-эксцентриситета: H = 1 − εx·cos(θ) − εy·sin(θ) [+ текстура]."""
    H0 = 1.0 - eps_x * np.cos(Phi_mesh) - eps_y * np.sin(Phi_mesh)
    H0 = np.sqrt(H0**2 + (p.sigma / p.c)**2)  # регуляризация шероховатости
    if not textured:
        return H0
    A = 2 * p.a_dim / p.L
    B = p.b_dim / p.R
    H_p = p.h_p / p.c
    return create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A, B
    )


# numpy 2.x removed ``np.trapz``; pick the modern alias when present.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def compute_hydro_forces(P, p_scale, Phi_mesh, phi_1D, Z_1D, R, L):
    """Вычислить компоненты гидродинамической силы на вал."""
    P_dim = P * p_scale
    Fx = -_trapz(_trapz(P_dim * np.cos(Phi_mesh), phi_1D, axis=1),
                 Z_1D, axis=0) * R * L / 2
    Fy = -_trapz(_trapz(P_dim * np.sin(Phi_mesh), phi_1D, axis=1),
                 Z_1D, axis=0) * R * L / 2
    return Fx, Fy


def compute_friction(P, p_scale, H, Phi_mesh, phi_1D, Z_1D,
                     eta, omega, R, L, c):
    """Сила трения."""
    P_dim = P * p_scale
    h_dim = H * c
    tau_couette = eta * omega * R / h_dim
    dP_dphi = np.gradient(P_dim, phi_1D[1] - phi_1D[0], axis=1)
    tau_pressure = h_dim / 2.0 * dP_dphi / R
    tau = tau_couette + tau_pressure
    F_friction = _trapz(_trapz(np.abs(tau), phi_1D, axis=1),
                        Z_1D, axis=0) * R * L / 2
    return F_friction


def get_step_deg(phi_deg, *, d_phi_base_deg: Optional[float] = None,
                  d_phi_peak_deg: float = 0.25,
                  peak_lo_deg: float = 330.0,
                  peak_hi_deg: float = 480.0):
    """Адаптивный шаг: ``d_phi_peak_deg`` у пика Вибе, ``d_phi_base_deg``
    вне. Дефолт ``d_phi_base_deg=params.d_phi_crank_deg`` сохраняет
    legacy behaviour.

    Stage Diesel Transient PeakWindow GridDiagnostic — default
    ``peak_hi_deg`` raised from 420° to 480° so the production
    metrics window 340°-480° is fully covered by the fine step.
    Recovery angles (down to ε < 0.80 typically at φ ≈ 430°-470°)
    used to be sampled at the coarse 1° base step which under-
    resolved the orbit and the post-peak SOR.
    """
    phi_mod = phi_deg % 720.0
    base = (float(d_phi_base_deg)
             if d_phi_base_deg is not None
             else float(params.d_phi_crank_deg))
    if peak_lo_deg <= phi_mod <= peak_hi_deg:
        return float(d_phi_peak_deg)
    return base


def texture_resolution_diagnostic(
    N_phi: int, N_z: int,
    R: float, L: float,
    a_dim: float, b_dim: float,
) -> Dict[str, Any]:
    """Diagnose whether a (Nφ × N_Z) grid resolves the elliptical
    texture pocket. ``a_dim`` and ``b_dim`` are physical axial and
    circumferential semi-axes (meters), matching ``DieselParams``.
    Pocket full angular width = 2·b_dim/R rad; full non-dim axial
    width = 4·a_dim/L (since physical z = (L/2)·Z so Z ∈ [-1, +1]).
    Returns cells per pocket in each direction, a categorical
    ``resolution_status`` ('ok' / 'marginal' / 'insufficient'), and
    the minimum Nφ at which cells_per_pocket_phi reaches 4.
    """
    N_phi = int(N_phi)
    N_z = int(N_z)
    cells_per_pocket_phi = float(N_phi) * float(b_dim) / (np.pi * float(R))
    cells_per_pocket_z = 2.0 * float(N_z) * float(a_dim) / float(L)
    if cells_per_pocket_phi >= 6.0:
        status = "ok"
    elif cells_per_pocket_phi >= 4.0:
        status = "marginal"
    else:
        status = "insufficient"
    recommended_n_phi_min = int(np.ceil(4.0 * np.pi * float(R) / float(b_dim)))
    return {
        "N_phi": N_phi,
        "N_z": N_z,
        "cells_per_pocket_phi": cells_per_pocket_phi,
        "cells_per_pocket_z": cells_per_pocket_z,
        "resolution_status": status,
        "recommended_n_phi_min": recommended_n_phi_min,
    }


def _solve_dynamic_with_retry(
    H, d_phi, d_Z, R, L,
    *,
    base_kw: Dict[str, Any],
    p_scale: float,
    Phi_mesh, phi_1D, Z_1D,
    retry_config: Optional[SolverRetryConfig],
    textured: bool,
):
    """Wrap solve_reynolds with SOR-warning capture, sanity check, and
    omega retry policy (Stage Texture Stability 2 reused).

    Returns ``(P, Fx_hyd, Fy_hyd, n_outer, ok, reason)``.

    On success ``ok=True`` and ``reason`` ∈ {"ok",
    "ok_retry_omega_<X>"}. On failure ``ok=False`` and ``reason``
    chains the failure reasons of every attempt; callers should treat
    the angle as ``solver_failed`` and reset their warm-start.
    """
    def _attempt(kw_in):
        sor_diverged = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = solve_reynolds(H, d_phi, d_Z, R, L, **kw_in)
            except Exception as exc:
                return (None, float("nan"), float("nan"), 0, False,
                        f"exception:{type(exc).__name__}:{exc}")
            for w in caught:
                if _SOR_WARN_FRAGMENT in str(w.message):
                    sor_diverged = True
                    break
        # solve_reynolds may return (P, residual, n_iter, converged) or
        # (P, residual, n_iter) (older builds without return_converged).
        P_ = res[0]
        n_outer_ = res[2] if len(res) >= 3 else 0
        converged = res[3] if len(res) >= 4 else (not sor_diverged)
        if sor_diverged or not bool(converged):
            return (P_, float("nan"), float("nan"), n_outer_, False,
                    "SOR_did_not_converge")
        # Sanity must be done on the DIMENSIONAL pressure (P * p_scale)
        # — bare P is non-dimensional and always small. Without this
        # rescale the SOR-divergent textured run can sneak past the
        # P-magnitude guard with p_max ~ 400 GPa.
        if not _solver_result_is_sane(0.0, P_ * float(p_scale)):
            return (P_, float("nan"), float("nan"), n_outer_, False,
                    f"non_physical_P:max|P_dim|="
                    f"{float(np.max(np.abs(P_) * float(p_scale))):.3e}")
        Fx, Fy = compute_hydro_forces(
            P_, p_scale, Phi_mesh, phi_1D, Z_1D, R, L)
        if not (np.isfinite(Fx) and np.isfinite(Fy)):
            return (P_, float("nan"), float("nan"), n_outer_, False,
                    "non_physical_F:nan")
        F_mag = float(np.hypot(Fx, Fy))
        if F_mag > _W_SANITY_LIMIT_N:
            return (P_, float("nan"), float("nan"), n_outer_, False,
                    f"non_physical_F:|F|={F_mag:.3e}")
        return (P_, float(Fx), float(Fy), int(n_outer_), True, "ok")

    primary = _attempt(base_kw)
    if primary[4]:
        return primary

    if retry_config is None or not retry_config.applicable(textured):
        return primary

    fail_chain = [f"primary:{primary[5]}"]
    P_init_retry = (None if retry_config.cold_start
                     else base_kw.get("P_init"))
    for omega in retry_config.omega_values:
        kw_retry = dict(base_kw)
        kw_retry["P_init"] = P_init_retry
        kw_retry["omega"] = float(omega)
        kw_retry["max_iter"] = int(retry_config.max_iter_retry)
        attempt = _attempt(kw_retry)
        if attempt[4]:
            return (*attempt[:5],
                     f"ok_retry_{_omega_tag(omega)}")
        fail_chain.append(
            f"retry_{_omega_tag(omega)}:{attempt[5]}")
    out = list(primary)
    out[5] = ";".join(fail_chain)
    return tuple(out)


# ─── Stage Diesel Transient Production Metrics ────────────────────

# Default firing sector for the BelAZ-class load profile (peak at
# φ ≈ 370°). 140° wide, captures both the buildup and the decay.
DEFAULT_FIRING_SECTOR_DEG: Tuple[float, float] = (340.0, 480.0)

# Recovery thresholds reported per config: how far past max-ε does the
# orbit need to travel before |ε| drops below each level. nan when the
# orbit never recovers below the level on the last cycle.
_RECOVERY_LEVELS = (0.90, 0.85, 0.80)
_HMIN_THRESHOLDS_M = (10e-6, 8e-6, 6e-6)
_AUC_RANGE_DEG = (360.0, 480.0)


def _firing_mask(phi_mod: np.ndarray,
                  firing_sector_deg: Tuple[float, float]) -> np.ndarray:
    """Bool mask of crank-angle steps falling inside the firing sector
    (handling the [-360, 1080] wrap explicitly)."""
    lo, hi = firing_sector_deg
    if lo <= hi:
        return (phi_mod >= lo) & (phi_mod <= hi)
    # Wrapped sector — e.g. (700, 50) — split into two bands.
    return (phi_mod >= lo) | (phi_mod <= hi)


def _percentile(arr: np.ndarray, q: float) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def _safe_min(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.min()) if a.size else float("nan")


def _safe_max(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else float("nan")


def _safe_mean(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _recovery_angle_deg(phi_mod: np.ndarray, eps: np.ndarray,
                         level: float) -> Tuple[float, bool]:
    """Distance in degrees from the per-cycle max-ε crank angle to the
    first subsequent step where ``|ε| < level``.

    Returns ``(angle_deg_or_nan, recovered)``.
    """
    if eps.size == 0:
        return float("nan"), False
    i_peak = int(np.argmax(eps))
    if not np.isfinite(eps[i_peak]):
        return float("nan"), False
    # Search forward from the peak; on the last cycle ``phi_mod`` is
    # nominally monotone so cumulative-from-peak is fine.
    after = eps[i_peak:]
    phi_after = phi_mod[i_peak:]
    drop = np.where(np.isfinite(after) & (after < float(level)))[0]
    if drop.size == 0:
        return float("nan"), False
    j = int(drop[0])
    return float(phi_after[j] - phi_after[0]), True


def _auc_eps_over_range(phi_mod: np.ndarray, eps: np.ndarray,
                         lo_deg: float, hi_deg: float) -> float:
    """∫|ε|(φ) dφ over [lo, hi] using trapezoid on the available
    samples. Returns ``nan`` if fewer than two finite points fall in
    the window."""
    mask = (phi_mod >= float(lo_deg)) & (phi_mod <= float(hi_deg))
    mask &= np.isfinite(eps)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(_trapz(eps[mask], phi_mod[mask]))


def _compute_production_metrics(*,
                                 cfg_list,
                                 last_start: int,
                                 phi_crank_deg: np.ndarray,
                                 eps_x_all: np.ndarray,
                                 eps_y_all: np.ndarray,
                                 hmin_all: np.ndarray,
                                 pmax_all: np.ndarray,
                                 P_loss_all: np.ndarray,
                                 valid_dynamic_all: np.ndarray,
                                 valid_no_clamp_all: np.ndarray,
                                 omega_rad_s: float,
                                 firing_sector_deg: Tuple[float, float],
                                 ) -> List[Dict[str, Any]]:
    """Per-config last-cycle production metrics.

    All metrics use the basic ``valid_no_clamp`` mask except recovery /
    AUC / max-ε which use ``valid_dynamic`` (so the contact-clamp
    boundary doesn't drown out the orbital signal). Returns a list of
    dicts, one per config.
    """
    out: List[Dict[str, Any]] = []
    sl = slice(int(last_start), None)
    phi_last = np.asarray(phi_crank_deg)[sl]
    phi_mod = phi_last % 720.0
    firing_mask = _firing_mask(phi_mod, firing_sector_deg)
    auc_lo, auc_hi = _AUC_RANGE_DEG
    for ic, _cfg in enumerate(cfg_list):
        vd = np.asarray(valid_dynamic_all[ic, sl], dtype=bool)
        vnc = np.asarray(valid_no_clamp_all[ic, sl], dtype=bool)
        ex = np.asarray(eps_x_all[ic, sl], dtype=float)
        ey = np.asarray(eps_y_all[ic, sl], dtype=float)
        eps_mag = np.sqrt(ex * ex + ey * ey)
        hmin = np.asarray(hmin_all[ic, sl], dtype=float)
        pmax = np.asarray(pmax_all[ic, sl], dtype=float)
        Plo = np.asarray(P_loss_all[ic, sl], dtype=float)

        # Pressure peak metrics — firing sector + valid_no_clamp.
        fmask = firing_mask & vnc & np.isfinite(pmax)
        rec: Dict[str, Any] = {
            "pmax_firing_p95":  _percentile(pmax[fmask], 95),
            "pmax_firing_p99":  _percentile(pmax[fmask], 99),
            "pmax_firing_max":  _safe_max(pmax[fmask]),
            "pmax_firing_count": int(fmask.sum()),
        }

        # h_min thresholds — last cycle, valid_no_clamp.
        hmask = vnc & np.isfinite(hmin)
        hmin_v = hmin[hmask]
        rec["hmin_p5"] = _percentile(hmin_v, 5)
        rec["hmin_min"] = _safe_min(hmin_v)
        # Per-step belows + angle equivalent (deg). The angle uses
        # the actual local Δφ (last cycle phi_mod is monotone for
        # adaptive grid) to weight each below-threshold step.
        if hmin_v.size > 0:
            phi_h = phi_mod[hmask]
            # local step length: forward difference (last entry uses
            # the same as previous to keep array length).
            if phi_h.size >= 2:
                d_phi_local = np.diff(phi_h)
                d_phi_local = np.append(
                    d_phi_local, d_phi_local[-1])
            else:
                d_phi_local = np.zeros_like(phi_h)
            for thr_m, key_steps, key_angle in (
                (10e-6, "steps_hmin_below_10um",
                 "angle_hmin_below_10um"),
                (8e-6, "steps_hmin_below_8um",
                 "angle_hmin_below_8um"),
                (6e-6, "steps_hmin_below_6um",
                 "angle_hmin_below_6um"),
            ):
                m = hmin_v < thr_m
                rec[key_steps] = int(m.sum())
                rec[key_angle] = float(np.sum(d_phi_local[m]))
        else:
            for key_steps, key_angle in (
                ("steps_hmin_below_10um", "angle_hmin_below_10um"),
                ("steps_hmin_below_8um",  "angle_hmin_below_8um"),
                ("steps_hmin_below_6um",  "angle_hmin_below_6um"),
            ):
                rec[key_steps] = 0
                rec[key_angle] = 0.0

        # Eccentricity peak (valid_dynamic — orbit even at clamp).
        emask = vd & np.isfinite(eps_mag)
        if int(emask.sum()) > 0:
            eps_v = eps_mag[emask]
            phi_v = phi_mod[emask]
            i_peak = int(np.argmax(eps_v))
            rec["max_eps_lastcycle"] = float(eps_v[i_peak])
            rec["phi_at_max_eps"] = float(phi_v[i_peak])
        else:
            rec["max_eps_lastcycle"] = float("nan")
            rec["phi_at_max_eps"] = float("nan")

        # ε at φ=421° (firing-peak exit) — interpolated from the last
        # cycle. Use linear interpolation against phi_mod sorted view;
        # if 421° falls outside the available range return nan.
        phi_target = 421.0
        finite_e = np.isfinite(eps_mag) & vd
        if finite_e.any():
            phi_v = phi_mod[finite_e]
            eps_v = eps_mag[finite_e]
            order = np.argsort(phi_v)
            phi_s = phi_v[order]
            eps_s = eps_v[order]
            if (phi_target >= phi_s[0]) and (phi_target <= phi_s[-1]):
                rec["eps_at_phi_421"] = float(
                    np.interp(phi_target, phi_s, eps_s))
            else:
                rec["eps_at_phi_421"] = float("nan")
        else:
            rec["eps_at_phi_421"] = float("nan")

        # Recovery angles to 0.90 / 0.85 / 0.80.
        if int(emask.sum()) > 0:
            phi_v = phi_mod[emask]
            eps_v = eps_mag[emask]
            for level in _RECOVERY_LEVELS:
                ang, ok = _recovery_angle_deg(phi_v, eps_v, level)
                key = f"angle_recovery_to_{str(level).replace('.', 'p')}"
                key_failed = f"recovery_failed_{str(level).replace('.', 'p')}"
                rec[key] = ang
                rec[key_failed] = bool(not ok)
        else:
            for level in _RECOVERY_LEVELS:
                rec[f"angle_recovery_to_{str(level).replace('.', 'p')}"] = (
                    float("nan"))
                rec[f"recovery_failed_{str(level).replace('.', 'p')}"] = True

        # AUC ε on [360°, 480°].
        rec["auc_eps_360_480"] = _auc_eps_over_range(
            phi_mod[emask] if emask.any() else phi_mod,
            eps_mag[emask] if emask.any() else eps_mag,
            auc_lo, auc_hi)

        # Power loss in firing sector — impulse over time.
        fmask_pl = firing_mask & vnc & np.isfinite(Plo)
        if int(fmask_pl.sum()) >= 2:
            phi_pl = phi_mod[fmask_pl]
            P_pl = Plo[fmask_pl]
            order = np.argsort(phi_pl)
            phi_s = phi_pl[order]
            P_s = P_pl[order]
            t_s = np.deg2rad(phi_s) / float(omega_rad_s)
            rec["ploss_impulse_firing_J"] = float(_trapz(P_s, t_s))
            rec["ploss_firing_mean_W"] = _safe_mean(P_pl)
            rec["ploss_firing_max_W"] = _safe_max(P_pl)
        else:
            rec["ploss_impulse_firing_J"] = float("nan")
            rec["ploss_firing_mean_W"] = float("nan")
            rec["ploss_firing_max_W"] = float("nan")

        rec["firing_sector_deg"] = (float(firing_sector_deg[0]),
                                      float(firing_sector_deg[1]))
        out.append(rec)
    return out


def _compute_paired_extended(cfg_list, paired_basic, prod_metrics,
                               last_start, valid_no_clamp_all,
                               valid_dynamic_all,
                               eps_x_all, eps_y_all,
                               hmin_all, pmax_all, P_loss_all,
                               omega_rad_s,
                               firing_sector_deg,
                               phi_crank_deg
                               ) -> List[Dict[str, Any]]:
    """Per-pair (smooth vs textured) deltas of the extended metrics
    on the **common_valid_no_clamp** mask of the last cycle.

    Stays consistent with ``_compute_paired_transient`` for the basic
    block — extended metrics are computed only on common_valid_no_clamp
    (or common_valid_dynamic for orbit-related entries) so the deltas
    are always paired on the same set of crank-angle steps.
    """
    by_oil: Dict[str, Dict[str, int]] = {}
    for ic, cfg in enumerate(cfg_list):
        oil_key = (cfg.get("oil") or {}).get("name", "")
        bucket = by_oil.setdefault(oil_key, {})
        if cfg.get("textured"):
            bucket["textured_idx"] = ic
        else:
            bucket["smooth_idx"] = ic
    sl = slice(int(last_start), None)
    phi_last = np.asarray(phi_crank_deg)[sl]
    phi_mod = phi_last % 720.0
    firing_mask = _firing_mask(phi_mod, firing_sector_deg)
    auc_lo, auc_hi = _AUC_RANGE_DEG
    out = []
    for oil_key, bucket in by_oil.items():
        if "textured_idx" not in bucket or "smooth_idx" not in bucket:
            continue
        i_s = bucket["smooth_idx"]
        i_t = bucket["textured_idx"]
        common_dyn = (np.asarray(valid_dynamic_all[i_s, sl], dtype=bool)
                       & np.asarray(valid_dynamic_all[i_t, sl], dtype=bool))
        common_noc = (np.asarray(valid_no_clamp_all[i_s, sl], dtype=bool)
                       & np.asarray(valid_no_clamp_all[i_t, sl], dtype=bool))
        rec = dict(
            oil_name=oil_key,
            smooth_idx=i_s, smooth_label=cfg_list[i_s]["label"],
            textured_idx=i_t, textured_label=cfg_list[i_t]["label"],
            common_valid_no_clamp_count=int(common_noc.sum()),
            common_valid_dynamic_count=int(common_dyn.sum()),
        )
        # Pressure-peak deltas — on common_valid_no_clamp ∩ firing.
        f_noc = common_noc & firing_mask
        for q, key in ((95, "pmax_firing_p95"),
                        (99, "pmax_firing_p99")):
            ps = np.asarray(pmax_all[i_s, sl])[f_noc]
            pt = np.asarray(pmax_all[i_t, sl])[f_noc]
            rec[f"smooth_{key}"] = _percentile(ps, q)
            rec[f"textured_{key}"] = _percentile(pt, q)
            rec[f"delta_{key}"] = (rec[f"textured_{key}"]
                                      - rec[f"smooth_{key}"])
        # max p_max in firing
        ps = np.asarray(pmax_all[i_s, sl])[f_noc]
        pt = np.asarray(pmax_all[i_t, sl])[f_noc]
        rec["smooth_pmax_firing_max"] = _safe_max(ps)
        rec["textured_pmax_firing_max"] = _safe_max(pt)
        rec["delta_pmax_firing_max"] = (
            rec["textured_pmax_firing_max"]
            - rec["smooth_pmax_firing_max"])

        # h_min P5 / min on common_no_clamp
        hs = np.asarray(hmin_all[i_s, sl])[common_noc]
        ht = np.asarray(hmin_all[i_t, sl])[common_noc]
        rec["smooth_hmin_p5"] = _percentile(hs, 5)
        rec["textured_hmin_p5"] = _percentile(ht, 5)
        rec["delta_hmin_p5"] = (rec["textured_hmin_p5"]
                                   - rec["smooth_hmin_p5"])
        rec["smooth_hmin_min"] = _safe_min(hs)
        rec["textured_hmin_min"] = _safe_min(ht)
        rec["delta_hmin_min"] = (rec["textured_hmin_min"]
                                    - rec["smooth_hmin_min"])
        # Steps below thresholds (common_no_clamp, both finite).
        hsv = hs[np.isfinite(hs)]
        htv = ht[np.isfinite(ht)]
        for thr_m, key in ((10e-6, "steps_hmin_below_10um"),
                             (8e-6, "steps_hmin_below_8um"),
                             (6e-6, "steps_hmin_below_6um")):
            ns = int(np.sum(hsv < thr_m))
            nt = int(np.sum(htv < thr_m))
            rec[f"smooth_{key}"] = ns
            rec[f"textured_{key}"] = nt
            rec[f"delta_{key}"] = nt - ns

        # Orbit metrics use common_valid_dynamic.
        ex_s = np.asarray(eps_x_all[i_s, sl])[common_dyn]
        ey_s = np.asarray(eps_y_all[i_s, sl])[common_dyn]
        ex_t = np.asarray(eps_x_all[i_t, sl])[common_dyn]
        ey_t = np.asarray(eps_y_all[i_t, sl])[common_dyn]
        eps_s = np.sqrt(ex_s ** 2 + ey_s ** 2)
        eps_t = np.sqrt(ex_t ** 2 + ey_t ** 2)
        rec["smooth_max_eps_lastcycle"] = _safe_max(eps_s)
        rec["textured_max_eps_lastcycle"] = _safe_max(eps_t)
        rec["delta_max_eps_lastcycle"] = (
            rec["textured_max_eps_lastcycle"]
            - rec["smooth_max_eps_lastcycle"])
        # ε at 421° via interp on common.
        phi_common = phi_mod[common_dyn]
        if phi_common.size >= 2:
            order = np.argsort(phi_common)
            phi_s_ = phi_common[order]
            if 421.0 >= phi_s_[0] and 421.0 <= phi_s_[-1]:
                rec["smooth_eps_at_phi_421"] = float(np.interp(
                    421.0, phi_s_, eps_s[order]))
                rec["textured_eps_at_phi_421"] = float(np.interp(
                    421.0, phi_s_, eps_t[order]))
            else:
                rec["smooth_eps_at_phi_421"] = float("nan")
                rec["textured_eps_at_phi_421"] = float("nan")
        else:
            rec["smooth_eps_at_phi_421"] = float("nan")
            rec["textured_eps_at_phi_421"] = float("nan")
        rec["delta_eps_at_phi_421"] = (
            rec["textured_eps_at_phi_421"]
            - rec["smooth_eps_at_phi_421"])
        rec["smooth_auc_eps_360_480"] = _auc_eps_over_range(
            phi_common, eps_s, auc_lo, auc_hi)
        rec["textured_auc_eps_360_480"] = _auc_eps_over_range(
            phi_common, eps_t, auc_lo, auc_hi)
        rec["delta_auc_eps_360_480"] = (
            rec["textured_auc_eps_360_480"]
            - rec["smooth_auc_eps_360_480"])

        # P_loss impulse / mean on common_no_clamp ∩ firing.
        f_noc_pl = common_noc & firing_mask
        Ps = np.asarray(P_loss_all[i_s, sl])[f_noc_pl]
        Pt = np.asarray(P_loss_all[i_t, sl])[f_noc_pl]
        phi_pl = phi_mod[f_noc_pl]
        if int(f_noc_pl.sum()) >= 2:
            order = np.argsort(phi_pl)
            t_s = np.deg2rad(phi_pl[order]) / float(omega_rad_s)
            rec["smooth_ploss_impulse_firing_J"] = float(_trapz(
                Ps[order], t_s))
            rec["textured_ploss_impulse_firing_J"] = float(_trapz(
                Pt[order], t_s))
        else:
            rec["smooth_ploss_impulse_firing_J"] = float("nan")
            rec["textured_ploss_impulse_firing_J"] = float("nan")
        rec["delta_ploss_impulse_firing_J"] = (
            rec["textured_ploss_impulse_firing_J"]
            - rec["smooth_ploss_impulse_firing_J"])
        rec["smooth_ploss_firing_mean_W"] = _safe_mean(Ps)
        rec["textured_ploss_firing_mean_W"] = _safe_mean(Pt)
        rec["delta_ploss_firing_mean_W"] = (
            rec["textured_ploss_firing_mean_W"]
            - rec["smooth_ploss_firing_mean_W"])

        rec["firing_sector_deg"] = (float(firing_sector_deg[0]),
                                      float(firing_sector_deg[1]))
        out.append(rec)
    return out


def _global_phi_for_paired(eps_x_all, eps_y_all, last_start):
    """phi_crank_deg slice we need is shared across configs; the
    paired helper is given the last-cycle slice indirectly via the
    array shapes. The runner passes in ``phi_crank_deg`` separately
    elsewhere; here we just need the last-cycle phi which is the
    second axis index. We rely on the runner already storing
    phi_crank_deg in ``results``; this helper is only used internally
    when we don't have that handle yet, so it computes a 0-based
    crank angle from the second-axis size. Callers in
    ``run_transient`` pass the real phi_crank_deg directly.
    """
    # Defensive fallback only — real callers pass phi_crank_deg.
    n = eps_x_all.shape[1] - int(last_start)
    return np.linspace(0.0, 720.0, max(n, 1), endpoint=False)


def _compute_paired_transient(cfg_list, last_start, valid_dynamic,
                               valid_no_clamp, T_eff_used, eta_eff,
                               P_loss, hmin, pmax, F_tr):
    """Stage Diesel Transient THD-0: paired smooth-vs-textured stats
    on common_valid_dynamic AND common_valid_no_clamp masks for the
    last cycle only.

    Mirrors the quasistatic ``_compute_paired`` but uses time-step
    arrays sliced to the last cycle and the dynamic validity gates.
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
    sl = slice(last_start, None)
    for oil_key, bucket in by_oil.items():
        if "textured_idx" not in bucket or "smooth_idx" not in bucket:
            continue
        i_s, i_t = bucket["smooth_idx"], bucket["textured_idx"]
        common_dyn = (np.asarray(valid_dynamic[i_s, sl], dtype=bool)
                       & np.asarray(valid_dynamic[i_t, sl], dtype=bool))
        common_noc = (np.asarray(valid_no_clamp[i_s, sl], dtype=bool)
                       & np.asarray(valid_no_clamp[i_t, sl], dtype=bool))
        rec = dict(
            oil_name=oil_key,
            smooth_idx=i_s, smooth_label=cfg_list[i_s]["label"],
            textured_idx=i_t, textured_label=cfg_list[i_t]["label"],
            common_valid_count=int(common_dyn.sum()),
            common_no_clamp_count=int(common_noc.sum()),
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
        if common_noc.any():
            T_s = T_eff_used[i_s, sl][common_noc]
            T_t = T_eff_used[i_t, sl][common_noc]
            P_s = P_loss[i_s, sl][common_noc]
            P_t = P_loss[i_t, sl][common_noc]
            e_s = eta_eff[i_s, sl][common_noc]
            e_t = eta_eff[i_t, sl][common_noc]
            h_s = hmin[i_s, sl][common_noc]
            h_t = hmin[i_t, sl][common_noc]
            pmx_s = pmax[i_s, sl][common_noc]
            pmx_t = pmax[i_t, sl][common_noc]
            rec["mean_dT_eff"] = float(np.mean(T_t - T_s))
            rec["mean_dP_loss"] = float(np.mean(P_t - P_s))
            rec["mean_deta_eff"] = float(np.mean(e_t - e_s))
            rec["mean_dh_min"] = float(np.mean(h_t - h_s))
            rec["min_dp_max"] = float(np.min(pmx_t - pmx_s))
            rec["max_dp_max"] = float(np.max(pmx_t - pmx_s))
            rec["mean_T_smooth"] = float(np.mean(T_s))
            rec["mean_T_textured"] = float(np.mean(T_t))
            rec["mean_P_loss_smooth"] = float(np.mean(P_s))
            rec["mean_P_loss_textured"] = float(np.mean(P_t))
        out.append(rec)
    return out


def run_transient(F_max=None, debug=False,
                  closure=DEFAULT_CLOSURE, cavitation=DEFAULT_CAVITATION,
                  *,
                  thermal: Optional[ThermalConfig] = None,
                  configs=None,
                  n_grid: Optional[int] = None,
                  n_cycles: Optional[int] = None,
                  d_phi_base_deg: Optional[float] = None,
                  d_phi_peak_deg: float = 0.25,
                  retry_config: Optional[SolverRetryConfig] = None,
                  envelope_abort: Optional[EnvelopeAbortConfig] = None,
                  firing_sector_deg: Optional[Tuple[float, float]] = None,
                  peak_lo_deg: float = 330.0,
                  peak_hi_deg: float = 480.0,
                  n_phi_grid: Optional[int] = None,
                  n_z_grid: Optional[int] = None):
    """Нестационарный расчёт.

    Stage Diesel Transient THD-0 — see module docstring. ``thermal=None``
    or ``mode="off"`` reproduces the legacy isothermal transient
    bit-for-bit. ``thermal.mode="global_static"`` uses the static
    energy target on every step; ``thermal.mode="global_relax"`` carries
    a thermal state across steps with first-order relaxation.

    Parameters
    ----------
    thermal : ThermalConfig | None
        ``None`` ⇒ legacy mode="off".
    configs : list[dict] | None
        Optional override of CONFIGS (e.g. one or two configs only).
    n_grid : int | None
        Override for ``params.N_grid_transient``.
    n_cycles : int | None
        Override for ``params.n_cycles``.
    d_phi_base_deg : float | None
        Base crank-angle step (deg). ``None`` keeps
        ``params.d_phi_crank_deg``.
    d_phi_peak_deg : float
        Adaptive step at the firing peak.
    retry_config : SolverRetryConfig | None
        Solver retry policy. ``None`` ⇒ default
        ``SolverRetryConfig()`` (textured-only, omegas (1.70, 1.55),
        cold_start=True), matching Stage Texture Stability 2.
    """
    if F_max is None:
        F_max = params.F_max_debug if debug else params.F_max
    if thermal is None:
        thermal = ThermalConfig(mode="off")
    is_off = thermal.is_off()
    cfg_list = list(configs) if configs is not None else CONFIGS
    n_cycles_eff = int(n_cycles) if n_cycles is not None else int(
        params.n_cycles)
    if retry_config is None:
        retry_config = SolverRetryConfig()
    if envelope_abort is None:
        envelope_abort = EnvelopeAbortConfig()
    if firing_sector_deg is None:
        firing_sector_deg = DEFAULT_FIRING_SECTOR_DEG
    firing_sector_deg = (float(firing_sector_deg[0]),
                          float(firing_sector_deg[1]))

    # Anisotropic grid: legacy isotropic ``n_grid`` still works when
    # neither ``n_phi_grid`` nor ``n_z_grid`` is supplied; otherwise
    # the explicit (Nφ, Nz) win. setup_grid already accepts (N_phi,
    # N_Z) so no solver-side change is required.
    N_legacy = int(n_grid) if n_grid is not None else int(
        params.N_grid_transient)
    N_phi_eff = int(n_phi_grid) if n_phi_grid is not None else N_legacy
    N_z_eff = int(n_z_grid) if n_z_grid is not None else N_legacy
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(
        N_phi_eff, N_z_eff)
    phi_c, Z_c = setup_texture(params)

    # Stage Diesel Transient PeakWindow GridDiagnostic — diagnose
    # whether the chosen Nφ × N_Z grid resolves the elliptical
    # texture pocket. The diagnostic depends only on grid + global
    # bearing geometry, so it is the same for every textured config.
    texture_res_diag = texture_resolution_diagnostic(
        N_phi_eff, N_z_eff,
        R=params.R, L=params.L,
        a_dim=params.a_dim, b_dim=params.b_dim,
    )
    if texture_res_diag["resolution_status"] == "insufficient":
        print(
            "  [WARN] texture pocket under-resolved: "
            f"cells_per_pocket_phi="
            f"{texture_res_diag['cells_per_pocket_phi']:.2f} < 4 "
            f"(N_phi={N_phi_eff}; recommend N_phi >= "
            f"{texture_res_diag['recommended_n_phi_min']})"
        )

    omega = 2 * np.pi * params.n / 60.0
    U = omega * params.R
    N_SUB = params.n_sub_iterations

    # Adaptive crank-angle grid.
    phi_list = []
    phi_cur = 0.0
    phi_total = 720.0 * n_cycles_eff
    while phi_cur < phi_total - 1e-9:
        phi_list.append(phi_cur)
        phi_cur += get_step_deg(
            phi_cur,
            d_phi_base_deg=d_phi_base_deg,
            d_phi_peak_deg=d_phi_peak_deg,
            peak_lo_deg=peak_lo_deg,
            peak_hi_deg=peak_hi_deg,
        )
    phi_crank_deg = np.array(phi_list)
    n_steps = len(phi_crank_deg)
    last_cycle_start_deg = 720.0 * (n_cycles_eff - 1)
    last_start = int(np.searchsorted(phi_crank_deg, last_cycle_start_deg))
    n_last = n_steps - last_start

    base_step = (float(d_phi_base_deg)
                  if d_phi_base_deg is not None
                  else float(params.d_phi_crank_deg))
    print(f"  Параметры: F_max={F_max/1e3:.0f} кН, "
          f"{n_steps} шагов (адаптивный: {d_phi_peak_deg}° пик / "
          f"{base_step}° вне), N_SUB={N_SUB}, {n_cycles_eff} цикла; "
          f"thermal mode={thermal.mode}, gamma={thermal.gamma_mix:.2f}, "
          f"tau_th={thermal.tau_th_s:.3f}s")

    n_cfg = len(cfg_list)
    eps_x_all = np.zeros((n_cfg, n_steps))
    eps_y_all = np.zeros((n_cfg, n_steps))
    hmin_all = np.full((n_cfg, n_steps), np.nan)
    pmax_all = np.full((n_cfg, n_steps), np.nan)
    f_all = np.full((n_cfg, n_steps), np.nan)
    Ftr_all = np.full((n_cfg, n_steps), np.nan)
    Nloss_all = np.full((n_cfg, n_steps), np.nan)
    Fx_hyd_all = np.full((n_cfg, n_steps), np.nan)
    Fy_hyd_all = np.full((n_cfg, n_steps), np.nan)
    # Stage Diesel Transient THD-0 — new arrays.
    T_eff_used_all = np.full((n_cfg, n_steps), np.nan)
    T_eff_all = np.full((n_cfg, n_steps), np.nan)
    T_target_all = np.full((n_cfg, n_steps), np.nan)
    eta_eff_all = np.full((n_cfg, n_steps), np.nan)
    eta_eff_next_all = np.full((n_cfg, n_steps), np.nan)
    P_loss_all = np.full((n_cfg, n_steps), np.nan)
    Q_all = np.full((n_cfg, n_steps), np.nan)
    mdot_all = np.full((n_cfg, n_steps), np.nan)
    mdot_floor_hit_all = np.zeros((n_cfg, n_steps), dtype=bool)
    solver_success_all = np.zeros((n_cfg, n_steps), dtype=bool)
    contact_clamp_all = np.zeros((n_cfg, n_steps), dtype=bool)
    retry_used_all = np.zeros((n_cfg, n_steps), dtype=bool)
    retry_omega_used_all = np.zeros((n_cfg, n_steps), dtype=float)
    cfg_times: List[float] = []
    contact_clamp_count = np.zeros(n_cfg, dtype=np.int32)
    solver_failed_count = np.zeros(n_cfg, dtype=np.int32)
    retry_recovered_count = np.zeros(n_cfg, dtype=np.int32)
    retry_exhausted_count = np.zeros(n_cfg, dtype=np.int32)
    omega_hits_per_cfg: List[Dict[str, int]] = [dict() for _ in range(n_cfg)]
    # Stage Load-Envelope-0: per-config abort/envelope diagnostics.
    aborted_arr = np.zeros(n_cfg, dtype=bool)
    abort_reason_arr = np.full(n_cfg, "", dtype="<U64")
    first_clamp_phi_arr = np.full(n_cfg, np.nan)
    first_solver_failed_phi_arr = np.full(n_cfg, np.nan)
    first_invalid_phi_arr = np.full(n_cfg, np.nan)
    steps_attempted_arr = np.zeros(n_cfg, dtype=np.int32)
    steps_completed_arr = np.zeros(n_cfg, dtype=np.int32)
    applicable_arr = np.zeros(n_cfg, dtype=bool)
    applicable_reason_arr = np.full(n_cfg, "", dtype="<U96")

    for ic, cfg in enumerate(cfg_list):
        eta_const = float(cfg["oil"]["eta_diesel"])
        rho = float(cfg["oil"]["rho"])

        if is_off:
            walther_fit = None
        else:
            walther_fit = build_oil_walther(
                cfg["oil"],
                cp_J_kgK=thermal.cp_J_kgK,
                gamma_mix=thermal.gamma_mix,
            )
        T_state_C = float(thermal.T_in_C)

        print(f"\n  [{ic+1}/{n_cfg}] {cfg['label']}...")
        t_cfg_start = _time.time()

        # Initial conditions.
        ex = params.eps_x0 * params.c
        ey = params.eps_y0 * params.c
        vx, vy = 0.0, 0.0
        ax_prev, ay_prev = 0.0, 0.0
        P_prev = None
        contact_count = 0

        def _clamp(ex_, ey_, vx_, vy_):
            clamped = False
            eps_mag_ = np.hypot(ex_, ey_) / params.c
            if eps_mag_ > params.eps_max:
                scale_ = params.eps_max / eps_mag_
                ex_ *= scale_
                ey_ *= scale_
                e_hat_x_ = ex_ / (eps_mag_ * params.c)
                e_hat_y_ = ey_ / (eps_mag_ * params.c)
                v_rad_ = vx_ * e_hat_x_ + vy_ * e_hat_y_
                if v_rad_ > 0:
                    vx_ -= v_rad_ * e_hat_x_
                    vy_ -= v_rad_ * e_hat_y_
                clamped = True
            return ex_, ey_, vx_, vy_, clamped

        progress_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            phi_deg = phi_crank_deg[step] % 720.0
            dt_step = np.deg2rad(get_step_deg(
                phi_crank_deg[step],
                d_phi_base_deg=d_phi_base_deg,
                d_phi_peak_deg=d_phi_peak_deg,
                peak_lo_deg=peak_lo_deg,
                peak_hi_deg=peak_hi_deg,
            )) / omega

            # Per-step thermal: η fixed for the whole Verlet substep.
            T_used_C = float(T_state_C if not is_off else thermal.T_in_C)
            if is_off or walther_fit is None:
                eta_step = eta_const
            else:
                eta_step = float(viscosity_at_T_C(walther_fit, T_used_C))
            p_scale_step = (
                6.0 * eta_step * omega * (params.R / params.c) ** 2)

            # Save state at start of step.
            ex_n, ey_n = ex, ey
            vx_n, vy_n = vx, vy

            # Initial Verlet predict.
            ex_pred = ex_n + vx_n * dt_step + 0.5 * ax_prev * dt_step**2
            ey_pred = ey_n + vy_n * dt_step + 0.5 * ay_prev * dt_step**2
            ex_pred, ey_pred, _, _, clamped_p = _clamp(
                ex_pred, ey_pred, vx_n, vy_n)
            if clamped_p:
                contact_count += 1
                P_prev = None

            vx_corr, vy_corr = vx_n, vy_n
            ax_new, ay_new = ax_prev, ay_prev
            P = None
            H = None
            Fx_hyd = float("nan")
            Fy_hyd = float("nan")
            solve_ok = False
            solve_reason = "no_attempt"
            retry_recovered_step = False
            omega_recovery: Optional[float] = None

            for k in range(N_SUB):
                eps_x_ = ex_pred / params.c
                eps_y_ = ey_pred / params.c
                H_ = build_H_2d(eps_x_, eps_y_, Phi_mesh, Z_mesh, params,
                                 textured=cfg["textured"],
                                 phi_c_flat=phi_c, Z_c_flat=Z_c)
                xp_, yp_, bt_ = squeeze_to_api_params(
                    -vx_corr, -vy_corr, params.c, omega, d_phi)
                base_kw = dict(
                    closure=closure, cavitation=cavitation,
                    tol=1e-5, max_iter=50000,
                    P_init=P_prev,
                    xprime=xp_, yprime=yp_, beta=bt_,
                )
                P_, Fx_, Fy_, _, ok_, reason_ = _solve_dynamic_with_retry(
                    H_, d_phi, d_Z, params.R, params.L,
                    base_kw=base_kw,
                    p_scale=p_scale_step,
                    Phi_mesh=Phi_mesh, phi_1D=phi_1D, Z_1D=Z_1D,
                    retry_config=retry_config,
                    textured=bool(cfg["textured"]),
                )
                outcome = _parse_retry_outcome(reason_)
                if outcome["retry_recovered"]:
                    retry_recovered_step = True
                    omega_recovery = outcome["retry_omega"]
                if not ok_:
                    P_prev = None
                    P = P_
                    H = H_
                    Fx_hyd = float("nan")
                    Fy_hyd = float("nan")
                    solve_ok = False
                    solve_reason = reason_
                    break
                P, H = P_, H_
                Fx_hyd, Fy_hyd = float(Fx_), float(Fy_)
                solve_ok = True
                solve_reason = reason_
                P_prev = P

                Fx_ext, Fy_ext = load_diesel(phi_deg, F_max=F_max)
                # load_diesel always returns (1,)-shape arrays for a
                # scalar phi; ``.item()`` extracts the scalar without
                # the np.float(array) DeprecationWarning.
                Fx_ext = float(np.asarray(Fx_ext).item())
                Fy_ext = float(np.asarray(Fy_ext).item())
                ax_new = (Fx_ext + Fx_hyd) / params.m_shaft
                ay_new = (Fy_ext + Fy_hyd) / params.m_shaft
                vx_corr = vx_n + 0.5 * (ax_prev + ax_new) * dt_step
                vy_corr = vy_n + 0.5 * (ay_prev + ay_new) * dt_step

                if k < N_SUB - 1:
                    ex_pred = (ex_n + vx_corr * dt_step
                                + 0.5 * ax_new * dt_step**2)
                    ey_pred = (ey_n + vy_corr * dt_step
                                + 0.5 * ay_new * dt_step**2)
                    ex_pred, ey_pred, vx_corr, vy_corr, cl = _clamp(
                        ex_pred, ey_pred, vx_corr, vy_corr)
                    if cl:
                        contact_count += 1
                        P_prev = None

            # Accept (even if solver failed — keep mechanical state
            # progressing without poisoned pressure).
            ex, ey = ex_pred, ey_pred
            vx, vy = vx_corr, vy_corr
            ax_prev, ay_prev = ax_new, ay_new

            ex, ey, vx, vy, clamped_final = _clamp(ex, ey, vx, vy)
            if clamped_final:
                contact_count += 1
                P_prev = None
            step_clamped = bool(clamped_p or clamped_final)

            # Per-step diagnostics.
            if solve_ok and P is not None and H is not None:
                h_dim = H * params.c
                h_min = float(np.min(h_dim))
                p_max = float(np.max(P * p_scale_step))
                F_friction = float(compute_friction(
                    P, p_scale_step, H, Phi_mesh, phi_1D, Z_1D,
                    eta_step, omega, params.R, params.L, params.c))
                F_hyd_mag = float(np.hypot(Fx_hyd, Fy_hyd))
                mu_val = (F_friction / max(F_hyd_mag, 1.0)
                            if F_hyd_mag > 1.0 else 0.0)
                P_loss_step = F_friction * U
                Q_step = compute_axial_leakage_m3_s(
                    P_dim=P * p_scale_step,
                    h_dim=h_dim,
                    phi_1D=phi_1D, Z_1D=Z_1D,
                    eta=eta_step, R=params.R, L=params.L,
                )
                mdot_raw = rho * abs(Q_step)
                mdot_step = max(mdot_raw, thermal.mdot_floor_kg_s)
                mdot_floor = bool(mdot_raw < thermal.mdot_floor_kg_s)
            else:
                solver_failed_count[ic] += 1
                h_min = float("nan")
                p_max = float("nan")
                F_friction = float("nan")
                mu_val = float("nan")
                P_loss_step = float("nan")
                Q_step = float("nan")
                mdot_step = float("nan")
                mdot_floor = False

            # Thermal update at the end of the mechanical step.
            T_target_C = float(thermal.T_in_C)
            if (not is_off) and solve_ok and np.isfinite(P_loss_step):
                T_target_C = float(global_static_step(
                    T_in_C=thermal.T_in_C,
                    P_loss_W=P_loss_step,
                    mdot_kg_s=mdot_step,
                    cp_J_kgK=thermal.cp_J_kgK,
                    gamma=thermal.gamma_mix,
                    model=walther_fit,
                ))
                if thermal.mode == "global_static":
                    T_state_C = T_target_C
                elif thermal.mode == "global_relax":
                    T_state_C = float(global_relax_step(
                        T_prev_C=T_state_C,
                        T_target_C=T_target_C,
                        dt_s=dt_step,
                        tau_th_s=thermal.tau_th_s,
                    ))
            elif is_off:
                T_state_C = float(thermal.T_in_C)
            # Else: solve failed AND not off → keep previous T_state.

            # Compute eta at the new T_state for diagnostics.
            if is_off or walther_fit is None:
                eta_next = eta_const
            else:
                eta_next = float(viscosity_at_T_C(walther_fit, T_state_C))

            eps_x_all[ic, step] = ex / params.c
            eps_y_all[ic, step] = ey / params.c
            hmin_all[ic, step] = h_min
            pmax_all[ic, step] = p_max
            f_all[ic, step] = mu_val
            Ftr_all[ic, step] = F_friction
            Nloss_all[ic, step] = P_loss_step
            Fx_hyd_all[ic, step] = Fx_hyd
            Fy_hyd_all[ic, step] = Fy_hyd
            T_eff_used_all[ic, step] = T_used_C
            T_eff_all[ic, step] = T_state_C
            T_target_all[ic, step] = T_target_C
            eta_eff_all[ic, step] = eta_step
            eta_eff_next_all[ic, step] = eta_next
            P_loss_all[ic, step] = P_loss_step
            Q_all[ic, step] = Q_step
            mdot_all[ic, step] = mdot_step
            mdot_floor_hit_all[ic, step] = mdot_floor
            solver_success_all[ic, step] = bool(solve_ok)
            contact_clamp_all[ic, step] = step_clamped
            retry_used_all[ic, step] = bool(retry_recovered_step)
            if retry_recovered_step:
                retry_recovered_count[ic] += 1
                retry_omega_used_all[ic, step] = float(omega_recovery or 0.0)
                if omega_recovery is not None:
                    tag = _omega_tag(omega_recovery)
                    omega_hits_per_cfg[ic][tag] = (
                        omega_hits_per_cfg[ic].get(tag, 0) + 1)
            if (not solve_ok) and ";" in solve_reason:
                # Chained reason ⇒ retry exhausted.
                retry_exhausted_count[ic] += 1

            if (step + 1) % progress_interval == 0:
                pct = 100 * (step + 1) / n_steps
                eps_now = np.hypot(ex, ey) / params.c
                Tref = (T_state_C if not is_off
                         else thermal.T_in_C)
                t_now = _time.time() - t_cfg_start
                # ETA assumes a constant rate from the start of this
                # config; firing peak slows it down, off-peak speeds up,
                # so ETA is a useful order-of-magnitude not a promise.
                eta_total = (t_now / max(pct, 1e-6) * 100.0
                              if pct > 0 else 0.0)
                eta_left = max(0.0, eta_total - t_now)
                msg = (f"    {pct:3.0f}%: φ={phi_deg:6.1f}°, "
                        f"|ε|={eps_now:.3f}, "
                        f"h_min={h_min*1e6 if np.isfinite(h_min) else float('nan'):.1f} мкм, "
                        f"p_max={p_max/1e6 if np.isfinite(p_max) else float('nan'):.1f} МПа, "
                        f"T_eff={Tref:.1f}°C, "
                        f"t={t_now:.0f}s, ETA={eta_left:.0f}s")
                print(msg, flush=True)

            # ── envelope diagnostics + abort policy (Section 1) ──
            if step_clamped and np.isnan(first_clamp_phi_arr[ic]):
                first_clamp_phi_arr[ic] = float(phi_deg)
            if (not solve_ok) and np.isnan(first_solver_failed_phi_arr[ic]):
                first_solver_failed_phi_arr[ic] = float(phi_deg)
            this_step_invalid = not bool(
                solver_success_all[ic, step]
                and not contact_clamp_all[ic, step]
                and np.isfinite(hmin_all[ic, step])
            )
            if this_step_invalid and np.isnan(first_invalid_phi_arr[ic]):
                first_invalid_phi_arr[ic] = float(phi_deg)
            if envelope_abort.enabled \
                    and (step + 1) >= envelope_abort.warmup_steps:
                attempted = step + 1
                clamp_so_far = int(np.sum(
                    contact_clamp_all[ic, : attempted]))
                fail_so_far = int(solver_failed_count[ic])
                clamp_frac = clamp_so_far / attempted
                fail_frac = fail_so_far / attempted
                # Streak of consecutive not-valid_dynamic steps from
                # the most recent step backward. Uses solver_success
                # AND finite-everything (mirrors valid_dynamic).
                consec = 0
                for k in range(step, -1, -1):
                    bad = not bool(
                        solver_success_all[ic, k]
                        and np.isfinite(hmin_all[ic, k])
                        and np.isfinite(pmax_all[ic, k])
                        and np.isfinite(Ftr_all[ic, k])
                    )
                    if bad:
                        consec += 1
                    else:
                        break
                abort_reason = ""
                if clamp_frac > envelope_abort.clamp_frac_max:
                    abort_reason = "clamp_frac_exceeded"
                elif fail_frac > envelope_abort.solver_fail_frac_max:
                    abort_reason = "solver_fail_frac_exceeded"
                elif consec >= envelope_abort.consecutive_invalid_max:
                    abort_reason = "consecutive_invalid_exceeded"
                if abort_reason:
                    aborted_arr[ic] = True
                    abort_reason_arr[ic] = abort_reason
                    steps_attempted_arr[ic] = attempted
                    steps_completed_arr[ic] = attempted
                    print(f"    [abort] config={cfg['label']!r} "
                          f"reason={abort_reason} "
                          f"step={attempted}/{n_steps} "
                          f"phi={phi_deg:.1f}° "
                          f"(clamp_frac={clamp_frac:.2f}, "
                          f"fail_frac={fail_frac:.2f}, "
                          f"consec_invalid={consec}); "
                          f"writing partial result and continuing",
                          flush=True)
                    break

        # If we did not abort, the natural loop end means we ran every
        # step. Otherwise steps_attempted/_completed were already set
        # above by the abort branch.
        if not aborted_arr[ic]:
            steps_attempted_arr[ic] = n_steps
            steps_completed_arr[ic] = n_steps

        contact_clamp_count[ic] = contact_count
        t_cfg = _time.time() - t_cfg_start
        cfg_times.append(t_cfg)
        n_solver_fail = int(solver_failed_count[ic])
        n_retry_rec = int(retry_recovered_count[ic])
        n_retry_exh = int(retry_exhausted_count[ic])
        print(f"    Контакт (clamp): {contact_count} / {n_steps} шагов")
        print(f"    Solver: failed={n_solver_fail}, "
              f"retry_recovered={n_retry_rec}, "
              f"retry_exhausted={n_retry_exh}")
        if not is_off:
            T_eff_last = T_eff_all[ic, last_start:]
            finite_T = T_eff_last[np.isfinite(T_eff_last)]
            if finite_T.size > 0:
                print(f"    T_eff last cycle: "
                      f"min/mean/max = {finite_T.min():.1f}/"
                      f"{finite_T.mean():.1f}/{finite_T.max():.1f} °C")
        print(f"    Время: {t_cfg:.1f} с")

    # Validity gates.
    finite_mask = (
        np.isfinite(hmin_all)
        & np.isfinite(pmax_all)
        & np.isfinite(Ftr_all)
        & np.isfinite(Nloss_all)
        & np.isfinite(Q_all)
        & np.isfinite(mdot_all)
        & np.isfinite(T_eff_all)
        & (eta_eff_all > 0)
    )
    valid_dynamic_all = solver_success_all & finite_mask
    valid_no_clamp_all = valid_dynamic_all & ~contact_clamp_all

    # Thermal periodic convergence: compare last cycle to previous on
    # the matching adaptive grid (same step count by construction).
    thermal_cycle_delta = np.full(n_cfg, np.nan)
    thermal_periodic_converged = np.zeros(n_cfg, dtype=bool)
    if n_cycles_eff >= 2 and not is_off:
        prev_cycle_start_deg = 720.0 * (n_cycles_eff - 2)
        prev_start = int(np.searchsorted(phi_crank_deg,
                                            prev_cycle_start_deg))
        prev_n = last_start - prev_start
        comp_n = min(prev_n, n_last)
        if comp_n > 0:
            for ic in range(n_cfg):
                T_prev_cycle = T_eff_all[ic, prev_start:prev_start + comp_n]
                T_last_cycle = T_eff_all[ic, last_start:last_start + comp_n]
                mask = np.isfinite(T_prev_cycle) & np.isfinite(T_last_cycle)
                if mask.any():
                    delta = float(np.max(np.abs(
                        T_last_cycle[mask] - T_prev_cycle[mask])))
                    thermal_cycle_delta[ic] = delta
                    thermal_periodic_converged[ic] = delta < 0.5

    paired = _compute_paired_transient(
        cfg_list, last_start, valid_dynamic_all, valid_no_clamp_all,
        T_eff_used_all, eta_eff_all, P_loss_all, hmin_all, pmax_all,
        Ftr_all)

    # Stage Diesel Transient Production Metrics — per-config + paired.
    production_metrics = _compute_production_metrics(
        cfg_list=cfg_list,
        last_start=last_start,
        phi_crank_deg=phi_crank_deg,
        eps_x_all=eps_x_all,
        eps_y_all=eps_y_all,
        hmin_all=hmin_all,
        pmax_all=pmax_all,
        P_loss_all=P_loss_all,
        valid_dynamic_all=valid_dynamic_all,
        valid_no_clamp_all=valid_no_clamp_all,
        omega_rad_s=float(omega),
        firing_sector_deg=firing_sector_deg,
    )
    paired_extended = _compute_paired_extended(
        cfg_list, paired, production_metrics,
        last_start, valid_no_clamp_all, valid_dynamic_all,
        eps_x_all, eps_y_all, hmin_all, pmax_all, P_loss_all,
        omega_rad_s=float(omega),
        firing_sector_deg=firing_sector_deg,
        phi_crank_deg=phi_crank_deg,
    )

    # Per-config envelope classification (Section 3 of the patch).
    for ic in range(n_cfg):
        n_completed = int(steps_completed_arr[ic])
        # Solver_success / valid counts use the COMPLETED steps only —
        # aborted configs would otherwise inflate denominator with
        # zero-filled tail rows.
        if n_completed > 0:
            sl = slice(0, n_completed)
            ss_count = int(np.sum(solver_success_all[ic, sl]))
            vd_count = int(np.sum(valid_dynamic_all[ic, sl]))
            vnc_count = int(np.sum(valid_no_clamp_all[ic, sl]))
        else:
            ss_count = vd_count = vnc_count = 0
        ok, reason = classify_envelope_per_config(
            n_completed=n_completed,
            solver_success_count=ss_count,
            valid_dynamic_count=vd_count,
            valid_no_clamp_count=vnc_count,
            retry_exhausted_count=int(retry_exhausted_count[ic]),
            aborted=bool(aborted_arr[ic]),
        )
        applicable_arr[ic] = ok
        applicable_reason_arr[ic] = reason

    phi_last = phi_crank_deg[last_start:]
    Fx_ext_last, Fy_ext_last = load_diesel(phi_last % 720.0, F_max=F_max)

    return {
        "phi_crank_deg": phi_crank_deg,
        "phi_last": phi_last,
        "last_start": last_start,
        "n_steps_per_cycle": n_last,
        "eps_x": eps_x_all,
        "eps_y": eps_y_all,
        "hmin": hmin_all,
        "pmax": pmax_all,
        "f": f_all,
        "F_tr": Ftr_all,
        "N_loss": Nloss_all,
        "Fx_hyd": Fx_hyd_all,
        "Fy_hyd": Fy_hyd_all,
        "Fy_ext_last": Fy_ext_last,
        "F_max": F_max,
        "configs": cfg_list,
        "cfg_times": cfg_times,
        # Stage Diesel Transient THD-0 — new keys.
        "T_eff_used": T_eff_used_all,
        "T_eff": T_eff_all,
        "T_target": T_target_all,
        "eta_eff": eta_eff_all,
        "eta_eff_next": eta_eff_next_all,
        "P_loss": P_loss_all,
        "Q": Q_all,
        "mdot": mdot_all,
        "mdot_floor_hit": mdot_floor_hit_all,
        "solver_success": solver_success_all,
        "valid_dynamic": valid_dynamic_all,
        "valid_no_clamp": valid_no_clamp_all,
        "contact_clamp": contact_clamp_all,
        "retry_used": retry_used_all,
        "retry_omega_used": retry_omega_used_all,
        "contact_clamp_count": contact_clamp_count,
        "solver_failed_count": solver_failed_count,
        "retry_recovered_count": retry_recovered_count,
        "retry_exhausted_count": retry_exhausted_count,
        "omega_hits_per_config": omega_hits_per_cfg,
        "thermal_cycle_delta": thermal_cycle_delta,
        "thermal_periodic_converged": thermal_periodic_converged,
        "paired_comparison": paired,
        "production_metrics": production_metrics,
        "paired_extended": paired_extended,
        "firing_sector_deg": firing_sector_deg,
        "thermal_config": thermal.to_dict(),
        "retry_config": retry_config.to_dict(),
        # Stage Diesel Transient Load-Envelope-0 — per-config envelope.
        "envelope_abort_config": envelope_abort.to_dict(),
        "aborted": aborted_arr,
        "abort_reason": abort_reason_arr,
        "first_clamp_phi": first_clamp_phi_arr,
        "first_solver_failed_phi": first_solver_failed_phi_arr,
        "first_invalid_phi": first_invalid_phi_arr,
        "steps_attempted": steps_attempted_arr,
        "steps_completed": steps_completed_arr,
        "applicable": applicable_arr,
        "applicable_reason": applicable_reason_arr,
        # Stage Diesel Transient PeakWindow GridDiagnostic.
        "peak_lo_deg": float(peak_lo_deg),
        "peak_hi_deg": float(peak_hi_deg),
        "N_phi_grid": int(N_phi_eff),
        "N_z_grid": int(N_z_eff),
        "texture_resolution_diagnostic": texture_res_diag,
    }
