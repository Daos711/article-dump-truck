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

# Stage J — per-config Ausas dynamic state owner (lives in the
# transient runner, not in solve_and_compute). The actual GPU
# backend is resolved lazily inside the adapter.
from models.diesel_ausas_adapter import (
    DieselAusasState,
    ausas_one_step_with_state,
)

# Stage J followup-2 — backend-agnostic mechanical-step kernel.
# The half-Sommerfeld path goes through the kernel as of Step 4;
# Ausas dynamic still uses the inline branch (Step 5 will route it
# through ``AusasDynamicBackend`` and the same kernel).
from models import diesel_coupling as _coupling

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
    # Stage J fu-2 fixup-3 — bumped from 30 to 50. The damped-Picard
    # no-advance fix makes ``solver_success_all[ic, step] = False``
    # on Picard-budget-exhausted steps (instead of the old false-
    # positive ``True`` from a state-desynced "advance" with poisoned
    # H_curr/H_prev). On a stiff combustion peak this can produce a
    # short streak of legitimately-non-converged steps while the
    # Picard contractivity detector finds the right relax floor; 30
    # would abort the run mid-stabilisation. 50 leaves headroom for
    # the existing detector budget without masking a true runaway
    # (the OR'd ``solver_fail_frac_max=0.30`` still fires if the run
    # is genuinely unrecoverable).
    consecutive_invalid_max: int = 50
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


def _angle_weighted_zero_record() -> Dict[str, float]:
    """Empty record used when no completed steps are available."""
    nan = float("nan")
    return dict(
        cycle_angle_deg=0.0,
        valid_no_clamp_angle_deg=0.0,
        valid_no_clamp_angle_frac=nan,
        contact_angle_deg=0.0,
        contact_angle_frac=nan,
        firing_angle_deg=0.0,
        valid_no_clamp_angle_firing_deg=0.0,
        valid_no_clamp_angle_firing_frac=nan,
        contact_angle_firing_deg=0.0,
        contact_angle_firing_frac=nan,
    )


def compute_angle_weighted_envelope(
    *,
    dphi: np.ndarray,
    valid_no_clamp: np.ndarray,
    contact_clamp: np.ndarray,
    valid_dynamic: np.ndarray,
    solver_success: np.ndarray,
    phi_mod: np.ndarray,
    n_completed: int,
    firing_sector_deg: Tuple[float, float],
    last_cycle_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Stage Diesel Transient AngleWeighted Metrics.

    Re-weight envelope statistics by the per-step angular increment
    Δφ so adaptive temporal grids (d_phi_peak ≪ d_phi_base near
    firing) do not over-represent clamp incidence in the firing
    sector. Existing count-based fields are kept verbatim — this
    helper layers angle-weighted fractions on top.

    Arrays are indexed [0:n_completed]; ``last_cycle_mask`` (bool)
    is intersected when supplied so callers can request last-cycle
    only statistics. ``firing_sector_deg`` is closed-interval
    [lo, hi] inclusive on phi_mod (= phi_crank_deg % 720).
    """
    if n_completed <= 0:
        return _angle_weighted_zero_record()
    n = int(n_completed)
    dphi_n = np.asarray(dphi[:n], dtype=float)
    vnc_n = np.asarray(valid_no_clamp[:n], dtype=bool)
    cc_n = np.asarray(contact_clamp[:n], dtype=bool)
    phi_n = np.asarray(phi_mod[:n], dtype=float)
    if last_cycle_mask is not None:
        keep = np.asarray(last_cycle_mask[:n], dtype=bool)
        dphi_n = dphi_n[keep]
        vnc_n = vnc_n[keep]
        cc_n = cc_n[keep]
        phi_n = phi_n[keep]
    cycle_angle = float(np.sum(dphi_n))
    vnc_angle = float(np.sum(dphi_n * vnc_n))
    cc_angle = float(np.sum(dphi_n * cc_n))
    lo, hi = float(firing_sector_deg[0]), float(firing_sector_deg[1])
    firing = (phi_n >= lo) & (phi_n <= hi)
    firing_angle = float(np.sum(dphi_n[firing]))
    vnc_firing = float(np.sum(dphi_n[firing] * vnc_n[firing]))
    cc_firing = float(np.sum(dphi_n[firing] * cc_n[firing]))

    def _safe_div(num: float, den: float) -> float:
        return (num / den) if den > 0 else float("nan")

    return dict(
        cycle_angle_deg=cycle_angle,
        valid_no_clamp_angle_deg=vnc_angle,
        valid_no_clamp_angle_frac=_safe_div(vnc_angle, cycle_angle),
        contact_angle_deg=cc_angle,
        contact_angle_frac=_safe_div(cc_angle, cycle_angle),
        firing_angle_deg=firing_angle,
        valid_no_clamp_angle_firing_deg=vnc_firing,
        valid_no_clamp_angle_firing_frac=_safe_div(vnc_firing,
                                                       firing_angle),
        contact_angle_firing_deg=cc_firing,
        contact_angle_firing_frac=_safe_div(cc_firing, firing_angle),
    )


def classify_envelope_per_config(*,
                                    n_completed: int,
                                    solver_success_count: int,
                                    valid_dynamic_count: int,
                                    valid_no_clamp_count: int,
                                    retry_exhausted_count: int,
                                    aborted: bool,
                                    angle_weighted_full: Optional[
                                        Dict[str, float]] = None,
                                    use_angle_weighted: bool = True,
                                    ) -> Tuple[bool, str]:
    """Per-config applicable gate.

    Returns ``(applicable, reason)`` where ``reason`` is "ok" on
    success or a human-readable failure string.

    Stage Diesel Transient AngleWeighted Metrics — when
    ``use_angle_weighted`` is True (default) and
    ``angle_weighted_full`` is supplied, the
    ``valid_no_clamp_frac_min`` gate is evaluated on the angle-
    weighted fraction (Δφ-weighted) instead of the count-based one.
    Solver-success and valid-dynamic gates remain count-based — the
    abort policy is unchanged. Setting ``use_angle_weighted=False``
    restores the legacy purely count-based contract for callers
    that still want the old behaviour.
    """
    if aborted:
        return False, "aborted_outside_envelope"
    if n_completed <= 0:
        return False, "no_steps_completed"
    sf = float(solver_success_count) / float(n_completed)
    vd = float(valid_dynamic_count) / float(n_completed)
    th = ENVELOPE_THRESHOLDS
    if sf < th["solver_success_frac_min"]:
        return False, (
            f"solver_success_frac={sf:.2f} < "
            f"{th['solver_success_frac_min']:.2f}")
    if vd < th["valid_dynamic_frac_min"]:
        return False, (
            f"valid_dynamic_frac={vd:.2f} < "
            f"{th['valid_dynamic_frac_min']:.2f}")
    use_angle = bool(use_angle_weighted) and (
        angle_weighted_full is not None)
    if use_angle:
        vnc_frac = float(angle_weighted_full.get(
            "valid_no_clamp_angle_frac", float("nan")))
        vnc_label = "valid_no_clamp_angle_frac"
    else:
        vnc_frac = float(valid_no_clamp_count) / float(n_completed)
        vnc_label = "valid_no_clamp_frac"
    if not np.isfinite(vnc_frac):
        return False, f"{vnc_label}_unknown"
    if vnc_frac < th["valid_no_clamp_frac_min"]:
        return False, (
            f"{vnc_label}={vnc_frac:.2f} < "
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
               textured=False, phi_c_flat=None, Z_c_flat=None,
               texture_kind: str = "dimple",
               groove_relief: Optional[np.ndarray] = None):
    """Зазор для 2D-эксцентриситета: H = 1 − εx·cos(θ) − εy·sin(θ) [+ текстура].

    Stage J — ``texture_kind`` selects between the legacy
    ellipsoidal-dimple texture (``"dimple"``, the historical default
    so ``textured=True`` keeps its meaning) and a precomputed
    additive groove relief (``"groove"``, with ``groove_relief``
    supplied by ``models.groove_geometry``). When
    ``texture_kind="none"`` the smooth base film is returned even if
    ``textured=True`` — this lets the runner explicitly disable
    texture without changing the ``cfg`` registry.
    """
    H0 = 1.0 - eps_x * np.cos(Phi_mesh) - eps_y * np.sin(Phi_mesh)
    H0 = np.sqrt(H0**2 + (p.sigma / p.c)**2)  # регуляризация шероховатости
    if (not textured) or texture_kind == "none":
        return H0
    if texture_kind == "groove":
        if groove_relief is None:
            raise ValueError(
                "build_H_2d(texture_kind='groove') requires a "
                "groove_relief array (use "
                "models.groove_geometry.build_herringbone_groove_relief"
                " or pass texture_kind='dimple' for the legacy path)."
            )
        return H0 + np.asarray(groove_relief, dtype=H0.dtype)
    if texture_kind != "dimple":
        raise ValueError(
            f"build_H_2d: unknown texture_kind {texture_kind!r}; "
            "valid values are 'dimple', 'groove', 'none'."
        )
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


def _print_ausas_debug_step(
    label, *,
    phi_deg, dt_s, aw_result,
    eps_x, eps_y, p_scale,
    Fx_hyd, Fy_hyd, F_max,
):
    """Stage J followup §4 — diagnostic line for the first N
    Ausas-dynamic Verlet substep / accepted-step decisions.

    Activated by ``run_transient(..., debug_first_steps=N)`` /
    CLI ``--debug-first-steps N``. Prints once per substep call
    (``TRIAL k=...``) plus once per accepted step
    (``ACCEPTED``); the output goes to stdout BEFORE the regular
    progress / summary lines so the operator can locate the
    Verlet break-down on a failing smoke without re-running.
    """
    Fx_ext_a, Fy_ext_a = load_diesel(phi_deg, F_max=F_max)
    Fx_ext = float(np.asarray(Fx_ext_a).item())
    Fy_ext = float(np.asarray(Fy_ext_a).item())
    F_ext_mag = float(np.hypot(Fx_ext, Fy_ext))
    F_hyd_mag = float(np.hypot(Fx_hyd, Fy_hyd))
    if (F_hyd_mag > 0.0 and F_ext_mag > 0.0
            and np.isfinite(F_hyd_mag) and np.isfinite(F_ext_mag)):
        dot_norm = float(
            (Fx_hyd * Fx_ext + Fy_hyd * Fy_ext)
            / (F_hyd_mag * F_ext_mag))
    else:
        dot_norm = float("nan")
    p_nd_max = float(aw_result.p_nd_max)
    p_dim_max_MPa = (p_nd_max * float(p_scale) * 1e-6
                     if np.isfinite(p_nd_max) else float("nan"))
    print(
        f"  [J-debug {label}] "
        f"phi={float(phi_deg):.2f}° "
        f"dt_s={float(dt_s):.4e}s dt_tau={float(aw_result.dt_ausas):.4e} "
        f"eps=({float(eps_x):+.4f},{float(eps_y):+.4f}) "
        f"|F_ext|={F_ext_mag/1e3:.2f}kN "
        f"({Fx_ext/1e3:+.2f},{Fy_ext/1e3:+.2f}) "
        f"|F_hyd|={F_hyd_mag/1e3:.4f}kN "
        f"({float(Fx_hyd)/1e3:+.4f},{float(Fy_hyd)/1e3:+.4f}) "
        f"dot_norm={dot_norm:+.3f} "
        f"p_nd_max={p_nd_max:.4e} p_dim_max={p_dim_max_MPa:.3f}MPa "
        f"theta=[{float(aw_result.theta_min):.3f},"
        f"{float(aw_result.theta_max):.3f}] "
        f"res={float(aw_result.residual):.2e} "
        f"n_inner={int(aw_result.n_inner)} "
        f"converged={bool(aw_result.converged)}",
        flush=True,
    )


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
    *,
    R: float, L: float,
    texture_kind: str = "dimple",
    a_dim: Optional[float] = None,
    b_dim: Optional[float] = None,
    w_branch_angle_rad: Optional[float] = None,
) -> Dict[str, Any]:
    """Diagnose whether a (Nφ × N_Z) grid resolves the active texture
    feature. Branches on ``texture_kind``:

    * ``"dimple"`` — elliptical pocket geometry. Requires the
      DieselParams semi-axes ``a_dim`` (axial, meters) and ``b_dim``
      (circumferential, meters). Pocket full angular width is
      ``2·b_dim/R`` rad; full non-dim axial width is ``4·a_dim/L``
      (physical z = (L/2)·Z, Z ∈ [-1, +1]). Returns
      ``cells_per_pocket_phi`` / ``cells_per_pocket_z``.
    * ``"groove"`` — herringbone branch geometry. Requires
      ``w_branch_angle_rad`` (the angular branch width on the Phi
      grid; numerically equal to ``w_g_m / R_m``, see
      ``config/diesel_groove_presets.py``). Returns
      ``cells_per_groove_width_phi`` (axial direction is
      well-resolved by L_branch ≈ 0.85·half-shell at any reasonable
      N_z, so it isn't a binding constraint).

    Returned ``resolution_status`` is the same three-level enum on
    both paths: ``ok`` (≥6 cells across the small length-scale),
    ``marginal`` (4–5), ``insufficient`` (<4). The
    ``recommended_n_phi_min`` is the smallest N_phi giving 4 cells
    across the active feature width.
    """
    N_phi = int(N_phi)
    N_z = int(N_z)
    R = float(R)
    L = float(L)
    if texture_kind == "groove":
        if w_branch_angle_rad is None:
            raise ValueError(
                "texture_resolution_diagnostic(texture_kind='groove')"
                " requires w_branch_angle_rad (the angular branch "
                "width in radians, = w_g/R; see "
                "config.diesel_groove_presets.resolve_groove_preset).")
        w_rad = float(w_branch_angle_rad)
        if w_rad <= 0.0:
            raise ValueError(
                "w_branch_angle_rad must be positive; got "
                f"{w_rad}")
        cells_per_groove_width_phi = (
            float(N_phi) * w_rad / (2.0 * np.pi))
        if cells_per_groove_width_phi >= 6.0:
            status = "ok"
        elif cells_per_groove_width_phi >= 4.0:
            status = "marginal"
        else:
            status = "insufficient"
        recommended_n_phi_min = int(np.ceil(
            4.0 * 2.0 * np.pi / w_rad))
        return {
            "texture_kind": "groove",
            "N_phi": N_phi,
            "N_z": N_z,
            "w_branch_angle_rad": w_rad,
            "cells_per_groove_width_phi": cells_per_groove_width_phi,
            "resolution_status": status,
            "recommended_n_phi_min": recommended_n_phi_min,
        }
    # Default / dimple path — preserves the legacy contract.
    if a_dim is None or b_dim is None:
        raise ValueError(
            "texture_resolution_diagnostic(texture_kind='dimple') "
            "requires a_dim and b_dim (DieselParams pocket semi-"
            "axes in meters).")
    a_dim_f = float(a_dim)
    b_dim_f = float(b_dim)
    cells_per_pocket_phi = float(N_phi) * b_dim_f / (np.pi * R)
    cells_per_pocket_z = 2.0 * float(N_z) * a_dim_f / L
    if cells_per_pocket_phi >= 6.0:
        status = "ok"
    elif cells_per_pocket_phi >= 4.0:
        status = "marginal"
    else:
        status = "insufficient"
    recommended_n_phi_min = int(np.ceil(4.0 * np.pi * R / b_dim_f))
    return {
        "texture_kind": "dimple",
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
                  n_z_grid: Optional[int] = None,
                  texture_kind: str = "dimple",
                  groove_preset: Optional[str] = None,
                  fidelity: Optional[str] = None,
                  ausas_options: Optional[Dict[str, Any]] = None,
                  save_field_checkpoints: bool = False,
                  debug_first_steps: int = 0,
                  debug_from_step: int = 0,
                  seed: int = 0,
                  # Stage J fu-2 Step 10 — coupling policy CLI surface.
                  guards_profile: str = "general",
                  coupling_override: str = "auto",
                  max_mech_inner: Optional[int] = None,
                  mech_relax_initial: Optional[float] = None,
                  mech_relax_min: Optional[float] = None,
                  # Stage J fu-2 Task 29 — failed-Ausas dump path.
                  # ``dump_failed_one_step_dir=None`` keeps the
                  # production path silent; setting a directory and
                  # the runner's CLI thread the writes through the
                  # adapter via reserved ``extra_options`` keys.
                  dump_failed_one_step_dir: Optional[str] = None,
                  dump_failed_one_step_limit: int = 20,
                  dump_failed_one_step_force_inputs: bool = True,
                  ausas_debug_checks: bool = False,
                  ausas_debug_check_every: int = 50,
                  ausas_debug_stop_on_nonfinite: bool = True,
                  ausas_debug_return_bad_state: bool = False,
                  ausas_debug_return_last_finite_state: bool = True):
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
    # Stage J — cavitation routing. ``half_sommerfeld`` (the
    # historical default) goes through the existing ``solve_reynolds``
    # path bit-for-bit. ``ausas_dynamic`` activates the per-config
    # ``DieselAusasState`` and the one-step adapter wired below.
    # Any unknown name is rejected loudly so silent fallback to the
    # legacy path cannot mask a typo.
    cavitation_str = str(cavitation)
    if cavitation_str not in ("half_sommerfeld", "ausas_dynamic"):
        raise ValueError(
            f"run_transient: unsupported cavitation {cavitation_str!r}; "
            "valid values are 'half_sommerfeld' (default, legacy) "
            "and 'ausas_dynamic' (Stage J).")
    use_ausas_dynamic = (cavitation_str == "ausas_dynamic")
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

    # Stage J — herringbone groove relief (built once per run on the
    # fixed grid; the relief is additive on top of the eccentricity-
    # driven base film). When ``texture_kind != "groove"`` the
    # relief stays None and ``build_H_2d`` falls back to either the
    # legacy dimple texture or the smooth base film.
    groove_relief: Optional[np.ndarray] = None
    groove_preset_resolved: Optional[Dict[str, Any]] = None
    groove_relief_stats: Optional[Dict[str, Any]] = None
    if texture_kind == "groove":
        if groove_preset is None:
            raise ValueError(
                "run_transient(texture_kind='groove') requires a "
                "groove_preset name (see config.diesel_groove_presets"
                ".GROOVE_PRESETS).")
        from config.diesel_groove_presets import resolve_groove_preset
        from models.groove_geometry import (
            build_herringbone_groove_relief, relief_stats,
        )
        groove_preset_resolved = resolve_groove_preset(
            str(groove_preset),
            R_m=float(params.R), L_m=float(params.L),
            c_m=float(params.c),
        )
        builder_kwargs = {
            k: groove_preset_resolved[k] for k in (
                "variant", "depth_nondim", "N_branch_per_side",
                "w_branch_nondim", "belt_half_nondim", "beta_deg",
                "ramp_frac", "taper_ratio", "apex_radius_frac",
                "chirality", "coverage_mode",
                "protected_lo_deg", "protected_hi_deg",
            )
        }
        groove_relief = build_herringbone_groove_relief(
            Phi_mesh, Z_mesh, **builder_kwargs)
        groove_relief_stats = relief_stats(groove_relief)
        if (groove_relief_stats["has_nan"]
                or groove_relief_stats["has_inf"]):
            raise ValueError(
                "groove relief contains NaN/inf — refusing to "
                "proceed (preset="
                f"{groove_preset!r}).")
        if groove_relief_stats["relief_min"] < -1e-12:
            raise ValueError(
                "groove relief contains negative values — grooves "
                "must increase H, not reduce it (preset="
                f"{groove_preset!r}, min="
                f"{groove_relief_stats['relief_min']:.3e}).")

    # Stage Diesel Transient PeakWindow GridDiagnostic — diagnose
    # whether the chosen Nφ × N_Z grid resolves the active texture
    # feature. The diagnostic depends only on grid + global
    # bearing geometry, so it is the same for every textured config.
    #
    # Stage J fu-2 fixup-1: branch on ``texture_kind`` so groove
    # runs are evaluated against the actual herringbone branch
    # width (``w_branch_angle_rad`` from the preset), not the
    # legacy dimple semi-axes ``params.a_dim`` / ``params.b_dim``
    # which describe a different texture model entirely.
    if texture_kind == "groove" and groove_preset_resolved is not None:
        texture_res_diag = texture_resolution_diagnostic(
            N_phi_eff, N_z_eff,
            R=params.R, L=params.L,
            texture_kind="groove",
            w_branch_angle_rad=float(
                groove_preset_resolved["w_branch_angle_rad"]),
        )
        if texture_res_diag["resolution_status"] == "insufficient":
            print(
                "  [WARN] groove branch width under-resolved: "
                f"cells_per_groove_width_phi="
                f"{texture_res_diag['cells_per_groove_width_phi']:.2f}"
                " < 4 "
                f"(N_phi={N_phi_eff}; recommend N_phi >= "
                f"{texture_res_diag['recommended_n_phi_min']})"
            )
        elif texture_res_diag["resolution_status"] == "marginal":
            print(
                "  [WARN] groove branch width marginal: "
                f"cells_per_groove_width_phi="
                f"{texture_res_diag['cells_per_groove_width_phi']:.2f}"
                " in [4, 6) — consider N_phi >= "
                f"{texture_res_diag['recommended_n_phi_min']}*1.5 "
                "for production accuracy."
            )
    else:
        texture_res_diag = texture_resolution_diagnostic(
            N_phi_eff, N_z_eff,
            R=params.R, L=params.L,
            texture_kind="dimple",
            a_dim=params.a_dim, b_dim=params.b_dim,
        )
        if texture_res_diag["resolution_status"] == "insufficient":
            print(
                "  [WARN] dimple pocket under-resolved: "
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
    # Stage Diesel Transient ClampAccounting Fix — distinct from
    # ``contact_clamp_all`` (step-mask): per-step *event* count
    # accumulating predictor + N_SUB substep + final clamp events
    # (so a single step can carry up to 1 + (N_SUB - 1) + 1 events).
    # ``contact_clamp_all`` keeps its existing semantics and remains
    # the only source for envelope abort fractions.
    contact_clamp_event_count_all = np.zeros((n_cfg, n_steps),
                                                dtype=np.int32)
    # Stage Diesel Transient AngleWeighted Metrics — per-step angular
    # increment (deg) and crank position (mod 720°) so envelope and
    # firing-sector statistics can be re-weighted by Δφ on the
    # adaptive temporal grid (count-based fractions over-state clamp
    # share when d_phi_peak << d_phi_base).
    d_phi_per_step_all = np.zeros((n_cfg, n_steps), dtype=float)
    phi_mod_per_step_all = np.zeros((n_cfg, n_steps), dtype=float)
    # Stage J — per-config Ausas dynamic diagnostics. Default-zero
    # so the legacy half-Sommerfeld path produces a result dict with
    # zero-filled Ausas fields rather than missing keys.
    ausas_converged_all = np.zeros((n_cfg, n_steps), dtype=bool)
    ausas_n_inner_all = np.zeros((n_cfg, n_steps), dtype=np.int32)
    ausas_cav_frac_all = np.zeros((n_cfg, n_steps), dtype=float)
    ausas_theta_min_all = np.full((n_cfg, n_steps), 1.0, dtype=float)
    ausas_theta_max_all = np.full((n_cfg, n_steps), 1.0, dtype=float)
    # Stage J Bug 4 follow-up — per-step residual and Ausas-domain dt
    # for sanity / convergence-quality diagnostics.
    ausas_residual_all = np.full((n_cfg, n_steps), np.nan, dtype=float)
    ausas_dt_tau_all = np.zeros((n_cfg, n_steps), dtype=float)
    # Stage J fu-2 Step 10 — per-step coupling diagnostics for the
    # Gate 3 summary block + npz schema. Both legacy and damped
    # paths populate these so the post-fact analysis script gets a
    # uniform schema. ``picard_shrinks`` and
    # ``fixed_point_converged`` are damped-only (zeros / False on
    # the legacy path); ``mech_relax_min_seen`` and ``n_trials``
    # are on both.
    coupling_picard_shrinks_all = np.zeros(
        (n_cfg, n_steps), dtype=np.int32)
    coupling_mech_relax_min_seen_all = np.full(
        (n_cfg, n_steps), np.nan, dtype=float)
    coupling_fp_converged_all = np.zeros(
        (n_cfg, n_steps), dtype=bool)
    coupling_n_trials_all = np.zeros(
        (n_cfg, n_steps), dtype=np.int32)
    # ``rejection_reason`` is a variable-length enum string per
    # step. Stored as object dtype (numpy will pickle on save).
    coupling_rejection_reason_all = np.full(
        (n_cfg, n_steps), "none", dtype=object)
    # Stage J fu-2 Task 32 — commit-semantics state-machine arrays.
    # Defaults match the kernel's ``MechanicalStepResult`` defaults
    # so the runner-side schema stays uniform across legacy and
    # damped paths and across older runs that pre-date Task 32.
    final_trial_status_all = np.full(
        (n_cfg, n_steps), "no_attempt", dtype=object)
    committed_state_status_all = np.full(
        (n_cfg, n_steps), "rejected_no_commit", dtype=object)
    accepted_state_source_all = np.full(
        (n_cfg, n_steps), "none", dtype=object)
    committed_state_is_finite_all = np.zeros(
        (n_cfg, n_steps), dtype=bool)
    final_trial_failure_kind_all = np.full(
        (n_cfg, n_steps), "", dtype=object)
    final_trial_residual_all = np.full(
        (n_cfg, n_steps), np.nan, dtype=float)
    final_trial_n_inner_all = np.zeros(
        (n_cfg, n_steps), dtype=np.int32)
    # Stage J Bug 4 follow-up — per-step force-balance diagnostics.
    F_hyd_x_all = np.zeros((n_cfg, n_steps), dtype=float)
    F_hyd_y_all = np.zeros((n_cfg, n_steps), dtype=float)
    F_ext_x_all = np.zeros((n_cfg, n_steps), dtype=float)
    F_ext_y_all = np.zeros((n_cfg, n_steps), dtype=float)
    force_balance_projection_all = np.full(
        (n_cfg, n_steps), np.nan, dtype=float)
    ausas_state_reset_count = np.zeros(n_cfg, dtype=np.int32)
    ausas_failed_step_count = np.zeros(n_cfg, dtype=np.int32)
    ausas_rejected_commit_count = np.zeros(n_cfg, dtype=np.int32)
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

    # Stage J fu-2 Task 29 — failed-Ausas dump path setup.
    # Single ``DumpConfig`` / ``DumpCounters`` per run (shared
    # across configs) so the limit is global. Disabled-by-default
    # production path: ``directory=None`` → no I/O, no perf hit.
    from models.diesel_ausas_dump_io import DumpConfig as _DumpCfg
    from models.diesel_ausas_dump_io import DumpCounters as _DumpCnt
    _dump_cfg = _DumpCfg(
        directory=dump_failed_one_step_dir,
        limit=int(dump_failed_one_step_limit),
        include_force_inputs=bool(
            dump_failed_one_step_force_inputs),
    )
    _dump_counters = _DumpCnt()

    # Stage J fu-2 Task 29 — capability-safe debug kwargs to forward
    # to ``ausas_unsteady_one_step_gpu``. Adapter strips reserved
    # ``__dump_*__`` keys before the backend call; the ``debug_*``
    # kwargs ARE forwarded, and the GPU side accepts whichever it
    # supports (unknown kwargs would TypeError on a real backend
    # but the test backends use ``**kwargs`` and silently swallow).
    _ausas_debug_options: Dict[str, Any] = {}
    if ausas_debug_checks:
        _ausas_debug_options["debug_checks"] = True
        _ausas_debug_options["debug_check_every"] = int(
            ausas_debug_check_every)
        _ausas_debug_options["debug_stop_on_nonfinite"] = bool(
            ausas_debug_stop_on_nonfinite)
        _ausas_debug_options["debug_return_bad_state"] = bool(
            ausas_debug_return_bad_state)
        _ausas_debug_options["debug_return_last_finite_state"] = bool(
            ausas_debug_return_last_finite_state)

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

        # Stage J Bug 5 — initialise the per-config Ausas state from
        # the actual first accepted gap. Using ``cold_start`` with
        # ``H_prev = ones`` injected an artificial squeeze pulse on
        # step #1 because the first accepted ``H`` was generally
        # non-unit (eps_x0/eps_y0 != 0); the resulting fictitious
        # ∂(θH)/∂t blew up the orbit. ``from_initial_gap`` aligns
        # ``H_prev`` with the true first H.
        ausas_state: Optional[DieselAusasState] = None
        if use_ausas_dynamic:
            H_initial = build_H_2d(
                ex / params.c, ey / params.c,
                Phi_mesh, Z_mesh, params,
                textured=cfg["textured"],
                phi_c_flat=phi_c, Z_c_flat=Z_c,
                texture_kind=texture_kind,
                groove_relief=groove_relief,
            )
            ausas_state = DieselAusasState.from_initial_gap(H_initial)

        # Stage J fu-2 Step 5 — instantiate the pressure backend
        # once per config. ``HalfSommerfeldBackend`` carries the
        # retry policy + textured-for-retry flag; ``AusasDynamicBackend``
        # carries the ``ausas_options`` dict. Both go through the
        # same ``advance_mechanical_step`` kernel call below.
        if use_ausas_dynamic:
            _backend = _coupling.AusasDynamicBackend(
                ausas_options=ausas_options or None)
        else:
            _backend = _coupling.HalfSommerfeldBackend(
                retry_config=retry_config,
                textured_for_retry=bool(cfg["textured"]),
            )

        # Stage J fu-2 Step 10 — resolve once per config. CLI flags
        # ``--coupling-policy`` / ``--max-mech-inner`` /
        # ``--mech-relax-initial`` / ``--mech-relax-min`` layer
        # overrides on top of the capability-based default.
        _coupling_policy = _coupling.resolve_policy(
            _backend,
            coupling_override=str(coupling_override),
            max_mech_inner=max_mech_inner,
            mech_relax_initial=mech_relax_initial,
            mech_relax_min=mech_relax_min,
        )
        # Mode default: ``hard`` for damped Ausas (per
        # POLICY_AUSAS_DYNAMIC.physical_guards_mode), ``diagnostic``
        # for legacy_verlet (preserves Gate 1 invariance —
        # diagnostic warns but does NOT reject).
        _guards_mode = (
            "hard" if _coupling_policy.name == "damped_implicit_film"
            else "diagnostic")
        _guards_cfg = _coupling.PhysicalGuardsConfig.from_profile(
            _guards_mode, str(guards_profile))

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
            d_phi_step_deg = float(get_step_deg(
                phi_crank_deg[step],
                d_phi_base_deg=d_phi_base_deg,
                d_phi_peak_deg=d_phi_peak_deg,
                peak_lo_deg=peak_lo_deg,
                peak_hi_deg=peak_hi_deg,
            ))
            dt_step = np.deg2rad(d_phi_step_deg) / omega
            d_phi_per_step_all[ic, step] = d_phi_step_deg
            phi_mod_per_step_all[ic, step] = float(phi_deg)

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

            # Stage J fu-2 Step 5 — both backends go through the
            # backend-agnostic mechanical-step kernel. ``_backend``
            # was instantiated once per config above; the same
            # ``advance_mechanical_step`` call dispatches Verlet
            # predict / for-k corrector / final clamp / (stateful)
            # commit. Half-Sommerfeld is bit-for-bit identical to
            # the pre-refactor runner (Gate 1 fixture); Ausas
            # dynamic uses ``POLICY_LEGACY_HS`` until Step 6 swaps
            # it for the damped policy.
            Fx_ext_step_a, Fy_ext_step_a = load_diesel(
                phi_deg, F_max=F_max)
            Fx_ext_step = float(np.asarray(Fx_ext_step_a).item())
            Fy_ext_step = float(np.asarray(Fy_ext_step_a).item())

            step_ctx = _coupling.StepContext(
                phi_deg=float(phi_deg),
                F_ext_x=Fx_ext_step, F_ext_y=Fy_ext_step,
                F_max=float(F_max),
                p_scale=float(p_scale_step),
                omega=float(omega), eta=float(eta_step),
                R=float(params.R), L=float(params.L),
                c=float(params.c),
                Phi_mesh=Phi_mesh, Z_mesh=Z_mesh,
                phi_1D=phi_1D, Z_1D=Z_1D,
                d_phi=float(d_phi), d_Z=float(d_Z),
                cfg=cfg,
                texture_kind=str(texture_kind),
                groove_relief=groove_relief,
                phi_c_flat=phi_c, Z_c_flat=Z_c,
                closure=closure, cavitation=cavitation,
                P_warm_init=P_prev,
            )

            def _build_H_for_kernel(eps_x_, eps_y_):
                return build_H_2d(
                    eps_x_, eps_y_, Phi_mesh, Z_mesh, params,
                    textured=cfg["textured"],
                    phi_c_flat=phi_c, Z_c_flat=Z_c,
                    texture_kind=texture_kind,
                    groove_relief=groove_relief,
                )

            # Stage J fu-2 Step 9 fixup — propagate the user's
            # ``--ausas-tol`` / ``--ausas-max-inner`` (carried in
            # ``ausas_options``) to the kernel's solver-validity
            # cap. Hard-coding 1e-6 / 5000 turned every legitimate
            # ``residual <= user_tol`` solve into a false
            # ``SOLVER_RESIDUAL`` reject and stalled the Picard loop.
            # The HS path ignores both, so threading is safe.
            _kernel_ausas_tol = float(
                (ausas_options or {}).get("tol", 1e-6))
            _kernel_ausas_max_inner = int(
                (ausas_options or {}).get("max_inner", 5000))
            _ms = _coupling.advance_mechanical_step(
                ex_n=ex_n, ey_n=ey_n,
                vx_n=vx_n, vy_n=vy_n,
                ax_prev=ax_prev, ay_prev=ay_prev,
                dt_phys_s=float(dt_step),
                backend=_backend,
                backend_state=ausas_state,
                # Stage J fu-2 Step 10 — policy + guards resolved
                # once per config; the kernel never sees the CLI.
                policy=_coupling_policy,
                guards_cfg=_guards_cfg,
                ausas_tol=_kernel_ausas_tol,
                ausas_max_inner=_kernel_ausas_max_inner,
                extra_options=(
                    {
                        **(ausas_options or {}),
                        **_ausas_debug_options,
                        "__dump_config__": _dump_cfg,
                        "__dump_counters__": _dump_counters,
                        "__dump_metadata__": dict(
                            step=int(step),
                            substep=-1,
                            trial=-1,
                            phi_deg=float(phi_deg),
                            eps_x=float(ex_n) / params.c,
                            eps_y=float(ey_n) / params.c,
                            config_label=str(cfg["label"]),
                            trial_kind="picard_trial",
                            texture_kind=str(texture_kind),
                            groove_preset=str(groove_preset)
                                if groove_preset else "",
                            cavitation=str(cavitation),
                            F_hyd_x=None, F_hyd_y=None,
                        ),
                    }
                    if use_ausas_dynamic else None),
                context=step_ctx,
                m_shaft=float(params.m_shaft),
                eps_max=float(params.eps_max),
                clamp_fn=_clamp,
                build_H_fn=_build_H_for_kernel,
                p_warm_init=P_prev,
                # Stage J fu-2 Step 9 diagnostic — only emit the
                # damped-policy per-iteration dump while the runner
                # is still inside the operator's debug window.
                debug_dump=(use_ausas_dynamic
                            and step >= int(debug_from_step)
                            and step < int(debug_from_step)
                            + int(debug_first_steps)),
            )

            # Apply kernel result to runner state — preserves
            # the legacy bookkeeping arrays bit-for-bit (Gate 1).
            contact_count += int(_ms.n_contact_events)
            step_event_count = int(_ms.n_contact_events)
            P_prev = _ms.p_warm_out

            # Stage J fu-2 Step 10 — per-step coupling diagnostics
            # for the Gate 3 summary block. Both backend paths
            # populate these so the per-step npz arrays remain on
            # a uniform schema (legacy_verlet returns 0 / False /
            # 1.0 for the damped-only fields per kernel defaults).
            coupling_picard_shrinks_all[ic, step] = int(
                _ms.picard_shrinks_count)
            coupling_mech_relax_min_seen_all[ic, step] = float(
                _ms.mech_relax_min_seen)
            coupling_fp_converged_all[ic, step] = bool(
                _ms.fixed_point_converged_flag)
            coupling_n_trials_all[ic, step] = int(_ms.n_trials)
            coupling_rejection_reason_all[ic, step] = (
                _ms.rejection_reason.value)
            # Stage J fu-2 Task 32 — commit-semantics state-machine
            # passthrough. Per-step arrays for the postprocessor /
            # summary writer to surface ``committed_converged`` vs.
            # ``committed_last_valid`` vs. ``rolled_back_previous``
            # vs. ``rejected_no_commit`` distinctions.
            final_trial_status_all[ic, step] = (
                str(_ms.final_trial_status))
            committed_state_status_all[ic, step] = (
                str(_ms.committed_state_status))
            accepted_state_source_all[ic, step] = (
                str(_ms.accepted_state_source))
            committed_state_is_finite_all[ic, step] = bool(
                _ms.committed_state_is_finite)
            final_trial_failure_kind_all[ic, step] = (
                str(_ms.final_trial_failure_kind))
            final_trial_residual_all[ic, step] = float(
                _ms.final_trial_residual)
            final_trial_n_inner_all[ic, step] = int(
                _ms.final_trial_n_inner)
            # Stage J fu-2 fixup-3 — runner-side reaction to kernel
            # Picard no-advance. On ``_ms.accepted=False`` the kernel
            # already rolled mechanics back to start-of-step values
            # (ex_n/ey_n/vx_n/vy_n/ax_prev/ay_prev) and zeroed the
            # trial outputs, but the per-step plotting arrays
            # (eps_x_all etc.) are written unconditionally below
            # from ``ex/ey/Fx_hyd/Fy_hyd``. Without this branch the
            # rollback step is silently logged as "orbit at the
            # previous accepted point" — looks like a successful
            # zero-Δε advance to envelope classification, which then
            # never trips ``solver_fail_frac_max`` and lets the run
            # continue indefinitely with a frozen orbit. Branch
            # explicitly so the no-advance intent is visible to the
            # reader (and to skip silent state mutation when both
            # the kernel and the runner already agree the step
            # didn't happen).
            if _ms.accepted:
                ex, ey = _ms.eps_x_new, _ms.eps_y_new
                vx, vy = _ms.vx_new, _ms.vy_new
                ax_prev, ay_prev = _ms.ax_new, _ms.ay_new
            else:
                # Pre-call values: ``ex_n / ey_n / vx_n / vy_n``
                # were saved at start-of-step (≈line 1535-1536);
                # ``ax_prev / ay_prev`` are still the inputs to
                # this kernel call (the lines that overwrote them
                # were skipped). Explicit assignment from the
                # pre-call vars rather than ``_ms.*_new`` makes
                # the no-advance intent obvious and avoids relying
                # on the kernel's identity echo to keep state in
                # sync.
                ex, ey = ex_n, ey_n
                vx, vy = vx_n, vy_n
                # ``ax_prev / ay_prev`` are unchanged — the
                # rollback assignment is a no-op.
            P = _ms.P_nd_committed
            H = _ms.H_committed
            Fx_hyd = _ms.Fx_hyd_committed
            Fy_hyd = _ms.Fy_hyd_committed
            solve_ok = (P is not None
                        and _ms.rejection_reason
                        == _coupling.RejectionReason.NONE)
            solve_reason = _ms.solve_reason
            retry_recovered_step = bool(_ms.retry_recovered)
            omega_recovery = _ms.omega_recovery
            clamped_p = bool(_ms.predictor_clamped)
            clamped_final = bool(_ms.final_clamped)
            step_clamped = bool(_ms.step_clamped)

            # Stage J Ausas-only diagnostic arrays (zero-defaults on
            # the HS path are correct — backend.stateful is False so
            # the kernel never invoked an Ausas commit).
            if use_ausas_dynamic:
                ausas_converged_all[ic, step] = bool(_ms.state_committed)
                ausas_n_inner_all[ic, step] = int(_ms.n_inner)
                ausas_cav_frac_all[ic, step] = float(
                    _ms.cav_frac_committed)
                ausas_theta_min_all[ic, step] = float(
                    _ms.theta_min_committed)
                ausas_theta_max_all[ic, step] = float(
                    _ms.theta_max_committed)
                ausas_residual_all[ic, step] = float(_ms.residual)
                ausas_dt_tau_all[ic, step] = float(
                    _ms.dt_ausas_committed)
                # ``--debug-first-steps N`` — stream per-trial
                # diagnostic lines from the kernel's trial_log, plus
                # one ACCEPTED line summarising the committed step.
                if (step >= int(debug_from_step)
                        and step < int(debug_from_step)
                        + int(debug_first_steps)):
                    for k_idx, tr in enumerate(_ms.trial_log):
                        _print_ausas_debug_step(
                            f"step={step:03d} TRIAL k={k_idx}",
                            phi_deg=phi_deg, dt_s=dt_step,
                            aw_result=tr.pressure_result,
                            eps_x=tr.eps_x_cand, eps_y=tr.eps_y_cand,
                            p_scale=p_scale_step,
                            Fx_hyd=tr.pressure_result.Fx_hyd,
                            Fy_hyd=tr.pressure_result.Fy_hyd,
                            F_max=F_max,
                        )
                    # Build a minimal "accepted" PressureSolveResult-
                    # shaped object the printer can consume.
                    _accepted = type(
                        "AcceptedView", (),
                        dict(
                            P_nd=_ms.P_nd_committed,
                            theta=_ms.theta_committed,
                            residual=_ms.residual,
                            n_inner=_ms.n_inner,
                            converged=bool(_ms.state_committed),
                            dt_phys_s=float(dt_step),
                            dt_ausas=_ms.dt_ausas_committed,
                            cav_frac=_ms.cav_frac_committed,
                            theta_min=_ms.theta_min_committed,
                            theta_max=_ms.theta_max_committed,
                            p_nd_max=(float(np.max(_ms.P_nd_committed))
                                      if _ms.P_nd_committed is not None
                                      else 0.0),
                        ),
                    )()
                    # Stage J fu-2 Task 32 — split misleading
                    # ``ACCEPTED`` into separate FINAL-TRIAL-* /
                    # COMMITTED-* / ROLLED-BACK lines so a NaN
                    # F_hyd / nan residual / converged=False step
                    # never prints under the same label as a
                    # genuinely converged commit.
                    _ftrial_status = str(getattr(
                        _ms, "final_trial_status", "no_attempt"))
                    _committed_status = str(getattr(
                        _ms, "committed_state_status",
                        "rejected_no_commit"))
                    if _ftrial_status != "converged":
                        # Separate FINAL-TRIAL-FAILED line — names
                        # the failure cause for the operator.
                        _print_ausas_debug_step(
                            f"step={step:03d} "
                            f"FINAL-TRIAL-FAILED({_ftrial_status})",
                            phi_deg=phi_deg, dt_s=dt_step,
                            aw_result=_accepted,
                            eps_x=ex / params.c, eps_y=ey / params.c,
                            p_scale=p_scale_step,
                            Fx_hyd=Fx_hyd, Fy_hyd=Fy_hyd,
                            F_max=F_max,
                        )
                    if _committed_status == "committed_converged":
                        _committed_label = (
                            f"step={step:03d} "
                            "COMMITTED(converged_trial)")
                    elif _committed_status == "committed_last_valid":
                        _committed_label = (
                            f"step={step:03d} "
                            "COMMITTED(last_valid_trial)")
                    elif _committed_status == "rolled_back_previous":
                        _committed_label = (
                            f"step={step:03d} "
                            f"ROLLED-BACK(reason={_ftrial_status})")
                    else:
                        _committed_label = (
                            f"step={step:03d} "
                            f"REJECTED-NO-COMMIT(reason={_ftrial_status})")
                    _print_ausas_debug_step(
                        _committed_label,
                        phi_deg=phi_deg, dt_s=dt_step,
                        aw_result=_accepted,
                        eps_x=ex / params.c, eps_y=ey / params.c,
                        p_scale=p_scale_step,
                        Fx_hyd=Fx_hyd, Fy_hyd=Fy_hyd,
                        F_max=F_max,
                    )

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
            # Stage J fu-2 fixup-3 — Picard no-advance: explicit
            # NaN on the per-step plotting arrays so envelope
            # classification (valid_dynamic, AUC, max-ε) treats
            # the step as invalid. h_min/p_max/F_friction/mu_val/
            # Fx_hyd/Fy_hyd are already NaN via the
            # ``solve_ok=False`` else branch above and the kernel's
            # rollback returns; eps_x_all / eps_y_all are written
            # from ``ex / ey`` which on rollback equal the pre-call
            # orbit position (last accepted point), so they need
            # an explicit override. The redundant NaN-writes for
            # the other six are intentional — they make the
            # rollback contract independent of upstream details.
            if not _ms.accepted:
                eps_x_all[ic, step] = float("nan")
                eps_y_all[ic, step] = float("nan")
                hmin_all[ic, step] = float("nan")
                pmax_all[ic, step] = float("nan")
                f_all[ic, step] = float("nan")
                Ftr_all[ic, step] = float("nan")
                Fx_hyd_all[ic, step] = float("nan")
                Fy_hyd_all[ic, step] = float("nan")
            # Stage J Bug 4 follow-up — record both force vectors and
            # the dot-product projection of F_hyd onto the resisting
            # direction (-F_ext). A physically scaled hydrodynamic
            # response should keep this projection POSITIVE: the
            # film pressure pushes against the external load. A
            # consistently negative or near-zero projection is the
            # signature of a sign / unit regression in the cavitation
            # backend coupling.
            Fx_ext_arr, Fy_ext_arr = load_diesel(phi_deg, F_max=F_max)
            Fx_ext_step = float(np.asarray(Fx_ext_arr).item())
            Fy_ext_step = float(np.asarray(Fy_ext_arr).item())
            F_hyd_x_all[ic, step] = float(Fx_hyd)
            F_hyd_y_all[ic, step] = float(Fy_hyd)
            F_ext_x_all[ic, step] = Fx_ext_step
            F_ext_y_all[ic, step] = Fy_ext_step
            F_ext_mag_step = np.hypot(Fx_ext_step, Fy_ext_step)
            if F_ext_mag_step > 1.0 and np.isfinite(Fx_hyd) \
                    and np.isfinite(Fy_hyd):
                ux = -Fx_ext_step / F_ext_mag_step
                uy = -Fy_ext_step / F_ext_mag_step
                force_balance_projection_all[ic, step] = (
                    float(Fx_hyd) * ux + float(Fy_hyd) * uy)
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
            contact_clamp_event_count_all[ic, step] = int(step_event_count)
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
        # Stage J — close-out per-config Ausas counters.
        if ausas_state is not None:
            ausas_state_reset_count[ic] = int(ausas_state.reset_count)
            ausas_failed_step_count[ic] = int(
                ausas_state.failed_step_count)
            ausas_rejected_commit_count[ic] = int(
                ausas_state.rejected_commit_count)
        t_cfg = _time.time() - t_cfg_start
        cfg_times.append(t_cfg)
        n_solver_fail = int(solver_failed_count[ic])
        n_retry_rec = int(retry_recovered_count[ic])
        n_retry_exh = int(retry_exhausted_count[ic])
        # Stage Diesel Transient ClampAccounting Fix — print steps
        # and events on separate lines so the "95.8% шагов в clamp"
        # misreading from earlier runs (which actually counted up to
        # 3 events per step) cannot recur.
        n_completed_for_print = int(steps_completed_arr[ic])
        n_step_clamped = int(np.sum(
            contact_clamp_all[ic, :n_completed_for_print]))
        n_clamp_events = int(np.sum(
            contact_clamp_event_count_all[ic, :n_completed_for_print]))
        denom = max(n_completed_for_print, 1)
        print(f"    Contact steps : {n_step_clamped} / "
              f"{n_completed_for_print} steps "
              f"({100.0 * n_step_clamped / denom:.1f}%)")
        print(f"    Contact events: {n_clamp_events} total "
              f"(predictor + substep + final, up to ~3 events/step)")
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
    # Stage Diesel Transient AngleWeighted Metrics — also build the
    # angle-weighted envelope dicts (full / last-cycle) so the
    # applicable gate can be evaluated on Δφ-weighted fractions, and
    # surface them in the results for the summary writer.
    angle_weighted_full_per_cfg: List[Dict[str, float]] = []
    angle_weighted_last_per_cfg: List[Dict[str, float]] = []
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
        # Build the full-run + last-cycle angle-weighted statistics.
        full_aw = compute_angle_weighted_envelope(
            dphi=d_phi_per_step_all[ic, :],
            valid_no_clamp=valid_no_clamp_all[ic, :],
            contact_clamp=contact_clamp_all[ic, :],
            valid_dynamic=valid_dynamic_all[ic, :],
            solver_success=solver_success_all[ic, :],
            phi_mod=phi_mod_per_step_all[ic, :],
            n_completed=n_completed,
            firing_sector_deg=firing_sector_deg,
            last_cycle_mask=None,
        )
        if n_completed > 0:
            last_mask_full = np.zeros(n_steps, dtype=bool)
            last_mask_full[last_start:n_completed] = True
        else:
            last_mask_full = np.zeros(n_steps, dtype=bool)
        last_aw = compute_angle_weighted_envelope(
            dphi=d_phi_per_step_all[ic, :],
            valid_no_clamp=valid_no_clamp_all[ic, :],
            contact_clamp=contact_clamp_all[ic, :],
            valid_dynamic=valid_dynamic_all[ic, :],
            solver_success=solver_success_all[ic, :],
            phi_mod=phi_mod_per_step_all[ic, :],
            n_completed=n_completed,
            firing_sector_deg=firing_sector_deg,
            last_cycle_mask=last_mask_full,
        )
        angle_weighted_full_per_cfg.append(full_aw)
        angle_weighted_last_per_cfg.append(last_aw)
        ok, reason = classify_envelope_per_config(
            n_completed=n_completed,
            solver_success_count=ss_count,
            valid_dynamic_count=vd_count,
            valid_no_clamp_count=vnc_count,
            retry_exhausted_count=int(retry_exhausted_count[ic]),
            aborted=bool(aborted_arr[ic]),
            angle_weighted_full=full_aw,
            use_angle_weighted=True,
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
        # Stage Diesel Transient ClampAccounting Fix — per-step
        # event count (int, may exceed 1 per step). Distinct from
        # ``contact_clamp`` (boolean step-mask).
        "contact_clamp_event_count": contact_clamp_event_count_all,
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
        # Stage Diesel Transient AngleWeighted Metrics.
        "d_phi_per_step": d_phi_per_step_all,
        "phi_mod_per_step": phi_mod_per_step_all,
        "angle_weighted_full": angle_weighted_full_per_cfg,
        "angle_weighted_last_cycle": angle_weighted_last_per_cfg,
        # Stage Diesel Transient PeakWindow GridDiagnostic.
        "peak_lo_deg": float(peak_lo_deg),
        "peak_hi_deg": float(peak_hi_deg),
        "N_phi_grid": int(N_phi_eff),
        "N_z_grid": int(N_z_eff),
        "texture_resolution_diagnostic": texture_res_diag,
        # Stage J followup-2 — reproducibility marker.
        "seed": int(seed),
        # Stage J — Ausas dynamic + groove diagnostics. The legacy
        # half-Sommerfeld path produces zero-filled Ausas arrays so
        # downstream consumers can read these unconditionally.
        "cavitation_model": cavitation_str,
        "texture_kind": str(texture_kind),
        "groove_preset": (str(groove_preset)
                            if groove_preset is not None else None),
        "groove_preset_resolved": groove_preset_resolved,
        "groove_relief_stats": groove_relief_stats,
        "fidelity": (str(fidelity) if fidelity is not None else None),
        "ausas_options": dict(ausas_options or {}),
        # Stage J fu-2 Task 29 — dump-path counters surfaced in
        # the summary writer's "Ausas failed one-step dumps" block.
        "ausas_dump_directory": dump_failed_one_step_dir,
        "ausas_dump_limit": int(dump_failed_one_step_limit),
        "ausas_dump_written": int(_dump_counters.written),
        "ausas_dump_suppressed": int(
            _dump_counters.suppressed_after_limit),
        "ausas_dump_write_failed": int(
            _dump_counters.write_failed),
        "ausas_dump_by_trigger": dict(_dump_counters.by_trigger),
        "ausas_converged": ausas_converged_all,
        "ausas_n_inner": ausas_n_inner_all,
        "ausas_cav_frac": ausas_cav_frac_all,
        "ausas_theta_min": ausas_theta_min_all,
        "ausas_theta_max": ausas_theta_max_all,
        "ausas_state_reset_count": ausas_state_reset_count,
        "ausas_failed_step_count": ausas_failed_step_count,
        "ausas_rejected_commit_count": ausas_rejected_commit_count,
        "ausas_residual": ausas_residual_all,
        "ausas_dt_tau": ausas_dt_tau_all,
        # Stage J fu-2 Step 10 — per-step coupling diagnostics
        # (Gate 3 schema). Globals are computed at write-time by
        # the summary writer; per-step arrays are written here for
        # post-fact analysis.
        "stage_j_picard_shrinks": coupling_picard_shrinks_all,
        "stage_j_mech_relax_min_seen": coupling_mech_relax_min_seen_all,
        "stage_j_fp_converged": coupling_fp_converged_all,
        "stage_j_n_trials": coupling_n_trials_all,
        "stage_j_rejection_reason": coupling_rejection_reason_all,
        # Stage J fu-2 Task 32 — commit-semantics arrays.
        "final_trial_status": final_trial_status_all,
        "committed_state_status": committed_state_status_all,
        "accepted_state_source": accepted_state_source_all,
        "committed_state_is_finite": committed_state_is_finite_all,
        "final_trial_failure_kind": final_trial_failure_kind_all,
        "final_trial_residual": final_trial_residual_all,
        "final_trial_n_inner": final_trial_n_inner_all,
        "F_hyd_x": F_hyd_x_all,
        "F_hyd_y": F_hyd_y_all,
        "F_ext_x": F_ext_x_all,
        "F_ext_y": F_ext_y_all,
        "force_balance_projection": force_balance_projection_all,
    }
