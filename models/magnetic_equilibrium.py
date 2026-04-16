"""Shared driver: 2D equilibrium + continuation + acceptance.

Используется в scripts/run_mag_smooth_continuation.py и
scripts/run_mag_textured_compare.py. Единый pipeline — чтобы
smooth и textured runs сохраняли сопоставимость.

Pipeline:
  find_equilibrium(H_builder, mag_model, W_applied, ...)
    → EquilibriumResult
      * X, Y, eps, attitude_deg
      * Fx_hydro, Fy_hydro, Fx_mag, Fy_mag
      * h_min, p_max, cav_frac, friction
      * rel_residual, n_iter, converged (bool)

  run_continuation(targets, H_builder, baseline, W_applied, ...)
    → список EquilibriumResult с меткой accepted/failed
"""
import numpy as np
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List

from .magnetic_force import (
    BaseMagneticForceModel,
    calibrate_Kmag_from_baseline_projection,
)


@dataclass
class EquilibriumResult:
    X: float
    Y: float
    eps: float
    attitude_deg: float
    Fx_hydro: float
    Fy_hydro: float
    Fx_mag: float
    Fy_mag: float
    h_min: float
    p_max: float
    cav_frac: float
    friction: float
    rel_residual: float
    n_iter: int
    converged: bool
    unload_share_target: float = 0.0
    unload_share_actual: float = 0.0
    hydro_share_actual: float = 0.0
    K_mag: float = 0.0


def _line_search(X, Y, dX, dY, mag_model, W_applied,
                  H_and_force, current_rel_R,
                  max_backtracks=12, eps_max=0.90,
                  armijo_c=1e-4):
    """Backtracking line search с Armijo-условием по ‖R‖.

    Принимает наибольший λ ∈ {1, 0.5, 0.25, ...}, при котором:
      1. ε_new < eps_max
      2. rel_R_new ≤ current_rel_R · (1 − armijo_c · λ)
         (мягкое уменьшение residual; armijo_c=1e-4 стандарт)

    Returns (X_new, Y_new, rel_R_new, Fx_h, Fy_h, Fx_m, Fy_m,
             h_min, p_max, cav, fr, lam) или None если ни один λ
    не прошёл оба условия.
    """
    Wa_norm = float(np.linalg.norm(W_applied))
    lam = 1.0
    for _ in range(max_backtracks + 1):
        X_try = X + lam * dX
        Y_try = Y + lam * dY
        eps_try = np.sqrt(X_try**2 + Y_try**2)
        if eps_try >= eps_max:
            lam *= 0.5
            continue
        Fx_h, Fy_h, h_min, p_max, cav, fr, P, theta = H_and_force(X_try, Y_try)
        Fx_m, Fy_m = mag_model.force(X_try, Y_try)
        Rx = Fx_h + Fx_m - W_applied[0]
        Ry = Fy_h + Fy_m - W_applied[1]
        rel_R = np.sqrt(Rx**2 + Ry**2) / max(Wa_norm, 1e-20)
        # Armijo condition: residual должен уменьшаться
        if rel_R <= current_rel_R * (1.0 - armijo_c * lam):
            return (X_try, Y_try, rel_R, Fx_h, Fy_h, Fx_m, Fy_m,
                    h_min, p_max, cav, fr, lam)
        lam *= 0.5
    return None


def find_equilibrium(H_and_force, mag_model, W_applied,
                      X0=0.0, Y0=-0.4,
                      max_iter=50, tol=1e-4,
                      step_cap=0.05, eps_max=0.90):
    """Newton-Raphson 2D с line search.

    Parameters
    ----------
    H_and_force : callable(X, Y) -> (Fx_hydro, Fy_hydro, h_min, p_max,
                                       cav_frac, friction, P, theta)
        Пользовательская функция (делает make_H + PS solve + metrics).
    mag_model : BaseMagneticForceModel
    W_applied : np.ndarray shape (2,)

    Returns
    -------
    EquilibriumResult
    """
    Wa_norm = float(np.linalg.norm(W_applied))
    X, Y = float(X0), float(Y0)
    dXY = 1e-5

    # Init evaluation
    Fx_h, Fy_h, h_min, p_max, cav_frac, friction, P, theta = H_and_force(X, Y)
    Fx_m, Fy_m = mag_model.force(X, Y)
    Rx = Fx_h + Fx_m - W_applied[0]
    Ry = Fy_h + Fy_m - W_applied[1]
    rel_R = np.sqrt(Rx**2 + Ry**2) / max(Wa_norm, 1e-20)
    n_it = 0
    converged = False

    while n_it < max_iter:
        if rel_R < tol:
            converged = True
            break

        # Jacobian
        J = np.zeros((2, 2))
        for col, (dX_, dY_) in enumerate([(dXY, 0.0), (0.0, dXY)]):
            Fxp, Fyp, _, _, _, _, _, _ = H_and_force(X + dX_, Y + dY_)
            Fxm_p, Fym_p = mag_model.force(X + dX_, Y + dY_)
            J[0, col] = ((Fxp + Fxm_p) - (Fx_h + Fx_m)) / dXY
            J[1, col] = ((Fyp + Fym_p) - (Fy_h + Fy_m)) / dXY

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-30:
            break
        dX = -(J[1, 1] * Rx - J[0, 1] * Ry) / det
        dY = -(-J[1, 0] * Rx + J[0, 0] * Ry) / det

        # Step cap
        cap = step_cap / max(abs(dX), abs(dY), 1e-20)
        if cap < 1.0:
            dX *= cap
            dY *= cap

        # Line search
        ls = _line_search(X, Y, dX, dY, mag_model, W_applied,
                           H_and_force, rel_R,
                           max_backtracks=8, eps_max=eps_max)
        if ls is None:
            # Нет шага с уменьшением ‖R‖ — остановиться
            break
        (X, Y, rel_R, Fx_h, Fy_h, Fx_m, Fy_m,
         h_min, p_max, cav_frac, friction, _lam) = ls
        n_it += 1

    eps = float(np.sqrt(X**2 + Y**2))
    attitude = float(np.rad2deg(np.arctan2(Y, X)))
    e_resist = -W_applied / max(Wa_norm, 1e-20)
    unload = float((Fx_m * e_resist[0] + Fy_m * e_resist[1]) / max(Wa_norm, 1e-20))
    hydro_share = float((Fx_h * e_resist[0] + Fy_h * e_resist[1]) / max(Wa_norm, 1e-20))

    return EquilibriumResult(
        X=float(X), Y=float(Y), eps=eps, attitude_deg=attitude,
        Fx_hydro=float(Fx_h), Fy_hydro=float(Fy_h),
        Fx_mag=float(Fx_m), Fy_mag=float(Fy_m),
        h_min=float(h_min), p_max=float(p_max),
        cav_frac=float(cav_frac), friction=float(friction),
        rel_residual=float(rel_R), n_iter=int(n_it),
        converged=bool(converged),
        unload_share_actual=unload,
        hydro_share_actual=hydro_share,
        K_mag=float(mag_model.scale),
    )


def run_continuation(targets, baseline_X, baseline_Y, W_applied,
                      mag_model_template, H_and_force,
                      X0_seed=None, Y0_seed=None,
                      min_substep=0.01, tol=1e-3,
                      step_cap=0.05, eps_max=0.90,
                      verbose=True):
    """Continuation по списку unload_share_target.

    Правила:
      * continuation seed — последняя успешно сошедшаяся точка
      * если target не сходится до rel_residual<tol, substeps по
        середине между prev_target и target, пока шаг >= min_substep
      * при failure — вернуть список с accepted меткой False для
        оставшихся targets, continuation прекращается.

    Returns: list of (target, EquilibriumResult, accepted_bool)
    """
    out = []
    X_seed = X0_seed if X0_seed is not None else baseline_X
    Y_seed = Y0_seed if Y0_seed is not None else baseline_Y
    prev_target = 0.0

    def solve_at_target(tg, X_start, Y_start):
        """Один solve с калибровкой K_mag."""
        if tg == 0.0:
            m = copy.copy(mag_model_template)
            m.scale = 0.0
        else:
            m = calibrate_Kmag_from_baseline_projection(
                mag_model_template, baseline_X, baseline_Y,
                W_applied, tg)
        r = find_equilibrium(H_and_force, m, W_applied,
                              X0=X_start, Y0=Y_start,
                              tol=tol, step_cap=step_cap, eps_max=eps_max)
        r.unload_share_target = float(tg)
        return r, m

    accept_chain_broken = False

    for target in targets:
        if accept_chain_broken:
            # Пропускаем, но логируем
            out.append((target, None, False))
            continue

        r, m_used = solve_at_target(target, X_seed, Y_seed)

        if r.converged and r.rel_residual < tol:
            out.append((target, r, True))
            X_seed, Y_seed = r.X, r.Y
            prev_target = target
            if verbose:
                print(f"  target={target*100:5.2f}%: ✓ ε={r.eps:.4f}, "
                      f"h_min={r.h_min*1e6:.2f}μm, "
                      f"p_max={r.p_max/1e6:.2f}MPa, "
                      f"res={r.rel_residual:.1e}, n_it={r.n_iter}, "
                      f"unload={r.unload_share_actual:+.4f}, "
                      f"hydro={r.hydro_share_actual:+.4f}")
            continue

        # Не сошёлся → substep refinement
        if verbose:
            print(f"  target={target*100:5.2f}%: ✗ "
                  f"(ε={r.eps:.4f}, res={r.rel_residual:.1e}) → substeps")

        sub_lo = prev_target
        sub_hi = target
        # Bisection-like: последовательно вставлять mid и сдвигать seed
        success_final = False
        n_subs = 0
        while (sub_hi - sub_lo) >= min_substep and n_subs < 12:
            mid = 0.5 * (sub_lo + sub_hi)
            r_mid, _ = solve_at_target(mid, X_seed, Y_seed)
            n_subs += 1
            if r_mid.converged and r_mid.rel_residual < tol:
                if verbose:
                    print(f"    sub {mid*100:5.2f}%: ✓ ε={r_mid.eps:.4f}, "
                          f"res={r_mid.rel_residual:.1e}")
                sub_lo = mid
                X_seed, Y_seed = r_mid.X, r_mid.Y
                # Попробовать target с нового seed
                r_try, _ = solve_at_target(target, X_seed, Y_seed)
                if r_try.converged and r_try.rel_residual < tol:
                    out.append((target, r_try, True))
                    X_seed, Y_seed = r_try.X, r_try.Y
                    prev_target = target
                    success_final = True
                    if verbose:
                        print(f"  target={target*100:5.2f}%: ✓ after "
                              f"{n_subs} substeps, ε={r_try.eps:.4f}, "
                              f"res={r_try.rel_residual:.1e}")
                    break
            else:
                sub_hi = mid
                if verbose:
                    print(f"    sub {mid*100:5.2f}%: ✗ "
                          f"(res={r_mid.rel_residual:.1e})")

        if not success_final:
            out.append((target, r, False))
            accept_chain_broken = True
            if verbose:
                print(f"  target={target*100:5.2f}%: FAILED — "
                      f"continuation stopped")

    return out
