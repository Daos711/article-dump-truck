"""Stage J — diesel-side adapter for the dynamic Ausas JFO solver.

This module owns three responsibilities:

1. ``DieselAusasState`` — the per-config dynamic state that survives
   between mechanical (Verlet) steps: previous accepted pressure
   ``P``, fluid fraction ``theta``, and previous gap ``H_prev``.
2. ``pad_phi_for_ausas`` / ``unpad_phi_from_ausas`` — explicit
   conversion between the diesel ``setup_grid(endpoint=False)``
   physical grid and an Ausas seam-padded grid (one ghost column on
   each circumferential end so the solver can apply the wrap-around
   semantics without re-discovering them).
3. ``ausas_one_step_with_state`` — one-step wrapper around
   ``ausas_unsteady_one_step_gpu`` that respects "commit only the
   accepted Verlet candidate"; trial calls receive ``commit=False``
   and never mutate the state.

The adapter does **not** import ``ausas_unsteady_one_step_gpu`` at
module-load time — the import is deferred until the first call so
contract tests on the legacy (half-Sommerfeld) path do not depend
on the GPU package. Any failure to import is caught and surfaced as
a clear ``ImportError`` with the dynamic-Ausas wiring instructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ─── State container ───────────────────────────────────────────────


@dataclass
class DieselAusasState:
    """Per-config dynamic state for the diesel Ausas path.

    ``P``, ``theta`` and ``H_prev`` live on the **padded** grid
    (Ausas seam ghost columns included) so the solver does not need
    extra boundary handling. Use the adapter helpers to convert
    to/from the diesel physical grid.
    """
    P: np.ndarray
    theta: np.ndarray
    H_prev: np.ndarray
    step_index: int = 0
    time_s: float = 0.0
    dt_last_s: float = 0.0
    valid: bool = True
    reset_count: int = 0
    failed_step_count: int = 0
    rejected_commit_count: int = 0

    @classmethod
    def cold_start(cls, *, N_phi: int, N_z: int) -> "DieselAusasState":
        """Cold-start state for an Ausas run: full-film
        (``theta = 1``), zero pressure, gap padded later from the
        first mechanical step."""
        N_phi_pad = int(N_phi) + 2
        return cls(
            P=np.zeros((int(N_z), N_phi_pad), dtype=float),
            theta=np.ones((int(N_z), N_phi_pad), dtype=float),
            H_prev=np.ones((int(N_z), N_phi_pad), dtype=float),
            step_index=0,
            time_s=0.0,
            dt_last_s=0.0,
            valid=True,
        )

    def reset(self, *, N_phi: int, N_z: int) -> None:
        """Reset to cold-start state. Bumps ``reset_count`` so the
        runner / summary writer can surface it."""
        cs = DieselAusasState.cold_start(N_phi=N_phi, N_z=N_z)
        self.P = cs.P
        self.theta = cs.theta
        self.H_prev = cs.H_prev
        self.valid = True
        self.dt_last_s = 0.0
        self.reset_count += 1


# ─── Phi padding helpers ───────────────────────────────────────────


def pad_phi_for_ausas(field: np.ndarray) -> np.ndarray:
    """Convert a (N_z, N_phi) physical field to a (N_z, N_phi + 2)
    seam-padded field by mirroring one ghost column on each side.

    The diesel ``setup_grid`` is endpoint-free on phi, so column 0
    and column N_phi are conjugate (``Phi[N_phi] == Phi[0] + 2π``).
    The seam wrap is therefore:

        col_left_ghost  = field[:, -1]        # i.e. phi = -dphi
        col_right_ghost = field[:, 0]         # i.e. phi = 2π
    """
    a = np.asarray(field)
    if a.ndim != 2:
        raise ValueError(
            f"pad_phi_for_ausas expects a 2D array, got shape {a.shape}"
        )
    N_z, N_phi = a.shape
    out = np.empty((N_z, N_phi + 2), dtype=a.dtype)
    out[:, 1:-1] = a
    out[:, 0] = a[:, -1]
    out[:, -1] = a[:, 0]
    return out


def unpad_phi_from_ausas(field_pad: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pad_phi_for_ausas`. Drops the two ghost
    columns and returns the physical (N_z, N_phi) array used by the
    diesel force / friction integrations."""
    a = np.asarray(field_pad)
    if a.ndim != 2:
        raise ValueError(
            f"unpad_phi_from_ausas expects a 2D array, got shape {a.shape}"
        )
    if a.shape[1] < 3:
        raise ValueError(
            f"unpad_phi_from_ausas: padded width {a.shape[1]} < 3"
        )
    return a[:, 1:-1]


# ─── Backend dispatch ──────────────────────────────────────────────


_AUSAS_ONE_STEP_BACKEND: Optional[Callable[..., Any]] = None
_AUSAS_BACKEND_PROBED: bool = False


def _resolve_ausas_backend() -> Optional[Callable[..., Any]]:
    """Locate ``ausas_unsteady_one_step_gpu`` lazily.

    Returns the callable if the GPU solver is importable in the
    current environment; ``None`` otherwise. The result is cached so
    repeat calls are O(1).
    """
    global _AUSAS_ONE_STEP_BACKEND, _AUSAS_BACKEND_PROBED
    if _AUSAS_BACKEND_PROBED:
        return _AUSAS_ONE_STEP_BACKEND
    _AUSAS_BACKEND_PROBED = True
    try:
        from reynolds_solver.cavitation.ausas import (  # type: ignore
            solver_dynamic_gpu as _sdg,
        )
    except Exception:
        _AUSAS_ONE_STEP_BACKEND = None
        return None
    fn = getattr(_sdg, "ausas_unsteady_one_step_gpu", None)
    _AUSAS_ONE_STEP_BACKEND = fn
    return fn


def set_ausas_backend_for_tests(
    fn: Optional[Callable[..., Any]],
) -> None:
    """Test hook — install a Python stub as the Ausas one-step
    backend so contract tests can verify the adapter without a GPU.

    Pass ``None`` to clear the override and return to lazy
    auto-discovery."""
    global _AUSAS_ONE_STEP_BACKEND, _AUSAS_BACKEND_PROBED
    _AUSAS_ONE_STEP_BACKEND = fn
    _AUSAS_BACKEND_PROBED = True


# ─── Commit-once one-step wrapper ──────────────────────────────────


def ausas_one_step_with_state(
    state: DieselAusasState,
    *,
    H_curr_phys: np.ndarray,
    dt_s: float,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    eta: float,
    omega: float,
    extra_options: Optional[Dict[str, Any]] = None,
    commit: bool,
) -> Dict[str, Any]:
    """Advance the Ausas state by one mechanical step.

    Parameters
    ----------
    state
        ``DieselAusasState`` carrying the previous accepted
        ``P``, ``theta`` and ``H_prev`` on the padded grid.
    H_curr_phys
        Current candidate film thickness on the physical (N_z, N_phi)
        diesel grid. The wrapper takes care of padding.
    dt_s, d_phi, d_Z, R, L, eta, omega
        Solver kwargs forwarded to ``ausas_unsteady_one_step_gpu``.
    extra_options
        Optional dict merged into the solver kwargs (e.g. ``omega_p``,
        ``omega_theta``, ``tol``, ``max_inner``, ``check_every``,
        ``scheme``).
    commit
        When ``False`` the state is **not** mutated — used for the
        Verlet trial path. When ``True`` the wrapper writes the new
        ``P``, ``theta``, ``H_prev``, ``step_index``, ``time_s`` and
        ``dt_last_s`` back into ``state``.

    Returns
    -------
    dict with keys:
        ``P_phys``    — pressure on the physical grid (callers
                          integrate this for force / friction).
        ``theta_phys``— fluid fraction on the physical grid.
        ``ok``        — solver convergence flag.
        ``n_inner``   — inner-iteration count reported by the solver.
        ``cav_frac``  — ``mean(theta_phys < 1)`` (dim-less cavitation
                          fraction).
        ``theta_min`` / ``theta_max``
        ``reason``    — human-readable solver status.
    """
    backend = _resolve_ausas_backend()
    if backend is None:
        raise ImportError(
            "ausas_unsteady_one_step_gpu is not importable in this "
            "environment. Install GPU_reynolds and ensure that "
            "``from reynolds_solver.cavitation.ausas.solver_dynamic_gpu"
            " import ausas_unsteady_one_step_gpu`` succeeds, or "
            "install a test backend via "
            "``models.diesel_ausas_adapter.set_ausas_backend_for_tests``."
        )

    H_curr_pad = pad_phi_for_ausas(H_curr_phys)

    kwargs = dict(
        H_curr=H_curr_pad,
        H_prev=state.H_prev,
        P_prev=state.P,
        theta_prev=state.theta,
        dt=float(dt_s),
        d_phi=float(d_phi),
        d_Z=float(d_Z),
        R=float(R),
        L=float(L),
        eta=float(eta),
        omega=float(omega),
    )
    if extra_options:
        kwargs.update(extra_options)

    try:
        out = backend(**kwargs)
    except Exception as exc:
        # Failure path: do NOT poison state, do NOT throw. The runner
        # treats the step as solver-failed and may choose to retry or
        # mark the step invalid.
        if commit:
            state.failed_step_count += 1
        return dict(
            P_phys=None,
            theta_phys=None,
            ok=False,
            n_inner=0,
            cav_frac=float("nan"),
            theta_min=float("nan"),
            theta_max=float("nan"),
            reason=f"ausas_one_step_failed: {type(exc).__name__}: {exc}",
        )

    P_pad, theta_pad, n_inner, ok = _unpack_ausas_return(out)

    P_phys = unpad_phi_from_ausas(P_pad)
    theta_phys = unpad_phi_from_ausas(theta_pad)
    cav = float(np.mean(theta_phys < 1.0)) if theta_phys.size else 0.0

    finite = (
        np.all(np.isfinite(P_pad))
        and np.all(np.isfinite(theta_pad))
    )
    if not finite:
        if commit:
            state.failed_step_count += 1
        return dict(
            P_phys=P_phys, theta_phys=theta_phys,
            ok=False, n_inner=int(n_inner),
            cav_frac=cav,
            theta_min=float(np.nanmin(theta_phys))
                if theta_phys.size else float("nan"),
            theta_max=float(np.nanmax(theta_phys))
                if theta_phys.size else float("nan"),
            reason="ausas_returned_nonfinite",
        )

    if commit:
        if not ok:
            # Solver explicitly reported non-convergence — count the
            # rejection but still leave the previous accepted state
            # intact (do not poison).
            state.rejected_commit_count += 1
        else:
            state.P = P_pad
            state.theta = theta_pad
            state.H_prev = H_curr_pad
            state.step_index += 1
            state.time_s += float(dt_s)
            state.dt_last_s = float(dt_s)
            state.valid = True

    return dict(
        P_phys=P_phys, theta_phys=theta_phys,
        ok=bool(ok), n_inner=int(n_inner),
        cav_frac=cav,
        theta_min=float(np.min(theta_phys))
            if theta_phys.size else float("nan"),
        theta_max=float(np.max(theta_phys))
            if theta_phys.size else float("nan"),
        reason=("ok" if ok else "ausas_not_converged"),
    )


def _unpack_ausas_return(
    out: Any,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Normalise the variety of return shapes the solver may use.

    Accepts:
        ``(P, theta)``                              (no diagnostics)
        ``(P, theta, n_inner)``                     (legacy)
        ``(P, theta, n_inner, converged)``          (modern)
        ``dict(P=..., theta=..., n_inner=..., converged=...)``
    """
    if isinstance(out, dict):
        P = np.asarray(out["P"])
        theta = np.asarray(out["theta"])
        n_inner = int(out.get("n_inner", 0))
        ok = bool(out.get("converged", True))
        return P, theta, n_inner, ok
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "Unrecognised ausas_unsteady_one_step_gpu return type: "
            f"{type(out)!r}"
        )
    if len(out) == 2:
        P, theta = out
        return np.asarray(P), np.asarray(theta), 0, True
    if len(out) == 3:
        P, theta, n_inner = out
        return np.asarray(P), np.asarray(theta), int(n_inner), True
    if len(out) >= 4:
        P, theta, n_inner, ok = out[:4]
        return (np.asarray(P), np.asarray(theta),
                int(n_inner), bool(ok))
    raise TypeError(
        f"Unrecognised ausas_unsteady_one_step_gpu return shape "
        f"{len(out)} of {type(out)!r}"
    )
