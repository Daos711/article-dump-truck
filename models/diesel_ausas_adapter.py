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

    ``P``, ``theta`` and ``H_prev`` live on the **unpadded physical
    grid** ``(N_z, N_phi)`` — same shape and indexing as the diesel
    ``setup_grid`` output. ``ausas_unsteady_one_step_gpu`` does its
    own ``_pack_ghosts(..., periodic_phi=True)`` internally so the
    adapter must NOT pre-pad the seam ghost columns; doing so
    triggers a double-wrap that corrupts every step (Stage J
    integration regression, Bug 3).
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
        (``theta = 1``), zero pressure, unit gap. Allocated on the
        unpadded ``(N_z, N_phi)`` physical grid."""
        return cls(
            P=np.zeros((int(N_z), int(N_phi)), dtype=float),
            theta=np.ones((int(N_z), int(N_phi)), dtype=float),
            H_prev=np.ones((int(N_z), int(N_phi)), dtype=float),
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
    """Mirror one ghost column on each circumferential side of a
    ``(N_z, N_phi)`` field, returning ``(N_z, N_phi + 2)``.

    .. note::
        Stage J integration regression (Bug 3) — this helper is
        **not** called inside the live ``ausas_one_step_with_state``
        path. ``ausas_unsteady_one_step_gpu`` performs
        ``_pack_ghosts(..., periodic_phi=True)`` internally, so
        pre-padding triggers a double-wrap. The helper survives
        only as a debug utility for offline introspection.

    Diesel ``setup_grid`` is endpoint-free on phi, so column 0 and
    column ``N_phi`` are conjugate (``Phi[N_phi] == Phi[0] + 2π``).
    The seam wrap is::

        col_left_ghost  = field[:, -1]        # phi = -dphi
        col_right_ghost = field[:, 0]         # phi = 2π
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
    columns; returns the physical ``(N_z, N_phi)`` array. Same
    note as ``pad_phi_for_ausas``: not called by the live path
    after Bug 3."""
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
    auto-discovery — the next call to :func:`_resolve_ausas_backend`
    will re-probe for the real GPU backend.

    Stage J integration regression follow-up — the previous
    implementation set ``_AUSAS_BACKEND_PROBED = True`` even when
    ``fn is None``, which permanently locked the resolver onto
    ``None``. Any contract test that installed a stub and then
    cleaned up with ``set_ausas_backend_for_tests(None)`` left the
    real-backend integration tests unable to ever discover the GPU
    package. Clearing the cache is the only correct semantics here.
    """
    global _AUSAS_ONE_STEP_BACKEND, _AUSAS_BACKEND_PROBED
    if fn is None:
        _AUSAS_ONE_STEP_BACKEND = None
        _AUSAS_BACKEND_PROBED = False
    else:
        _AUSAS_ONE_STEP_BACKEND = fn
        _AUSAS_BACKEND_PROBED = True


# ─── Commit-once one-step wrapper ──────────────────────────────────


_DEFAULT_AUSAS_TOL = 1e-6
_DEFAULT_AUSAS_MAX_INNER = 200


def ausas_one_step_with_state(
    state: DieselAusasState,
    *,
    H_curr_phys: np.ndarray,
    dt_s: float,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    extra_options: Optional[Dict[str, Any]] = None,
    commit: bool,
) -> Dict[str, Any]:
    """Advance the Ausas state by one mechanical step.

    Stage J integration regression — Bugs 1/2/3 fixed:

    * **Bug 1** — ``eta`` and ``omega`` are NOT forwarded to
      ``ausas_unsteady_one_step_gpu``. The dynamic Ausas solver uses
      a different non-dimensionalisation; viscosity enters through
      the discretisation coefficients, not as an explicit kwarg.
      The adapter signature therefore no longer accepts them.
    * **Bug 2** — the 4-tuple return shape is interpreted as
      ``(P, theta, residual, n_inner)`` (matches the GPU solver
      docstring), and convergence is derived from
      ``n_inner < max_inner`` AND ``residual <= tol`` (legacy
      shapes that already report a ``converged`` flag still honour
      it).
    * **Bug 3** — state arrays live on the unpadded physical grid
      ``(N_z, N_phi)``; the adapter does NOT pre-pad. The GPU
      solver does its own ``_pack_ghosts(..., periodic_phi=True)``,
      so adding ghost columns here would double-wrap.

    Parameters
    ----------
    state
        ``DieselAusasState`` carrying the previous accepted
        ``P``, ``theta`` and ``H_prev`` on the unpadded physical grid.
    H_curr_phys
        Current candidate film thickness on the physical
        ``(N_z, N_phi)`` diesel grid.
    dt_s, d_phi, d_Z, R, L
        Solver kwargs forwarded to ``ausas_unsteady_one_step_gpu``.
    extra_options
        Optional dict merged into the solver kwargs (e.g.
        ``omega_p``, ``omega_theta``, ``alpha``, ``tol``,
        ``max_inner``, ``check_every``, ``scheme``).
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
        ``ok``        — solver convergence flag derived from the
                          residual / inner-iteration budget.
        ``n_inner``   — inner-iteration count reported by the solver.
        ``residual``  — final residual reported by the solver
                          (``nan`` if the legacy short shape is in
                          use).
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

    # Unpadded physical grid in / out — solver pads internally.
    H_curr = np.asarray(H_curr_phys)

    kwargs = dict(
        H_curr=H_curr,
        H_prev=state.H_prev,
        P_prev=state.P,
        theta_prev=state.theta,
        dt=float(dt_s),
        d_phi=float(d_phi),
        d_Z=float(d_Z),
        R=float(R),
        L=float(L),
    )
    if extra_options:
        kwargs.update(extra_options)
    # Convergence threshold for the derived ``ok`` flag — matches
    # the kwarg the runner can override.
    tol = float(kwargs.get("tol", _DEFAULT_AUSAS_TOL))
    max_inner = int(kwargs.get("max_inner", _DEFAULT_AUSAS_MAX_INNER))

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
            residual=float("nan"),
            cav_frac=float("nan"),
            theta_min=float("nan"),
            theta_max=float("nan"),
            reason=f"ausas_one_step_failed: {type(exc).__name__}: {exc}",
        )

    P_phys, theta_phys, residual, n_inner, ok_explicit = (
        _unpack_ausas_return(out))

    # Derive ``ok``: short shapes return ok_explicit=None; for those
    # use n_inner < max_inner AND residual <= tol (residual stays
    # NaN in 2-/3-tuple legacy shapes, in which case we trust
    # n_inner < max_inner alone).
    if ok_explicit is None:
        ok_residual = (np.isfinite(residual) and residual <= tol) \
            or (not np.isfinite(residual))
        ok = bool(n_inner < max_inner) and bool(ok_residual)
    else:
        ok = bool(ok_explicit)

    cav = float(np.mean(theta_phys < 1.0)) if theta_phys.size else 0.0

    finite = (
        np.all(np.isfinite(P_phys))
        and np.all(np.isfinite(theta_phys))
    )
    if not finite:
        if commit:
            state.failed_step_count += 1
        return dict(
            P_phys=P_phys, theta_phys=theta_phys,
            ok=False, n_inner=int(n_inner),
            residual=float(residual)
                if np.isfinite(residual) else float("nan"),
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
            state.P = np.asarray(P_phys)
            state.theta = np.asarray(theta_phys)
            state.H_prev = H_curr
            state.step_index += 1
            state.time_s += float(dt_s)
            state.dt_last_s = float(dt_s)
            state.valid = True

    return dict(
        P_phys=P_phys, theta_phys=theta_phys,
        ok=bool(ok), n_inner=int(n_inner),
        residual=float(residual)
            if np.isfinite(residual) else float("nan"),
        cav_frac=cav,
        theta_min=float(np.min(theta_phys))
            if theta_phys.size else float("nan"),
        theta_max=float(np.max(theta_phys))
            if theta_phys.size else float("nan"),
        reason=("ok" if ok else "ausas_not_converged"),
    )


def _unpack_ausas_return(
    out: Any,
) -> Tuple[np.ndarray, np.ndarray, float, int, Optional[bool]]:
    """Normalise the variety of ``ausas_unsteady_one_step_gpu``
    return shapes onto ``(P, theta, residual, n_inner, ok_explicit)``.

    ``ok_explicit`` is ``None`` when the shape carries no convergence
    flag — in which case the caller derives ``ok`` from
    ``n_inner < max_inner`` and ``residual <= tol``.

    Accepts:
        ``(P, theta)``
            no diagnostics → residual=NaN, n_inner=0,
            ok_explicit=None.
        ``(P, theta, n_inner)``
            legacy short shape → residual=NaN, ok_explicit=None.
        ``(P, theta, residual, n_inner)``
            **canonical real-solver shape** (Stage J Bug 2 fix):
            third element is the **residual** scalar, fourth is
            the inner-iteration count. ok_explicit=None.
        ``(P, theta, residual, n_inner, converged)``
            future-proofing extension that does include an explicit
            convergence flag.
        ``dict(P=..., theta=..., residual=..., n_inner=...,
               converged=...)``
            keyword shape — every field optional except ``P`` and
            ``theta``.
    """
    if isinstance(out, dict):
        P = np.asarray(out["P"])
        theta = np.asarray(out["theta"])
        residual = float(out.get("residual", float("nan")))
        n_inner = int(out.get("n_inner", 0))
        ok_explicit: Optional[bool] = (
            bool(out["converged"]) if "converged" in out else None)
        return P, theta, residual, n_inner, ok_explicit
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "Unrecognised ausas_unsteady_one_step_gpu return type: "
            f"{type(out)!r}"
        )
    if len(out) == 2:
        P, theta = out
        return (np.asarray(P), np.asarray(theta),
                float("nan"), 0, None)
    if len(out) == 3:
        P, theta, n_inner = out
        return (np.asarray(P), np.asarray(theta),
                float("nan"), int(n_inner), None)
    if len(out) == 4:
        # Canonical real-solver shape: (P, theta, residual, n_inner).
        P, theta, residual, n_inner = out
        return (np.asarray(P), np.asarray(theta),
                float(residual), int(n_inner), None)
    if len(out) >= 5:
        P, theta, residual, n_inner, ok = out[:5]
        return (np.asarray(P), np.asarray(theta),
                float(residual), int(n_inner), bool(ok))
    raise TypeError(
        f"Unrecognised ausas_unsteady_one_step_gpu return shape "
        f"{len(out)} of {type(out)!r}"
    )
