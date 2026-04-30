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

# Stage J fu-2 Task 29 — dump-writer module + DumpConfig / DumpCounters.
# Imported at module top so the adapter can read cfg.directory cheaply
# and the production path stays None-by-default (no I/O).
from models.diesel_ausas_dump_io import (
    DumpConfig,
    DumpCounters,
    OPTIONAL_FORCE_INPUT_KEYS,
    build_dump_filename,
    evaluate_triggers,
    write_dump_npz,
)


# ─── State container ───────────────────────────────────────────────


@dataclass
class DieselAusasState:
    """Per-config dynamic state for the diesel Ausas path.

    ``P``, ``theta`` and ``H_prev`` live on the **unpadded physical
    grid** ``(N_z, N_phi)`` — same shape and indexing as the diesel
    ``setup_grid`` output. The adapter (Stage J fu-2 ghost-grid
    migration) is the single locus of pad/unpad knowledge: it pads
    these arrays to ``(N_z, N_phi+2)`` immediately before forwarding
    to ``ausas_unsteady_one_step_gpu`` and unpads the backend's
    returned padded arrays before storing them back. Pipeline state
    therefore stays on the physical grid for forces, summary,
    npz schema, and runner-side bookkeeping.
    """
    P: np.ndarray
    theta: np.ndarray
    H_prev: np.ndarray
    step_index: int = 0
    time_s: float = 0.0
    dt_last_s: float = 0.0
    dt_ausas_last: float = 0.0
    valid: bool = True
    reset_count: int = 0
    failed_step_count: int = 0
    rejected_commit_count: int = 0

    @classmethod
    def from_initial_gap(
        cls,
        H_initial: np.ndarray,
        *,
        P_warm: Optional[np.ndarray] = None,
        theta_warm: Optional[np.ndarray] = None,
    ) -> "DieselAusasState":
        """Stage J Bug 5 — initialise from the actual first accepted
        diesel gap so step #1 has no artificial squeeze.

        ``H_prev`` is set to ``H_initial`` (NOT to a unit gap), and
        ``P`` / ``theta`` default to the identity (zero / full film)
        unless explicit warm-up arrays are supplied. The legacy
        :meth:`cold_start` constructor used ``H_prev = ones`` which
        injected a fictitious squeeze pulse on the first mechanical
        step — that is the regression this constructor closes.
        """
        H = np.asarray(H_initial, dtype=float).copy()
        return cls(
            P=(np.asarray(P_warm, dtype=float).copy()
               if P_warm is not None else np.zeros_like(H)),
            theta=(np.asarray(theta_warm, dtype=float).copy()
                   if theta_warm is not None else np.ones_like(H)),
            H_prev=H,
        )

    @classmethod
    def cold_start(cls, *, N_phi: int, N_z: int) -> "DieselAusasState":
        """Legacy cold-start (``H_prev = ones``).

        .. deprecated::
            Stage J Bug 5 — prefer :meth:`from_initial_gap` so the
            first mechanical step does not see an artificial squeeze
            pulse. ``cold_start`` is retained only for tests that
            still assert against the unit-gap shape; new runner
            code should not call it.
        """
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


# ─── Time non-dimensionalisation ──────────────────────────────────


def ausas_dt_from_physical(dt_s: float, omega_shaft: float) -> float:
    """Stage J Bug 4 — convert a physical timestep (seconds) to the
    Ausas non-dimensional time ``τ = ω · t``.

    The dynamic Ausas discretisation in
    ``ausas_unsteady_one_step_gpu`` uses the same non-dimensional
    pressure scale ``p_scale = 6 η ω (R/c)²`` as the diesel runner;
    this convention requires time to be measured in shaft-rotation
    units (``τ = ω t``), not physical seconds. The previous adapter
    forwarded ``dt_phys`` directly, which inflated the unsteady
    coefficient ``β = 2 d_phi² / dt`` by roughly ``ω_shaft ≈ 199``
    at 1900 rpm and made the squeeze term dominate the Couette
    pressure-generation term — converging to a physically wrong
    pressure field that then collapsed the orbit on the first step.

    This helper is the single-line conversion the adapter applies
    before forwarding ``dt`` to the GPU backend.
    """
    return float(dt_s) * float(omega_shaft)


# ─── Structured one-step result ────────────────────────────────────


@dataclass
class DieselAusasStepResult:
    """Stage J Bug 4-5 follow-up — replaces the legacy result dict.

    Carries the canonical solver outputs (``P_nd``, ``theta``,
    ``residual``, ``n_inner``), the derived ``converged`` flag, the
    committed gap ``H_curr``, and both timestep representations
    (``dt_phys_s`` in seconds and ``dt_ausas`` in non-dimensional
    shaft-rotation units) so the runner / summary writer can
    sanity-check the conversion without re-deriving it.

    Fields are populated even on failure (``converged=False``); the
    arrays may be ``None`` when the backend raised.
    """
    P_nd: Optional[np.ndarray]
    theta: Optional[np.ndarray]
    residual: float
    n_inner: int
    converged: bool
    H_curr: Optional[np.ndarray]
    dt_phys_s: float
    dt_ausas: float
    reason: str = "ok"
    # Stage J fu-2 Task 29 — pass-through fields from the
    # gpu-reynolds Task 12 dict-API. Empty / NaN defaults preserve
    # the legacy 4-tuple-return path (``failure_kind=""`` and
    # ``nonfinite_count=0`` map to "no info" → classifier falls
    # back to the numerical-residual rules).
    failure_kind: str = ""
    first_nan_field: str = ""
    first_nan_index: Tuple[int, ...] = ()
    first_nan_is_ghost: bool = False
    first_nan_is_axial_boundary: bool = False
    first_nan_is_phi_seam: bool = False
    nan_iter: int = -1
    nonfinite_count: int = 0
    residual_linf: float = float("nan")
    residual_rms: float = float("nan")
    residual_l2_abs: float = float("nan")

    @property
    def cav_frac(self) -> float:
        if self.theta is None or not self.theta.size:
            return float("nan")
        return float(np.mean(self.theta < 1.0))

    @property
    def theta_min(self) -> float:
        if self.theta is None or not self.theta.size:
            return float("nan")
        return float(np.min(self.theta))

    @property
    def theta_max(self) -> float:
        if self.theta is None or not self.theta.size:
            return float("nan")
        return float(np.max(self.theta))

    @property
    def p_nd_max(self) -> float:
        if self.P_nd is None or not self.P_nd.size:
            return float("nan")
        return float(np.max(self.P_nd))


def ausas_result_is_physical(
    result: DieselAusasStepResult, *, tol: float, max_inner: int,
) -> bool:
    """Stage J Bug 4-5 follow-up — promote convergence from "no
    exception" to a physical contract.

    Returns True iff every gate passes:

    * ``result.converged`` is True (set by the adapter from the
      solver-side ``residual``/``n_inner`` budget — see
      :func:`ausas_one_step_with_state`);
    * ``result.residual`` is finite and ``≤ tol``;
    * ``result.n_inner`` is strictly below ``max_inner``;
    * ``result.P_nd`` is finite element-wise;
    * ``result.theta`` is element-wise in ``[0, 1]``.

    Used by the runner to decide whether the step's pressure /
    forces are trustworthy. Failing the gate flips the step into
    ``solver_failed`` for envelope metrics — the abort policy is
    unchanged.
    """
    if not bool(result.converged):
        return False
    if not np.isfinite(result.residual) or result.residual > float(tol):
        return False
    if int(result.n_inner) >= int(max_inner):
        return False
    if result.P_nd is None or not np.all(np.isfinite(result.P_nd)):
        return False
    if result.theta is None:
        return False
    if (result.theta_min < -1e-9) or (result.theta_max > 1.0 + 1e-9):
        return False
    return True


# ─── Phi padding helpers ───────────────────────────────────────────


def pad_phi_for_ausas(field: np.ndarray) -> np.ndarray:
    """Mirror one ghost column on each circumferential side of a
    ``(N_z, N_phi)`` field, returning ``(N_z, N_phi + 2)``.

    Called by ``ausas_one_step_with_state`` immediately before the
    backend call. Diesel ``setup_grid`` is endpoint-free on phi,
    so column 0 and column ``N_phi`` are conjugate
    (``Phi[N_phi] == Phi[0] + 2π``). The seam wrap is::

        col_left_ghost  = field[:, -1]        # phi = -dphi
        col_right_ghost = field[:, 0]         # phi = 2π

    The current solver-side ``_pack_ghosts(..., periodic_phi=True)``
    is an idempotent in-place refresh
    (``arr[:, 0] = arr[:, N-2]; arr[:, -1] = arr[:, 1]``) and is
    compatible with adapter-side padding — it will simply re-mirror
    the seam ghosts on top of the values this helper already wrote.
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
    columns; returns the physical ``(N_z, N_phi)`` array. Called
    by ``ausas_one_step_with_state`` immediately after the backend
    call to restore physical-grid arrays for state commit and the
    ``DieselAusasStepResult`` payload."""
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


def ausas_one_step_with_state(
    state: DieselAusasState,
    *,
    H_curr: np.ndarray,
    dt_s: float,
    omega_shaft: float,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    extra_options: Optional[Dict[str, Any]] = None,
    commit: bool,
) -> DieselAusasStepResult:
    """Advance the Ausas state by one mechanical step.

    Stage J Bug 4 follow-up — ``omega_shaft`` is a required kwarg
    so the adapter can convert the physical timestep ``dt_s``
    (seconds) into the Ausas non-dimensional time
    ``dt_ausas = ω·dt_s`` that the GPU solver actually consumes.
    Forwarding ``dt_s`` directly inflated the unsteady coefficient
    by ``ω ≈ 199`` at 1900 rpm and made the orbit collapse on the
    first step.

    Stage J Bug 5 follow-up — the runner now initialises ``state``
    via :meth:`DieselAusasState.from_initial_gap` so ``H_prev``
    matches the actual first accepted gap; the legacy
    ``cold_start`` (``H_prev = ones``) injected an artificial
    squeeze pulse on step #1.

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
    * **Bug 3 (Stage J fu-2 ghost-grid migration)** — state arrays
      stay on the unpadded physical grid ``(N_z, N_phi)``, but the
      adapter is the single locus of pad/unpad: it calls
      :func:`pad_phi_for_ausas` on ``H_curr``, ``H_prev``,
      ``P_prev``, ``theta_prev`` immediately before the backend
      call (so the backend sees padded ``(N_z, N_phi+2)``), and
      :func:`unpad_phi_from_ausas` on the backend's returned
      ``P`` and ``theta`` before commit / return. ``d_phi`` stays
      at the physical spacing ``2π/N_phi``. The boundary
      convention (``periodic_phi=True``, ``periodic_z=False``,
      ``p_bc_z0=p_bc_zL=0``, ``theta_bc_z0=theta_bc_zL=1``) is
      passed explicitly. The current solver-side
      ``_pack_ghosts`` is idempotent so this is compatible
      without a solver-side change.

    Parameters
    ----------
    state
        ``DieselAusasState`` carrying the previous accepted
        ``P_nd``, ``theta`` and ``H_prev`` on the unpadded physical
        grid.
    H_curr
        Current candidate film thickness on the physical
        ``(N_z, N_phi)`` diesel grid (``H = h / c``).
    dt_s
        Mechanical timestep in **physical seconds**. The adapter
        converts to non-dimensional ``dt_ausas = ω·dt_s`` before
        forwarding to the GPU backend.
    omega_shaft
        Shaft angular velocity (rad/s). Required for the
        non-dimensional time conversion.
    d_phi, d_Z, R, L
        Solver kwargs forwarded to ``ausas_unsteady_one_step_gpu``
        unchanged.
    extra_options
        Optional dict merged into the solver kwargs (e.g.
        ``omega_p``, ``omega_theta``, ``alpha``, ``tol``,
        ``max_inner``, ``check_every``). ``scheme`` is intentionally
        NOT documented here — the one-step GPU solver implements
        only Jacobi sweeps; passing ``scheme`` would either be
        ignored or raise on the GPU side, depending on the
        ``ausas_unsteady_one_step_gpu`` signature in the active
        ``gpu-reynolds`` build. Stage J fu-2 fixup-1 removed the
        corresponding ``--ausas-scheme`` CLI flag.
    commit
        When ``False`` the state is **not** mutated — used for the
        Verlet trial path. When ``True`` the wrapper writes the new
        ``P``, ``theta``, ``H_prev``, ``step_index``, ``time_s``,
        ``dt_last_s`` and ``dt_ausas_last`` back into ``state``.

    Returns
    -------
    :class:`DieselAusasStepResult`
        Structured result; ``P_nd`` / ``theta`` are on the unpadded
        physical grid. The runner is expected to pass ``P_nd``
        (NOT a re-dimensionalised pressure) into
        ``compute_hydro_forces`` etc., which already multiply by
        ``p_scale_step``.
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

    # Stage J fu-2 ghost-grid migration — adapter-side padding.
    # State arrays live on the unpadded physical grid; the adapter
    # pads them to ``(N_z, N_phi+2)`` immediately before the
    # backend call (``padded[:, 0] = phys[:, -1]``,
    # ``padded[:, -1] = phys[:, 0]``) and unpads the backend's
    # padded return arrays before storing in ``state`` /
    # ``DieselAusasStepResult``. ``d_phi`` is the PHYSICAL spacing
    # ``2π / N_phi``, NOT divided by the padded width.
    H_curr_phys = np.asarray(H_curr, dtype=float)
    H_prev_phys = np.asarray(state.H_prev, dtype=float)
    P_prev_phys = np.asarray(state.P, dtype=float)
    theta_prev_phys = np.asarray(state.theta, dtype=float)

    # Stage J Bug 4 — convert physical timestep to Ausas non-dim
    # ``τ`` units before forwarding to the GPU backend.
    dt_ausas = ausas_dt_from_physical(dt_s, omega_shaft)

    kwargs = dict(
        H_curr=pad_phi_for_ausas(H_curr_phys),
        H_prev=pad_phi_for_ausas(H_prev_phys),
        P_prev=pad_phi_for_ausas(P_prev_phys),
        theta_prev=pad_phi_for_ausas(theta_prev_phys),
        dt=float(dt_ausas),
        d_phi=float(d_phi),          # physical spacing 2π/N_phi
        d_Z=float(d_Z),
        R=float(R),
        L=float(L),
        # Boundary convention — explicit per the ghost-grid contract.
        # The current solver-side ``_pack_ghosts`` is an idempotent
        # in-place refresh, so passing already-padded arrays is
        # compatible (it just refreshes the seam ghosts).
        periodic_phi=True,
        periodic_z=False,
        p_bc_z0=0.0,
        p_bc_zL=0.0,
        theta_bc_z0=1.0,
        theta_bc_zL=1.0,
    )
    if extra_options:
        # Stage J fu-2 Task 29 — strip reserved adapter-only keys
        # (``__dump_config__`` / ``__dump_metadata__`` /
        # ``__dump_counters__``) before forwarding to the GPU
        # backend; they're routing metadata for the dump path, not
        # solver kwargs.
        _solver_kwargs = {
            k: v for k, v in extra_options.items()
            if not (isinstance(k, str) and k.startswith("__"))
        }
        kwargs.update(_solver_kwargs)
    # ``max_inner`` is only used to derive the convergence flag when
    # the solver does not return one explicitly. If the caller did
    # not override it, we don't know the solver's internal budget,
    # so we MUST trust the solver: a finite return value with no
    # explicit non-convergence flag means the step is accepted. We
    # never compare ``residual`` against a local guess at ``tol``
    # because the solver's internal stopping criterion can be
    # tighter or looser than any default we'd pick here, and a
    # mismatched local tol would silently drop accepted steps as
    # ``rejected_commit_count`` (the precise regression that broke
    # the first GPU integration smoke).
    user_max_inner = (
        int(extra_options["max_inner"])
        if (extra_options is not None
            and "max_inner" in extra_options) else None)

    try:
        out = backend(**kwargs)
    except Exception as exc:
        # Failure path: do NOT poison state, do NOT throw. The runner
        # treats the step as solver-failed and may choose to retry or
        # mark the step invalid.
        if commit:
            state.failed_step_count += 1
        return DieselAusasStepResult(
            P_nd=None, theta=None,
            residual=float("nan"), n_inner=0,
            converged=False,
            H_curr=H_curr_phys,
            dt_phys_s=float(dt_s), dt_ausas=float(dt_ausas),
            reason=f"ausas_one_step_failed: {type(exc).__name__}: {exc}",
        )

    P_pad, theta_pad, residual, n_inner, ok_explicit = (
        _unpack_ausas_return(out))

    # Stage J fu-2 ghost-grid migration — shape-validate the
    # backend's padded return AND unpad before any consumer sees
    # the arrays. The diesel pipeline (forces, summary, npz, state
    # commit) all expect physical ``(N_z, N_phi)``.
    expected_pad_shape = (
        H_curr_phys.shape[0], H_curr_phys.shape[1] + 2)
    if P_pad.shape != expected_pad_shape:
        raise ValueError(
            f"Ausas backend returned P shape {P_pad.shape}, "
            f"expected padded {expected_pad_shape}")
    if theta_pad.shape != expected_pad_shape:
        raise ValueError(
            f"Ausas backend returned theta shape {theta_pad.shape}, "
            f"expected padded {expected_pad_shape}")
    P_nd = unpad_phi_from_ausas(P_pad)
    theta_phys = unpad_phi_from_ausas(theta_pad)

    # Derive ``ok``:
    # * If the solver explicitly returned a convergence flag, honour
    #   it (5-tuple / dict shapes only).
    # * Otherwise: trust the solver. A finite return with no
    #   explicit failure flag means the step is accepted. The only
    #   information we can use independently is ``user_max_inner``
    #   — when the caller passed ``max_inner`` in ``extra_options``,
    #   we know the budget, and ``n_inner >= max_inner`` then
    #   indicates the solver hit the budget without converging.
    #   When the caller did NOT pass ``max_inner``, the solver's
    #   internal default is unknown and we must NOT second-guess it.
    if ok_explicit is None:
        if user_max_inner is not None and n_inner >= user_max_inner:
            converged = False
        else:
            converged = True
    else:
        converged = bool(ok_explicit)

    finite = (
        np.all(np.isfinite(P_nd))
        and np.all(np.isfinite(theta_phys))
    )
    if not finite:
        if commit:
            state.failed_step_count += 1
        return DieselAusasStepResult(
            P_nd=np.asarray(P_nd), theta=np.asarray(theta_phys),
            residual=float(residual)
                if np.isfinite(residual) else float("nan"),
            n_inner=int(n_inner),
            converged=False,
            H_curr=H_curr_phys,
            dt_phys_s=float(dt_s), dt_ausas=float(dt_ausas),
            reason="ausas_returned_nonfinite",
        )

    if commit:
        if not converged:
            # Solver reported non-convergence — count the rejection
            # but leave the previous accepted state intact.
            state.rejected_commit_count += 1
        else:
            state.P = np.asarray(P_nd)
            state.theta = np.asarray(theta_phys)
            state.H_prev = H_curr_phys
            state.step_index += 1
            state.time_s += float(dt_s)
            state.dt_last_s = float(dt_s)
            state.dt_ausas_last = float(dt_ausas)
            state.valid = True

    # Stage J fu-2 Task 29 — extract dump signals from the dict
    # return shape (empty for legacy 4-tuple returns).
    _dump_signals = _unpack_ausas_dump_signals(out)

    # Stage J fu-2 Task 29 — opportunistic dump of the failed
    # one-step input state. Only fires when the runner explicitly
    # threaded ``__dump_config__`` + ``__dump_metadata__`` through
    # ``extra_options`` AND at least one trigger condition matches.
    # Pure observability; never raises into the runner.
    _maybe_emit_failed_dump(
        extra_options=extra_options,
        H_curr_phys=H_curr_phys, H_prev_phys=H_prev_phys,
        P_prev_phys=P_prev_phys, theta_prev_phys=theta_prev_phys,
        P_nd=P_nd, theta_phys=theta_phys,
        residual=float(residual) if np.isfinite(residual)
            else float("nan"),
        n_inner=int(n_inner), converged=bool(converged),
        commit=commit,
        dt_s=float(dt_s), d_phi=float(d_phi), d_Z=float(d_Z),
        kwargs_snapshot=kwargs,
        dump_signals=_dump_signals,
        user_max_inner=user_max_inner,
    )

    return DieselAusasStepResult(
        P_nd=np.asarray(P_nd), theta=np.asarray(theta_phys),
        residual=float(residual)
            if np.isfinite(residual) else float("nan"),
        n_inner=int(n_inner),
        converged=bool(converged),
        H_curr=H_curr_phys,
        dt_phys_s=float(dt_s), dt_ausas=float(dt_ausas),
        reason=("ok" if converged else "ausas_not_converged"),
        failure_kind=str(_dump_signals.get("failure_kind", "")),
        first_nan_field=str(_dump_signals.get("first_nan_field", "")),
        first_nan_index=tuple(
            _dump_signals.get("first_nan_index", ()) or ()),
        first_nan_is_ghost=bool(
            _dump_signals.get("first_nan_is_ghost", False)),
        first_nan_is_axial_boundary=bool(
            _dump_signals.get("first_nan_is_axial_boundary", False)),
        first_nan_is_phi_seam=bool(
            _dump_signals.get("first_nan_is_phi_seam", False)),
        nan_iter=int(_dump_signals.get("nan_iter", -1)),
        nonfinite_count=int(
            _dump_signals.get("nonfinite_count", 0)),
        residual_linf=float(
            _dump_signals.get("residual_linf", float("nan"))),
        residual_rms=float(
            _dump_signals.get("residual_rms", float("nan"))),
        residual_l2_abs=float(
            _dump_signals.get("residual_l2_abs", float("nan"))),
    )


def _maybe_emit_failed_dump(
    *,
    extra_options: Optional[Dict[str, Any]],
    H_curr_phys: np.ndarray, H_prev_phys: np.ndarray,
    P_prev_phys: np.ndarray, theta_prev_phys: np.ndarray,
    P_nd: np.ndarray, theta_phys: np.ndarray,
    residual: float, n_inner: int, converged: bool,
    commit: bool, dt_s: float, d_phi: float, d_Z: float,
    kwargs_snapshot: Dict[str, Any],
    dump_signals: Dict[str, Any],
    user_max_inner: Optional[int],
) -> None:
    """Stage J fu-2 Task 29 — write a ``.npz`` of the failed
    one-step call when the runner threaded the dump-config /
    metadata through ``extra_options``. Catches its own exceptions
    so the diesel runner never trips on a dump-write failure.
    """
    if not extra_options:
        return
    cfg = extra_options.get("__dump_config__")
    meta = extra_options.get("__dump_metadata__")
    counters = extra_options.get("__dump_counters__")
    if cfg is None or meta is None or counters is None:
        return
    if cfg.directory is None:
        return
    # Trigger evaluation
    F_hyd_x = meta.get("F_hyd_x")
    F_hyd_y = meta.get("F_hyd_y")
    triggers = evaluate_triggers(
        cfg=cfg,
        converged=bool(converged),
        failure_kind=str(dump_signals.get("failure_kind", "")),
        residual=float(residual),
        n_inner=int(n_inner),
        n_inner_max=int(user_max_inner)
            if user_max_inner is not None else int(n_inner) + 1,
        F_hyd_x=F_hyd_x, F_hyd_y=F_hyd_y,
    )
    if not triggers:
        return
    # Limit gate
    if counters.written >= int(cfg.limit):
        counters.suppressed_after_limit += 1
        for t in triggers:
            counters.by_trigger[t] = counters.by_trigger.get(t, 0) + 1
        return
    primary = triggers[0]
    fname = build_dump_filename(
        step=int(meta.get("step", -1)),
        substep=int(meta.get("substep", -1)),
        trial=int(meta.get("trial", -1)),
        commit=bool(commit),
        primary_trigger=primary,
    )
    payload: Dict[str, Any] = {
        # Mandatory solver inputs
        "H_prev": np.asarray(H_prev_phys, dtype=float),
        "H_curr": np.asarray(H_curr_phys, dtype=float),
        "P_prev": np.asarray(P_prev_phys, dtype=float),
        "theta_prev": np.asarray(theta_prev_phys, dtype=float),
        "dt_s": float(dt_s),
        "d_phi": float(d_phi),
        "d_Z": float(d_Z),
        "periodic_phi": bool(kwargs_snapshot.get("periodic_phi", True)),
        "periodic_z": bool(kwargs_snapshot.get("periodic_z", False)),
        "bc_z_low": kwargs_snapshot.get("p_bc_z0"),
        "bc_z_high": kwargs_snapshot.get("p_bc_zL"),
        "bc_phi_low": kwargs_snapshot.get("theta_bc_z0"),
        "bc_phi_high": kwargs_snapshot.get("theta_bc_zL"),
        "solver_kwargs": {
            k: v for k, v in kwargs_snapshot.items()
            if k not in {"H_curr", "H_prev", "P_prev", "theta_prev"}
        },
        # Step metadata
        "step": int(meta.get("step", -1)),
        "substep": int(meta.get("substep", -1)),
        "trial": int(meta.get("trial", -1)),
        "commit": bool(commit),
        "phi_deg": float(meta.get("phi_deg", float("nan"))),
        "eps_x": float(meta.get("eps_x", float("nan"))),
        "eps_y": float(meta.get("eps_y", float("nan"))),
        "config_label": str(meta.get("config_label", "")),
        "trial_kind": str(meta.get("trial_kind", "")),
        "texture_kind": str(meta.get("texture_kind", "")),
        "groove_preset": str(meta.get("groove_preset", "")),
        "cavitation": str(meta.get("cavitation", "")),
        "n_phi": int(H_curr_phys.shape[1]),
        "n_z": int(H_curr_phys.shape[0]),
        # Solver result + nonfinite diagnostics
        "converged": bool(converged),
        "failure_kind": str(dump_signals.get("failure_kind", "")),
        "first_nan_field": str(
            dump_signals.get("first_nan_field", "")),
        "first_nan_index": tuple(
            dump_signals.get("first_nan_index", ()) or ()),
        "first_nan_is_ghost": bool(
            dump_signals.get("first_nan_is_ghost", False)),
        "first_nan_is_axial_boundary": bool(
            dump_signals.get(
                "first_nan_is_axial_boundary", False)),
        "first_nan_is_phi_seam": bool(
            dump_signals.get("first_nan_is_phi_seam", False)),
        "nan_iter": int(dump_signals.get("nan_iter", -1)),
        "residual": float(residual),
        "residual_linf": float(
            dump_signals.get("residual_linf", residual)),
        "residual_rms": float(
            dump_signals.get("residual_rms", float("nan"))),
        "residual_l2_abs": float(
            dump_signals.get("residual_l2_abs", float("nan"))),
        "n_inner": int(n_inner),
        "nonfinite_count": int(
            dump_signals.get("nonfinite_count", 0)),
        # Shape diagnostics — adapter unpadded the backend's return,
        # so the in-RAM shapes are the physical (N_z, N_phi).
        "P_shape_raw": tuple(P_nd.shape),
        "theta_shape_raw": tuple(theta_phys.shape),
        "expected_physical_shape": tuple(H_curr_phys.shape),
        "expected_padded_shape": (
            H_curr_phys.shape[0], H_curr_phys.shape[1] + 2),
        "P_is_padded": False,
        "theta_is_padded": False,
        # Commit-semantics metadata (Task 32 sync) — runner threads
        # these through meta when known. Defaults preserve schema.
        "final_trial_status": str(meta.get(
            "final_trial_status", "no_attempt")),
        "committed_state_status": str(meta.get(
            "committed_state_status", "rejected_no_commit")),
        "accepted_state_source": str(meta.get(
            "accepted_state_source", "none")),
        "committed_state_is_finite": bool(meta.get(
            "committed_state_is_finite", False)),
    }
    if cfg.include_force_inputs:
        # Optional force-integration inputs. Runner populates these
        # via meta when available; absent values stay as None / NaN.
        for k in OPTIONAL_FORCE_INPUT_KEYS:
            if k in meta:
                payload[k] = meta[k]
    try:
        target = write_dump_npz(cfg.directory, fname, payload)
        counters.written += 1
        for t in triggers:
            counters.by_trigger[t] = counters.by_trigger.get(t, 0) + 1
        # One-line stdout marker for the operator.
        print(
            f"[AUSAS-DUMP] step={meta.get('step', -1)} "
            f"trial={meta.get('trial', -1)} "
            f"trigger={primary} file={target}",
            flush=True)
    except Exception as exc:
        counters.write_failed += 1
        # Non-fatal — surface a one-line warning so the run
        # log carries the failure trail.
        print(
            f"[AUSAS-DUMP-FAILED] step={meta.get('step', -1)} "
            f"trial={meta.get('trial', -1)} "
            f"reason={type(exc).__name__}: {exc}",
            flush=True)


def _unpack_ausas_dump_signals(
    out: Any,
) -> Dict[str, Any]:
    """Stage J fu-2 Task 29 — pull the gpu-reynolds Task 12 dump
    signals out of a dict return shape. Returns an empty dict for
    non-dict shapes (legacy 4-tuple) so callers can ``.get(..., default)``
    without checking the input type.
    """
    if not isinstance(out, dict):
        return {}
    out_d: Dict[str, Any] = {}
    for k in (
        "failure_kind",
        "first_nan_field",
        "first_nan_index",
        "first_nan_is_ghost",
        "first_nan_is_axial_boundary",
        "first_nan_is_phi_seam",
        "nan_iter",
        "nonfinite_count",
        "residual_linf",
        "residual_rms",
        "residual_l2_abs",
    ):
        if k in out:
            out_d[k] = out[k]
    return out_d


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
        ``dict(P=..., theta=..., residual_linf=..., residual_rms=...,
               residual_l2_abs=..., residual=..., n_inner=...,
               converged=...)``
            **canonical post-Task-12 dict shape** from the
            gpu-reynolds Stage J fu-2 sync. Required keys:
            ``P`` and ``theta``. Optional residual variants in
            preference order:

            * ``residual_linf``  — L∞ (max) norm. **Canonical** —
              what the kernel's solver-validity gate compares
              against ``ausas_tol`` (Task 12 contract); pre-Task-12
              callers used a max-norm residual under the legacy
              ``residual`` key, so the meaning is preserved.
            * ``residual_rms``    — diagnostic RMS norm
              (currently unused downstream).
            * ``residual_l2_abs`` — diagnostic absolute L2 norm
              (currently unused downstream).
            * ``residual``       — legacy alias kept for
              backward compatibility with shipping code that
              hasn't migrated yet.

            The canonical scalar returned in position 2 of the
            5-tuple is, in order: ``residual_linf`` if present,
            else ``residual``, else NaN. ``residual_rms`` and
            ``residual_l2_abs`` are read but currently discarded
            (the kernel only consumes one residual scalar — see
            ``models.diesel_coupling.guards.check_solver_validity``);
            adding them to ``DieselAusasStepResult`` would let the
            npz schema surface them, but that's a follow-up that
            depends on the kernel actually using either.
    """
    if isinstance(out, dict):
        P = np.asarray(out["P"])
        theta = np.asarray(out["theta"])
        # Canonical post-Task-12: ``residual_linf`` is the max-norm
        # residual that the kernel's solver-validity gate compares
        # against ``ausas_tol``. Fall back to legacy ``residual``
        # (same semantics in shipping code) if the new key isn't
        # present, then NaN as last resort.
        if "residual_linf" in out:
            residual = float(out["residual_linf"])
        elif "residual" in out:
            residual = float(out["residual"])
        else:
            residual = float("nan")
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
