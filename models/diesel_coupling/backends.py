"""Stage J followup-2 — pressure-backend abstraction.

A ``PressureBackend`` is the runner's view of "compute hydrodynamic
pressure for a candidate film thickness". The kernel never branches
on backend name; it queries ``backend.stateful`` and
``backend.requires_implicit_mech_coupling`` to choose the
``CouplingPolicy``. Two production implementations live here:

* ``HalfSommerfeldBackend`` — wraps the existing
  ``solve_reynolds(..., cavitation="half_sommerfeld", ...)``. Stateless;
  the ``commit`` and ``state`` arguments are ignored.
* ``AusasDynamicBackend`` — wraps
  ``models.diesel_ausas_adapter.ausas_one_step_with_state``. Stateful;
  ``commit=False`` for trial Verlet candidates, ``commit=True`` only
  for the accepted mechanical step (after physical / mechanical
  guards pass).

Step 2 (this file) is signature-only. Step 4 wires
``HalfSommerfeldBackend.solve_trial`` to the existing legacy code
path; Step 5 wires ``AusasDynamicBackend.solve_trial`` to the
adapter. No solver-side imports happen at module load — the
existing import-time pattern is preserved (see the runner's
top-of-file ``from reynolds_solver import solve_reynolds``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np


# ─── Per-step context ──────────────────────────────────────────────


@dataclass(frozen=True)
class StepContext:
    """Read-only per-step context handed to backends and the kernel.

    Frozen so a backend cannot accidentally mutate the runner-owned
    geometry / textural / thermal state. Populated once per
    mechanical step by ``run_transient`` immediately before the
    ``advance_mechanical_step`` call.

    Field groups:
        Crank / load
            ``phi_deg``, ``F_ext_x``, ``F_ext_y``, ``F_max``
        Pressure scaling / kinematics
            ``p_scale``, ``omega``, ``eta``
        Geometry
            ``R``, ``L``, ``c``,
            ``Phi_mesh``, ``Z_mesh``, ``phi_1D``, ``Z_1D``,
            ``d_phi``, ``d_Z``
        Texture (per-config)
            ``cfg`` — the ``CONFIGS[ic]`` entry,
            ``texture_kind``, ``groove_relief``,
            ``phi_c_flat``, ``Z_c_flat`` (legacy dimple centres)
        Half-Sommerfeld closure (legacy backend only)
            ``closure``, ``cavitation``, ``P_warm_init``
    """
    phi_deg: float
    F_ext_x: float
    F_ext_y: float
    F_max: float
    p_scale: float
    omega: float
    eta: float
    R: float
    L: float
    c: float
    Phi_mesh: np.ndarray
    Z_mesh: np.ndarray
    phi_1D: np.ndarray
    Z_1D: np.ndarray
    d_phi: float
    d_Z: float
    cfg: dict
    texture_kind: str
    groove_relief: Optional[np.ndarray]
    phi_c_flat: Optional[np.ndarray]
    Z_c_flat: Optional[np.ndarray]
    closure: str
    cavitation: str
    P_warm_init: Optional[np.ndarray]


# ─── Backend-uniform pressure result ───────────────────────────────


@dataclass
class PressureSolveResult:
    """Backend-uniform outcome of one pressure solve.

    Mutable (not frozen) so the kernel can post-annotate it with a
    rejection reason without copying. ``theta`` is ``None`` for the
    half-Sommerfeld backend (no fluid fraction in that closure);
    guards skip theta checks accordingly.
    """
    P_nd: Optional[np.ndarray]
    theta: Optional[np.ndarray]
    Fx_hyd: float
    Fy_hyd: float
    H_used: np.ndarray
    residual: float
    n_inner: int
    converged: bool
    backend_name: str
    reason: str


# ─── Protocol ──────────────────────────────────────────────────────


class PressureBackend(Protocol):
    """Capability-typed pressure backend.

    The kernel uses ``stateful`` and ``requires_implicit_mech_coupling``
    to pick the coupling policy:

    +--------------------------+---------+-----------------------+
    | Backend                  | stateful| requires_implicit_... |
    +==========================+=========+=======================+
    | ``half_sommerfeld``      | False   | False                 |
    +--------------------------+---------+-----------------------+
    | ``ausas_dynamic``        | True    | True                  |
    +--------------------------+---------+-----------------------+
    | (Stage K) ``hs_pv``      | False   | False                 |
    +--------------------------+---------+-----------------------+
    | (Stage K) ``ausas_pv``   | True    | True                  |
    +--------------------------+---------+-----------------------+
    """
    name: str
    stateful: bool
    requires_implicit_mech_coupling: bool

    def solve_trial(
        self,
        H_curr: np.ndarray,
        H_prev: Optional[np.ndarray],
        dt_phys: float,
        omega: float,
        state: Optional[Any],
        commit: bool,
        context: StepContext,
        extra_options: Optional[Dict[str, Any]] = None,
        *,
        p_warm_init: Optional[np.ndarray] = None,
        vx_squeeze: float = 0.0,
        vy_squeeze: float = 0.0,
    ) -> PressureSolveResult:
        """Solve the pressure problem for a candidate film.

        Per-trial kwargs:

        ``p_warm_init``
            Legacy SOR warm-start hint; consumed by
            ``HalfSommerfeldBackend``. Ignored by stateful backends
            (Ausas carries its own warm via ``state``).
        ``vx_squeeze`` / ``vy_squeeze``
            Current Verlet corrector velocities that drive the
            squeeze term ``squeeze_to_api_params(-vx, -vy, c, omega,
            d_phi)``. Consumed by ``HalfSommerfeldBackend``;
            ignored by Ausas (squeeze comes from H_prev → H_curr).

        The kernel updates these between trials so the within-step
        warm / squeeze chain matches the legacy runner bit-for-bit.
        """
        ...


# ─── Concrete backends (skeletons) ─────────────────────────────────


class HalfSommerfeldBackend:
    """Legacy half-Sommerfeld closure via ``solve_reynolds``.

    Stateless: ``commit`` and ``state`` arguments are ignored.
    The SOR warm-start ``p_warm_init`` is consumed per-call
    (the kernel threads the up-to-date warm through the within-
    step trial chain so behaviour matches the legacy runner
    bit-for-bit).
    """
    name: str = "half_sommerfeld"
    stateful: bool = False
    requires_implicit_mech_coupling: bool = False

    def __init__(
        self,
        retry_config: Any,
        textured_for_retry: bool,
    ) -> None:
        self._retry_config = retry_config
        self._textured_for_retry = bool(textured_for_retry)

    def solve_trial(
        self,
        H_curr: np.ndarray,
        H_prev: Optional[np.ndarray],
        dt_phys: float,
        omega: float,
        state: Optional[Any],
        commit: bool,
        context: StepContext,
        extra_options: Optional[Dict[str, Any]] = None,
        *,
        p_warm_init: Optional[np.ndarray] = None,
        vx_squeeze: float = 0.0,
        vy_squeeze: float = 0.0,
    ) -> PressureSolveResult:
        # Stage J fu-2 Step 4 — extracted from the legacy
        # ``for k in range(N_SUB):`` body. Behaviour is bit-for-bit
        # identical to the pre-refactor runner (Gate 1 contract).
        # Local imports avoid a top-level cycle through the runner.
        # ``squeeze_to_api_params`` lives in ``reynolds_solver`` but
        # the diesel runner already resolves it via a try-chain
        # (``reynolds_solver.dynamic.squeeze`` /
        # ``reynolds_solver.squeeze`` / ``reynolds_solver``); we
        # reuse that resolution by importing through the runner's
        # namespace.
        from models.diesel_transient import (
            _solve_dynamic_with_retry,
            squeeze_to_api_params,
        )

        xp_, yp_, bt_ = squeeze_to_api_params(
            -float(vx_squeeze),
            -float(vy_squeeze),
            context.c, context.omega, context.d_phi,
        )
        base_kw = dict(
            closure=context.closure,
            cavitation=context.cavitation,
            tol=1e-5,
            max_iter=50000,
            P_init=p_warm_init,
            xprime=xp_, yprime=yp_, beta=bt_,
        )
        P_, Fx_, Fy_, n_outer, ok_, reason_ = _solve_dynamic_with_retry(
            H_curr, context.d_phi, context.d_Z,
            context.R, context.L,
            base_kw=base_kw,
            p_scale=context.p_scale,
            Phi_mesh=context.Phi_mesh,
            phi_1D=context.phi_1D, Z_1D=context.Z_1D,
            retry_config=self._retry_config,
            textured=self._textured_for_retry,
        )
        Fx_f = float(Fx_) if np.isfinite(Fx_) else float("nan")
        Fy_f = float(Fy_) if np.isfinite(Fy_) else float("nan")
        return PressureSolveResult(
            P_nd=P_,
            theta=None,
            Fx_hyd=Fx_f,
            Fy_hyd=Fy_f,
            H_used=H_curr,
            residual=float("nan"),     # legacy SOR doesn't surface
            n_inner=int(n_outer),
            converged=bool(ok_),
            backend_name=self.name,
            reason=str(reason_),
        )


class AusasDynamicBackend:
    """Dynamic Ausas JFO via ``ausas_one_step_with_state``.

    Step 2 — skeleton only. Step 5 wires ``solve_trial`` to the
    adapter. ``state`` must be a ``DieselAusasState`` carrying the
    previous-accepted ``(P_nd, theta, H_prev)``; ``commit=True``
    only for the accepted mechanical step after all guards pass.
    """
    name: str = "ausas_dynamic"
    stateful: bool = True
    requires_implicit_mech_coupling: bool = True

    def __init__(
        self,
        ausas_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def solve_trial(
        self,
        H_curr: np.ndarray,
        H_prev: Optional[np.ndarray],
        dt_phys: float,
        omega: float,
        state: Optional[Any],
        commit: bool,
        context: StepContext,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> PressureSolveResult:
        ...
