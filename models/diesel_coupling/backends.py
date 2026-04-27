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
    ) -> PressureSolveResult:
        ...


# ─── Concrete backends (skeletons) ─────────────────────────────────


class HalfSommerfeldBackend:
    """Legacy half-Sommerfeld closure via ``solve_reynolds``.

    Step 2 — skeleton only. Step 4 wires ``solve_trial`` to the
    existing ``_solve_dynamic_with_retry`` call from
    ``models.diesel_transient``. ``commit`` and ``state`` are
    ignored (stateless backend).
    """
    name: str = "half_sommerfeld"
    stateful: bool = False
    requires_implicit_mech_coupling: bool = False

    def __init__(
        self,
        retry_config: Any,
        textured_for_retry: bool,
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
