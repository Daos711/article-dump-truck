"""Stage J followup-2 — coupling policies.

The kernel selects a ``CouplingPolicy`` from the backend's
capabilities (``backend.stateful``, ``backend.requires_implicit_mech_coupling``).
Policy selection is by **capability**, not by name. This is what
makes future PV (Stage K) trivial: a stateless PV-on-half-Sommerfeld
backend selects ``POLICY_LEGACY_HS`` automatically; a stateful
PV-on-Ausas backend selects ``POLICY_AUSAS_DYNAMIC``. No new
runner branches needed.

Two production policies:

* ``POLICY_LEGACY_HS`` — bit-for-bit equivalent to the existing
  ``for k in range(N_SUB=3)`` corrector body inside
  ``run_transient``. Guards in *diagnostic* mode (warn-only).
  No line search, no relax shrinking. Locked in by Gate 1.
* ``POLICY_AUSAS_DYNAMIC`` — damped implicit film coupling per
  doc 1 §4.3. Up to 8 mechanical inner iterations; relax starts
  at 0.25 and halves on rejection down to 0.03125; line-search on
  candidate-blend; physical guards in *hard* mode; no commit on
  clamped step.

CLI overrides are surfaced via ``resolve_policy_overrides`` for
diagnostic / regression runs (e.g. forcing the legacy policy on
the Ausas backend, which is expected to fail acceptance — that's
the point of the override).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional


# Locked from current ``run_transient`` to keep Gate 1 bit-for-bit.
N_SUB_LEGACY_DEFAULT: int = 3


@dataclass(frozen=True)
class CouplingPolicy:
    """Frozen so a backend cannot mutate the runner-owned policy."""
    name: str
    max_mech_inner: int
    mech_relax_initial: float
    mech_relax_min: float
    enable_line_search: bool
    enable_physical_guards: bool
    physical_guards_mode: Literal["off", "diagnostic", "hard"]
    commit_on_clamp: bool
    require_solver_converged: bool
    max_delta_eps_inner: Optional[float]
    max_delta_eps_step: Optional[float]
    # Fixed-point convergence tolerance for the damped policy
    # (|Δε| between successive trials, non-dimensional ε units).
    # The legacy policy iterates a fixed N_SUB times and never
    # terminates early, so the value is irrelevant there.
    eps_update_tol: float = 1.0


POLICY_LEGACY_HS: CouplingPolicy = CouplingPolicy(
    name="legacy_verlet",
    max_mech_inner=N_SUB_LEGACY_DEFAULT,
    mech_relax_initial=1.0,
    mech_relax_min=1.0,
    enable_line_search=False,
    enable_physical_guards=True,
    physical_guards_mode="diagnostic",
    commit_on_clamp=False,
    require_solver_converged=True,
    max_delta_eps_inner=None,
    max_delta_eps_step=None,
    eps_update_tol=1.0,    # not used by legacy policy
)


POLICY_AUSAS_DYNAMIC: CouplingPolicy = CouplingPolicy(
    name="damped_implicit_film",
    # Stage J fu-2 Step 9 fixup-2 — expert patch raised the budget
    # from 8 to 24. With Picard contractivity shrink, relax can fall
    # to 0.03125 (8× smaller step); the residual contraction at this
    # relax may need 15-20 iterations to bring delta_eps below the
    # 1e-4 tolerance. 24 leaves headroom for stiff transient peaks.
    max_mech_inner=24,
    mech_relax_initial=0.25,
    mech_relax_min=0.03125,
    enable_line_search=True,
    enable_physical_guards=True,
    physical_guards_mode="hard",
    commit_on_clamp=False,
    require_solver_converged=True,
    max_delta_eps_inner=0.10,
    max_delta_eps_step=0.25,
    eps_update_tol=1.0e-4,
)


def select_policy(backend) -> CouplingPolicy:
    """Capability-based policy selection.

    A backend that is stateful OR requires implicit mechanical
    coupling gets the damped policy. Anything else (stateless,
    explicit-friendly) gets the legacy policy. Future PV backends
    plug in via the same protocol — no new branch here.
    """
    if backend.stateful or backend.requires_implicit_mech_coupling:
        return POLICY_AUSAS_DYNAMIC
    return POLICY_LEGACY_HS


def resolve_policy_overrides(
    base: CouplingPolicy,
    *,
    explicit_name: Optional[str] = None,
    max_mech_inner: Optional[int] = None,
    mech_relax_initial: Optional[float] = None,
    mech_relax_min: Optional[float] = None,
    physical_guards_mode: Optional[
        Literal["off", "diagnostic", "hard"]] = None,
    max_delta_eps_inner: Optional[float] = None,
    max_delta_eps_step: Optional[float] = None,
) -> CouplingPolicy:
    """Apply CLI overrides for diagnostic / regression runs.

    All ``None`` arguments leave the ``base`` policy field untouched.
    ``explicit_name`` is a documentation marker only — it does NOT
    silently swap the entire policy; callers must pass the explicit
    constant they want and then layer the overrides on top.
    """
    overrides: dict = {}
    if max_mech_inner is not None:
        overrides["max_mech_inner"] = int(max_mech_inner)
    if mech_relax_initial is not None:
        overrides["mech_relax_initial"] = float(mech_relax_initial)
    if mech_relax_min is not None:
        overrides["mech_relax_min"] = float(mech_relax_min)
    if physical_guards_mode is not None:
        overrides["physical_guards_mode"] = physical_guards_mode
    if max_delta_eps_inner is not None:
        overrides["max_delta_eps_inner"] = float(max_delta_eps_inner)
    if max_delta_eps_step is not None:
        overrides["max_delta_eps_step"] = float(max_delta_eps_step)
    if not overrides:
        return base
    return replace(base, **overrides)


def resolve_policy(
    backend,
    *,
    coupling_override: Literal[
        "auto", "legacy_verlet", "damped_implicit_film"] = "auto",
    max_mech_inner: Optional[int] = None,
    mech_relax_initial: Optional[float] = None,
    mech_relax_min: Optional[float] = None,
    physical_guards_mode: Optional[
        Literal["off", "diagnostic", "hard"]] = None,
    max_delta_eps_inner: Optional[float] = None,
    max_delta_eps_step: Optional[float] = None,
) -> CouplingPolicy:
    """One-shot policy resolution for the runner.

    1. Pick the BASE policy:
       * ``coupling_override == "auto"`` → ``select_policy(backend)``
         (capability-based default — recommended).
       * ``coupling_override == "legacy_verlet"`` → POLICY_LEGACY_HS
         regardless of backend (diagnostic / regression).
       * ``coupling_override == "damped_implicit_film"`` →
         POLICY_AUSAS_DYNAMIC regardless of backend (forces the damped
         policy on a stateless backend — expected to mis-converge,
         that's the whole point of the override).
    2. Layer non-None CLI overrides on top via
       :func:`resolve_policy_overrides`.

    All overrides are documented in ``scripts/run_diesel_thd_transient.py``
    under the ``--max-mech-inner`` / ``--mech-relax-initial`` /
    ``--mech-relax-min`` / ``--guards-profile`` flags.
    """
    if coupling_override == "auto":
        base = select_policy(backend)
    elif coupling_override == "legacy_verlet":
        base = POLICY_LEGACY_HS
    elif coupling_override == "damped_implicit_film":
        base = POLICY_AUSAS_DYNAMIC
    else:
        raise ValueError(
            f"resolve_policy: unknown coupling_override "
            f"{coupling_override!r}; expected 'auto', "
            f"'legacy_verlet', or 'damped_implicit_film'.")
    return resolve_policy_overrides(
        base,
        max_mech_inner=max_mech_inner,
        mech_relax_initial=mech_relax_initial,
        mech_relax_min=mech_relax_min,
        physical_guards_mode=physical_guards_mode,
        max_delta_eps_inner=max_delta_eps_inner,
        max_delta_eps_step=max_delta_eps_step,
    )
