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
)


POLICY_AUSAS_DYNAMIC: CouplingPolicy = CouplingPolicy(
    name="damped_implicit_film",
    max_mech_inner=8,
    mech_relax_initial=0.25,
    mech_relax_min=0.03125,
    enable_line_search=True,
    enable_physical_guards=True,
    physical_guards_mode="hard",
    commit_on_clamp=False,
    require_solver_converged=True,
    max_delta_eps_inner=0.10,
    max_delta_eps_step=0.25,
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

    Step 2 — skeleton only. Step 10 wires this to the CLI.
    """
    ...
