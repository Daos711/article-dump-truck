"""Stage J followup-2 Step 11 (Gate 4) — policy-selection contract.

Pins capability-based dispatch in
:func:`models.diesel_coupling.select_policy`. Four backends covered:

1. :class:`HalfSommerfeldBackend` (stateless + explicit-friendly)
   → ``POLICY_LEGACY_HS``
2. :class:`AusasDynamicBackend` (stateful + explicit-friendly)
   → ``POLICY_AUSAS_DYNAMIC``
3. inline ``_StubBackend`` for a future PV-on-HS backend
   (stateless + explicit) → ``POLICY_LEGACY_HS``
4. inline ``_StubBackend`` for a future PV-on-Ausas backend
   (stateful + implicit-required) → ``POLICY_AUSAS_DYNAMIC``

The dispatch contract is OR-of-flags::

    if backend.stateful or backend.requires_implicit_mech_coupling:
        return POLICY_AUSAS_DYNAMIC
    return POLICY_LEGACY_HS

so the future Stage K (piezoviscous coupling) plugs in via the same
two-flag protocol — no runner branch on backend name, no new
policy. Stage K's first candidate is expected to be
``pv_half_sommerfeld`` (stateless + explicit) which routes to the
LEGACY policy by capability, exactly like the existing HS path.

Field-based shape assertion (vs ``is`` identity) so the contract
remains valid if a future runner-side wrapper layers field
overrides on top of the base policies.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from models.diesel_coupling import (
    AusasDynamicBackend,
    CouplingPolicy,
    HalfSommerfeldBackend,
    POLICY_AUSAS_DYNAMIC,
    POLICY_LEGACY_HS,
    select_policy,
)


@dataclass(frozen=True)
class _StubBackend:
    """Minimal placeholder for a backend that has not been
    implemented yet (Stage K piezoviscous). Carries only the two
    capability flags ``select_policy`` reads — anything else
    (``solve_trial``, ``PressureSolveResult`` fields) is the real
    backend's responsibility when it lands."""
    name: str
    stateful: bool
    requires_implicit_mech_coupling: bool


def _assert_same_policy_shape(
        actual: CouplingPolicy, expected: CouplingPolicy) -> None:
    """Pin the eight policy fields the kernel and runner read on
    every step. Resilient to future wrapper layers that might
    apply CLI overrides via ``dataclasses.replace`` (in which case
    the result is no longer ``is`` the base, but its shape on
    these fields must still match)."""
    assert actual.name == expected.name, (
        f"name: actual={actual.name!r} expected={expected.name!r}")
    assert actual.max_mech_inner == expected.max_mech_inner
    assert actual.mech_relax_initial == expected.mech_relax_initial
    assert actual.mech_relax_min == expected.mech_relax_min
    assert actual.enable_line_search == expected.enable_line_search
    assert (actual.physical_guards_mode
            == expected.physical_guards_mode)
    assert actual.commit_on_clamp == expected.commit_on_clamp
    assert (actual.require_solver_converged
            == expected.require_solver_converged)


# ─── 1. Real backends — the two paths in production today ─────────


def test_select_policy_half_sommerfeld_picks_legacy_verlet():
    """``HalfSommerfeldBackend`` (stateless + explicit-friendly)
    must dispatch to ``POLICY_LEGACY_HS``. This pins the Gate 1
    invariance contract on the policy side: any change that
    silently re-routes HS to a different policy would break the
    bit-for-bit legacy path."""
    backend = HalfSommerfeldBackend(retry_config=None,
                                     textured_for_retry=False)
    policy = select_policy(backend)
    _assert_same_policy_shape(policy, POLICY_LEGACY_HS)


def test_select_policy_ausas_dynamic_picks_damped_implicit_film():
    """``AusasDynamicBackend`` (stateful) must dispatch to
    ``POLICY_AUSAS_DYNAMIC``. This is the Gate 2 / Stage J
    production path."""
    backend = AusasDynamicBackend(ausas_options=None)
    policy = select_policy(backend)
    _assert_same_policy_shape(policy, POLICY_AUSAS_DYNAMIC)


# ─── 2. Stub-PV backends — Stage K future paths ─────────────────


def test_select_policy_stub_pv_on_half_sommerfeld_picks_legacy():
    """A stateless PV-on-HS backend (no Ausas state, no implicit
    coupling required) must route to the LEGACY policy via the
    same capability dispatch — no new branch needed when Stage K
    lands. Expected first PV implementation per the architectural
    note."""
    pv_on_hs = _StubBackend(
        name="pv_half_sommerfeld",
        stateful=False,
        requires_implicit_mech_coupling=False,
    )
    policy = select_policy(pv_on_hs)
    _assert_same_policy_shape(policy, POLICY_LEGACY_HS)


def test_select_policy_stub_pv_on_ausas_picks_damped():
    """A stateful PV-on-Ausas backend (carries Ausas state AND
    requires implicit coupling because the squeeze-driven film
    pressure is now also viscosity-dependent) must route to the
    DAMPED policy. Both flags are True here so this exercises the
    OR-side of the dispatch contract."""
    pv_on_ausas = _StubBackend(
        name="pv_ausas_dynamic",
        stateful=True,
        requires_implicit_mech_coupling=True,
    )
    policy = select_policy(pv_on_ausas)
    _assert_same_policy_shape(policy, POLICY_AUSAS_DYNAMIC)
