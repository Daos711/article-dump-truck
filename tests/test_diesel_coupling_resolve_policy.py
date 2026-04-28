"""Stage J followup-2 Step 10 — policy resolution + override tests.

Pins ``resolve_policy()`` and ``resolve_policy_overrides()``
behavior used by the runner to wire CLI flags
(``--coupling-policy`` / ``--max-mech-inner`` / etc) onto the
backend-derived default policy.
"""
from __future__ import annotations

import pytest

from models.diesel_coupling import (
    POLICY_AUSAS_DYNAMIC,
    POLICY_LEGACY_HS,
    resolve_policy,
    resolve_policy_overrides,
    select_policy,
)


class _StatelessBackend:
    """Stub for half-Sommerfeld-like backends."""
    name = "half_sommerfeld"
    stateful = False
    requires_implicit_mech_coupling = False


class _StatefulBackend:
    """Stub for Ausas-like backends."""
    name = "ausas_dynamic"
    stateful = True
    requires_implicit_mech_coupling = False


class _ImplicitOnlyBackend:
    """Stub for a hypothetical PV-on-HS backend that's stateless
    but demands implicit coupling. Should still get the damped
    policy — capability dispatch is on EITHER flag."""
    name = "pv_on_hs"
    stateful = False
    requires_implicit_mech_coupling = True


# ─── 1. resolve_policy(coupling_override="auto") routes by capability ─


def test_resolve_policy_auto_stateless_picks_legacy():
    p = resolve_policy(_StatelessBackend(), coupling_override="auto")
    assert p.name == "legacy_verlet"
    assert p.max_mech_inner == POLICY_LEGACY_HS.max_mech_inner


def test_resolve_policy_auto_stateful_picks_damped():
    p = resolve_policy(_StatefulBackend(), coupling_override="auto")
    assert p.name == "damped_implicit_film"
    assert p.max_mech_inner == POLICY_AUSAS_DYNAMIC.max_mech_inner


def test_resolve_policy_auto_implicit_only_picks_damped():
    """A stateless-but-implicit-required backend (future PV-on-HS)
    must STILL pick the damped policy — capability flag is OR'd
    in ``select_policy``."""
    p = resolve_policy(_ImplicitOnlyBackend(),
                        coupling_override="auto")
    assert p.name == "damped_implicit_film"


# ─── 2. Manual override forces the policy regardless of backend ──


def test_resolve_policy_manual_legacy_on_stateful():
    """Forcing legacy_verlet on a stateful backend is a diagnostic
    use case (expected to mis-converge); resolver MUST honor it."""
    p = resolve_policy(_StatefulBackend(),
                        coupling_override="legacy_verlet")
    assert p.name == "legacy_verlet"


def test_resolve_policy_manual_damped_on_stateless():
    """Forcing damped_implicit_film on a stateless backend (HS) is
    also diagnostic; resolver MUST honor it."""
    p = resolve_policy(_StatelessBackend(),
                        coupling_override="damped_implicit_film")
    assert p.name == "damped_implicit_film"


def test_resolve_policy_unknown_override_raises():
    with pytest.raises(ValueError, match="unknown coupling_override"):
        resolve_policy(_StatefulBackend(),
                        coupling_override="quantum_voodoo")


# ─── 3. Field overrides layer on top ─────────────────────────────


def test_resolve_policy_max_mech_inner_override():
    p = resolve_policy(
        _StatefulBackend(),
        coupling_override="auto",
        max_mech_inner=42,
    )
    assert p.name == "damped_implicit_film"
    assert p.max_mech_inner == 42
    # Other fields preserved.
    assert p.mech_relax_initial == POLICY_AUSAS_DYNAMIC.mech_relax_initial
    assert p.mech_relax_min == POLICY_AUSAS_DYNAMIC.mech_relax_min


def test_resolve_policy_relax_floor_overrides():
    p = resolve_policy(
        _StatefulBackend(),
        coupling_override="auto",
        mech_relax_initial=0.5,
        mech_relax_min=0.125,
    )
    assert p.mech_relax_initial == 0.5
    assert p.mech_relax_min == 0.125
    # Untouched fields preserve base policy values.
    assert p.max_mech_inner == POLICY_AUSAS_DYNAMIC.max_mech_inner
    assert p.eps_update_tol == POLICY_AUSAS_DYNAMIC.eps_update_tol


def test_resolve_policy_no_overrides_returns_base_unchanged():
    """Identity case — when every override is None, the base
    policy is returned without an unnecessary ``replace`` call."""
    p_legacy = resolve_policy(_StatelessBackend())
    assert p_legacy is POLICY_LEGACY_HS
    p_damped = resolve_policy(_StatefulBackend())
    assert p_damped is POLICY_AUSAS_DYNAMIC


def test_resolve_policy_overrides_helper_isolated():
    """``resolve_policy_overrides`` MUST work on any base; it
    doesn't care how the base was selected."""
    base = POLICY_AUSAS_DYNAMIC
    out = resolve_policy_overrides(
        base,
        max_mech_inner=99,
        mech_relax_initial=0.3,
        physical_guards_mode="diagnostic",
    )
    assert out.max_mech_inner == 99
    assert out.mech_relax_initial == 0.3
    assert out.physical_guards_mode == "diagnostic"
    # Base is frozen — verify replace() cloned without mutating.
    assert base.max_mech_inner == POLICY_AUSAS_DYNAMIC.max_mech_inner
    assert base.physical_guards_mode == "hard"
