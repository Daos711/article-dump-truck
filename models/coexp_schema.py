"""Schema + dataclasses for co-design (coexp_v1).

Единственный источник constants для всех coexp-скриптов. Schema version
пишется в каждый manifest, plot-пайплайн падает при mismatch.

Reference texture fixed per ТЗ §1.2: zone 0-90°, hp=30 μm, a=1.5 mm,
b=1.2 mm, sf=1.5, profile=smoothcap.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .bore_profiles import canonical_spec, profile_hash


SCHEMA_VERSION = "coexp_v1"

# ── Fixed first-pass texture (ТЗ §1.2) ─────────────────────────────

REFERENCE_TEXTURE: Dict[str, Any] = {
    "zone_deg": [0.0, 90.0],
    "hp_um": 30.0,
    "a_mm": 1.5,
    "b_mm": 1.2,
    "sf": 1.5,
    "profile": "smoothcap",
}

# screening eccentricities (ТЗ §6 Phase S1)
SCREENING_EPS: List[float] = [0.30, 0.40, 0.50]

# load share for equilibrium phase (ТЗ §6 Phase E1)
DEFAULT_WY_SHARE: float = 0.25

# acceptance tolerance for equilibrium NR
DEFAULT_TOL_ACCEPT: float = 5e-3
DEFAULT_STEP_CAP: float = 0.10
DEFAULT_EPS_MAX: float = 0.90

# Fixed initial NR seed for equilibrium phase (clarification 4).
# Запрет «переиспользовать» predecessor equilibrium — это делает
# результат зависимым от порядка обхода top-6.
EQUILIBRIUM_NR_SEED_X0: float = 0.0
EQUILIBRIUM_NR_SEED_Y0: float = -0.4


# ── Geometry & sanity thresholds (ТЗ §4.1, §4.2) ──────────────────

HARD_GEOMETRY = {
    "h_min_floor": 0.05,
    "deltaH_amp_cap": 0.15,
}

SMOOTH_SANITY_VS_CYL = {
    "h_min_ratio_min": 0.85,       # h_smooth_profile / h_smooth_cyl >= 0.85
    "p_max_ratio_max": 1.25,       # <= 1.25
    "friction_ratio_max": 1.10,    # <= 1.10
}

# ── Objective constants (ТЗ §7) ────────────────────────────────────

J_SCREEN_WEIGHTS: Dict[float, float] = {
    0.30: 0.20,
    0.40: 0.30,
    0.50: 0.50,
}

J_EPS_COEFFS = {"h": 2.0, "p": 1.5, "f": 0.5, "cav": 2.0}
J_EQ_COEFFS = {"h": 2.5, "p": 2.0, "f": 0.5, "cav": 2.0}

HARD_TEXTURE_FAIL_PER_EPS = {
    "h_r_min": 0.97,
    "p_r_max": 1.08,
    "f_r_max": 1.08,
    "c_d_max": 0.03,
}

# ε-fail count at which whole profile is marked screen_fail (ТЗ §7.3)
SCREEN_FAIL_AT_EPS_FAIL_COUNT = 2

EQ_USEFUL_GATES = {
    "h_r_min_excl": 1.005,   # strict >
    "p_r_max": 1.000,        # <=
    "f_r_max": 1.02,         # <=
    "c_d_max": 0.02,         # <=
}


# ── Screening budget + pass-rate policy (clarification 5) ─────────

SCREENING_LHS_SEED = 20260416
SCREENING_N_LHS_PER_FAMILY = 48
SCREENING_N_REFINE_PER_FAMILY = 8
SCREENING_N_REFINE_BASES = 2  # top-2 within each family
SCREENING_FAMILIES: List[str] = ["two_lobe", "local_bump", "mixed_harmonic"]

CONFIRM_TOP_N = 12
EQUILIBRIUM_TOP_N = 6
EQUILIBRIUM_FINE_TOP_N = 3

# Effective-budget sanity bands (ТЗ clarification 5):
#   >= 50% pass rate — ok
#   < 30%            — warning (ranges too aggressive, fix/restart)
PASS_RATE_OK = 0.50
PASS_RATE_WARN = 0.30


# ── Dataclasses for payloads ──────────────────────────────────────

@dataclass
class ProfileSpec:
    """Macro bore profile parametrization."""
    family: str
    params: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return dict(family=self.family, params=dict(self.params))

    def hash(self) -> str:
        return profile_hash(self.as_dict())


@dataclass
class TextureSpec:
    """Reference texture — fixed on first pass."""
    zone_deg: List[float] = field(default_factory=lambda: [0.0, 90.0])
    hp_um: float = 30.0
    a_mm: float = 1.5
    b_mm: float = 1.2
    sf: float = 1.5
    profile: str = "smoothcap"

    def as_dict(self) -> Dict[str, Any]:
        return canonical_spec(asdict(self))

    def hash(self) -> str:
        blob = json.dumps(self.as_dict(),
                           sort_keys=True, ensure_ascii=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class ExperimentSpec:
    """Pair of (profile, texture) — the ONLY allowed pair source.

    `experiment_id` объединяет profile и texture hash, плюс suffix case
    приклеивается на уровне pair (smooth|textured) — см. coexp_pairing.
    """
    profile: ProfileSpec
    texture: TextureSpec
    family: str  # duplicate of profile.family for quick access
    profile_id: str  # hash-based id

    @property
    def experiment_id(self) -> str:
        return f"exp_{self.profile_id}_{self.texture.hash()[:4]}"


def make_experiment_spec(profile_spec: Dict[str, Any],
                          texture_spec: Optional[Dict[str, Any]] = None
                          ) -> ExperimentSpec:
    """Фабрика ExperimentSpec с канонизацией и hash-id."""
    ps = ProfileSpec(family=profile_spec["family"],
                     params=dict(profile_spec["params"]))
    tx = TextureSpec(**(texture_spec or REFERENCE_TEXTURE))
    return ExperimentSpec(
        profile=ps, texture=tx,
        family=ps.family,
        profile_id=ps.hash(),
    )


__all__ = [
    "SCHEMA_VERSION",
    "REFERENCE_TEXTURE",
    "SCREENING_EPS",
    "DEFAULT_WY_SHARE",
    "DEFAULT_TOL_ACCEPT", "DEFAULT_STEP_CAP", "DEFAULT_EPS_MAX",
    "EQUILIBRIUM_NR_SEED_X0", "EQUILIBRIUM_NR_SEED_Y0",
    "HARD_GEOMETRY", "SMOOTH_SANITY_VS_CYL",
    "J_SCREEN_WEIGHTS", "J_EPS_COEFFS", "J_EQ_COEFFS",
    "HARD_TEXTURE_FAIL_PER_EPS",
    "SCREEN_FAIL_AT_EPS_FAIL_COUNT",
    "EQ_USEFUL_GATES",
    "SCREENING_LHS_SEED",
    "SCREENING_N_LHS_PER_FAMILY", "SCREENING_N_REFINE_PER_FAMILY",
    "SCREENING_N_REFINE_BASES", "SCREENING_FAMILIES",
    "CONFIRM_TOP_N", "EQUILIBRIUM_TOP_N", "EQUILIBRIUM_FINE_TOP_N",
    "PASS_RATE_OK", "PASS_RATE_WARN",
    "ProfileSpec", "TextureSpec", "ExperimentSpec",
    "make_experiment_spec",
]
