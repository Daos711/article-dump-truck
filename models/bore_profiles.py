"""Macro bore profile families for co-design (ТЗ coexp_v1).

Три семейства: two-lobe (A), local raised-cosine (B), low-order mixed
harmonic (C). Все ΔH_bore(φ) безразмерны (единица = радиальный зазор c)
и НЕ зависят от Z.

Циклические фазовые интервалы обрабатываются явно: параметризация
через unwrapped range (lo..hi_unwrapped с автоматическим mod 360 внутри),
см. make_phase_sampler / wrap_deg. Это закрывает clarification (1)
из ТЗ — LHS не ломается на разрыве 360°→0°.

Profile hash (clarification 3) — округление float до 6 знаков ДО
json.dumps+sha256, чтобы избежать флап-тестов из-за бит-репрезентации.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


# ─── Phase/angle helpers (cyclic-aware) ────────────────────────────

def wrap_deg(angle_deg: float) -> float:
    """Завернуть угол в [0, 360)."""
    return float(angle_deg) % 360.0


def wrap_rad(angle_rad: float) -> float:
    return float(angle_rad) % (2.0 * math.pi)


def cyclic_range_unwrap(lo_deg: float, hi_deg: float) -> Tuple[float, float]:
    """Раскрутить циклический интервал в монотонную ось для LHS.

    Для интервала, который идёт «через 0°» (lo > hi), возвращается
    (lo, hi + 360). Sampler берёт uniform на [lo, hi_unwrapped], а
    потом вызывающий код делает `% 360`.

    Примеры:
      [330, 60] → (330, 420)
      [0, 60]   → (0, 60)
      [90, 270] → (90, 270)
    """
    lo = float(lo_deg)
    hi = float(hi_deg)
    if hi < lo:
        hi = hi + 360.0
    return lo, hi


# ─── Profile spec → ΔH_bore callable ───────────────────────────────

def _eval_twolobe(phi_rad: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    A2 = float(params["A2"])
    phi2 = math.radians(float(params["phi2_deg"]))
    return A2 * np.cos(2.0 * (phi_rad - phi2))


def _raised_cosine(phi_rad: np.ndarray, phi0_rad: float,
                    sigma_rad: float) -> np.ndarray:
    """1/2·(1+cos(π·(φ-φ0)/σ)) внутри [-σ, σ], 0 вне. φ циклично."""
    d = np.mod(phi_rad - phi0_rad + math.pi, 2.0 * math.pi) - math.pi
    out = np.zeros_like(phi_rad, dtype=float)
    mask = np.abs(d) <= sigma_rad
    out[mask] = 0.5 * (1.0 + np.cos(math.pi * d[mask] / sigma_rad))
    return out


def _eval_localbump(phi_rad: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    Ab = float(params["Ab"])
    phi0 = math.radians(float(params["phi0_deg"]))
    sigma = math.radians(float(params["sigma_deg"]))
    B = _raised_cosine(phi_rad, phi0, sigma)
    # Среднее снимается на ПЕРИОДИЧЕСКОЙ оси φ ∈ [0, 2π). Используем ту
    # же ось, что и вызывающий код (Phi_mesh строится linspace endpoint=False).
    # Для корректности мы считаем mean на том массиве, который дал phi_rad.
    B_mean = float(np.mean(B))
    return Ab * (B - B_mean)


def _eval_mixed(phi_rad: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    A2 = float(params["A2"])
    A3 = float(params["A3"])
    phi0 = math.radians(float(params["phi0_deg"]))
    return (A2 * np.cos(2.0 * (phi_rad - phi0))
            + A3 * np.cos(3.0 * (phi_rad - phi0)))


FAMILIES: Dict[str, Dict[str, Any]] = {
    "two_lobe": {
        "evaluator": _eval_twolobe,
        "param_names": ["A2", "phi2_deg"],
        "ranges": {
            "A2": (0.00, 0.12),
            "phi2_deg": (330.0, 60.0),  # cyclic
        },
        "cyclic": {"phi2_deg": True},
    },
    "local_bump": {
        "evaluator": _eval_localbump,
        "param_names": ["Ab", "phi0_deg", "sigma_deg"],
        "ranges": {
            "Ab": (-0.10, +0.10),
            "phi0_deg": (300.0, 120.0),  # cyclic
            "sigma_deg": (10.0, 40.0),
        },
        "cyclic": {"phi0_deg": True},
    },
    "mixed_harmonic": {
        "evaluator": _eval_mixed,
        "param_names": ["A2", "A3", "phi0_deg"],
        "ranges": {
            "A2": (-0.08, +0.08),
            "A3": (-0.04, +0.04),
            "phi0_deg": (330.0, 90.0),  # cyclic
        },
        "cyclic": {"phi0_deg": True},
    },
}


def make_bore_profile(spec: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Собрать callable `bore_profile_fn(phi_rad) -> ΔH_bore(phi)`.

    Parameters
    ----------
    spec : dict
        Обязательные ключи: "family", "params".

    Returns
    -------
    callable(phi_rad: np.ndarray) → np.ndarray
        Возвращает ΔH_bore той же формы. Нормализация не меняется.
    """
    family = spec["family"]
    if family not in FAMILIES:
        raise ValueError(f"unknown bore family {family!r}")
    evaluator = FAMILIES[family]["evaluator"]
    params = copy.deepcopy(spec["params"])

    def _fn(phi_rad: np.ndarray) -> np.ndarray:
        return evaluator(np.asarray(phi_rad, dtype=float), params)

    # attach metadata — useful for contract tests and manifests
    _fn.__family__ = family
    _fn.__params__ = params
    _fn.__spec__ = copy.deepcopy(spec)
    return _fn


def evaluate_deltaH(spec: Dict[str, Any], phi_rad: np.ndarray) -> np.ndarray:
    """Short-cut without capturing closure."""
    return make_bore_profile(spec)(phi_rad)


# ─── Spec ⇄ dict + hashing ─────────────────────────────────────────

def _round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Рекурсивно округляет float-значения в dict/list/tuple.

    Отвечает за воспроизводимость hash (clarification 3). Ровно `ndigits`
    знаков — baseline значение 6 покрывает и LHS precision (~1e-6), и
    ширину допустимых диапазонов (`A2 ≤ 0.12`).
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj  # hash по строке сохранит 'NaN'/'Infinity'
        return round(obj, ndigits)
    if isinstance(obj, (int, bool)) or obj is None:
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    # np scalar
    try:
        return _round_floats(obj.item(), ndigits)  # type: ignore
    except AttributeError:
        return obj


def canonical_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Канонический вид spec: rounded floats + sorted keys on serialize."""
    return _round_floats(spec, ndigits=6)


def profile_hash(spec: Dict[str, Any]) -> str:
    """SHA-256 hex (16 chars) от канонизированного spec.

    Хеш стабилен для одного и того же spec при разных sys/OS/numpy-версиях
    (используется только python-float + json сериализация).
    """
    canonical = canonical_spec(spec)
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def profile_to_dict(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Safe copy — уже dict, но гарантируем deepcopy + canonicalization."""
    return canonical_spec(copy.deepcopy(spec))


def profile_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Sanity-check schema + возвращает копию spec."""
    if "family" not in d or "params" not in d:
        raise ValueError("spec missing family/params")
    if d["family"] not in FAMILIES:
        raise ValueError(f"unknown family {d['family']!r}")
    required = FAMILIES[d["family"]]["param_names"]
    for k in required:
        if k not in d["params"]:
            raise ValueError(
                f"spec for {d['family']} missing param {k!r}")
    return copy.deepcopy(d)


# ─── Geometry checks (hard rejects, ТЗ §4.1) ───────────────────────

def deltaH_extrema(spec: Dict[str, Any],
                    phi_rad: np.ndarray | None = None) -> Tuple[float, float]:
    """Вернуть (min, max) ΔH_bore на дискретной периодической сетке."""
    if phi_rad is None:
        phi_rad = np.linspace(0.0, 2.0 * math.pi, 2048, endpoint=False)
    vals = evaluate_deltaH(spec, phi_rad)
    return float(np.min(vals)), float(np.max(vals))


def hard_geometry_fail(spec: Dict[str, Any],
                        eps_values: List[float],
                        h_min_floor: float = 0.05,
                        deltaH_amp_cap: float = 0.15,
                        phi_rad: np.ndarray | None = None) -> Tuple[bool, str]:
    """True если ТЗ §4.1 reject. Возвращает (fail, reason_if_fail).

    Правила:
      min_{φ} H_smooth = 1 − |ε| + ΔH_min  ≤ 0.05  → fail (для любого ε)
      max |ΔH_bore| > 0.15                       → fail
      NaN/Inf                                    → fail
    """
    if phi_rad is None:
        phi_rad = np.linspace(0.0, 2.0 * math.pi, 2048, endpoint=False)
    try:
        vals = evaluate_deltaH(spec, phi_rad)
    except Exception as e:
        return True, f"evaluator raised: {e}"
    if not np.all(np.isfinite(vals)):
        return True, "NaN/Inf in deltaH"
    dmin, dmax = float(np.min(vals)), float(np.max(vals))
    amp = max(abs(dmin), abs(dmax))
    if amp > deltaH_amp_cap:
        return True, f"|deltaH|_max={amp:.4f} > {deltaH_amp_cap}"
    # Smooth H without regularization: H = 1 + eps*cos(phi) + deltaH
    # min_phi ≈ (1 − eps) + min_phi(deltaH) но cos и deltaH имеют
    # разные минимумы. Корректно: min_phi [1 + eps*cos(phi) + deltaH(phi)].
    for eps in eps_values:
        H = 1.0 + float(eps) * np.cos(phi_rad) + vals
        if float(np.min(H)) <= h_min_floor:
            return True, (f"min H_smooth={float(np.min(H)):.4f} "
                          f"at eps={eps} ≤ {h_min_floor}")
    return False, ""


# ─── Public API list ────────────────────────────────────────────────
__all__ = [
    "FAMILIES",
    "make_bore_profile",
    "evaluate_deltaH",
    "profile_hash",
    "profile_to_dict",
    "profile_from_dict",
    "canonical_spec",
    "cyclic_range_unwrap",
    "wrap_deg",
    "wrap_rad",
    "deltaH_extrema",
    "hard_geometry_fail",
]
