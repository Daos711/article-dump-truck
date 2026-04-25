"""LHS sampling + local refinement + ranking for screening (ТЗ §5).

Cyclic-aware sampling (clarification 1): любые фазовые параметры с
cyclic=True сэмплятся в unwrapped-диапазоне [lo, hi_unwrapped], затем
приводятся `% 360`. LHS-точки детерминированы при фиксированном seed
(contract T4).

Ranking: только non-hard-fail профили попадают в top-N (contract T9).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from .bore_profiles import (
    FAMILIES, cyclic_range_unwrap, wrap_deg, profile_hash,
    hard_geometry_fail,
)
from .coexp_schema import (
    SCREENING_LHS_SEED, SCREENING_N_LHS_PER_FAMILY,
    SCREENING_N_REFINE_PER_FAMILY, SCREENING_N_REFINE_BASES,
    SCREENING_FAMILIES,
    HARD_GEOMETRY, SCREENING_EPS,
    CONFIRM_TOP_N,
)


def _sample_family_lhs(family: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """LHS-выборка параметров для одного семейства.

    Фазовые параметры (cyclic=True) сэмплятся в unwrapped range и
    нормируются `% 360` — закрывает clarification 1.
    """
    fam = FAMILIES[family]
    names = list(fam["param_names"])
    d = len(names)
    sampler = qmc.LatinHypercube(d=d, seed=int(seed))
    unit = sampler.random(n=int(n))  # shape (n, d)

    samples: List[Dict[str, Any]] = []
    for i in range(n):
        params: Dict[str, float] = {}
        for j, name in enumerate(names):
            lo, hi = fam["ranges"][name]
            is_cyclic = bool(fam["cyclic"].get(name, False))
            if is_cyclic:
                lo_u, hi_u = cyclic_range_unwrap(lo, hi)
                x = lo_u + (hi_u - lo_u) * float(unit[i, j])
                params[name] = wrap_deg(x)
            else:
                params[name] = float(lo) + float(hi - lo) * float(unit[i, j])
        samples.append(dict(family=family, params=params))
    return samples


def generate_initial_doe(seed: int = SCREENING_LHS_SEED,
                         n_per_family: int = SCREENING_N_LHS_PER_FAMILY,
                         families: Optional[List[str]] = None
                         ) -> List[Dict[str, Any]]:
    """Сгенерировать initial DOE: n_per_family × len(families) профилей.

    Seed одинаковый для всех семейств — но t.к. dimensionality у каждого
    своя, scipy.qmc берёт независимые последовательности. Для семейств
    с разной размерностью это корректно.
    """
    families = list(families if families is not None else SCREENING_FAMILIES)
    out: List[Dict[str, Any]] = []
    for fam in families:
        out.extend(_sample_family_lhs(fam, n_per_family, seed))
    return out


def generate_local_refinement(
        best_specs: List[Dict[str, Any]],
        n_per_base: int = SCREENING_N_REFINE_PER_FAMILY,
        amp_frac: float = 0.10,
        phase_deg: float = 10.0,
        seed: int = SCREENING_LHS_SEED + 1,
) -> List[Dict[str, Any]]:
    """Вокруг каждой базовой точки сгенерировать local LHS-box.

    Правила box'ов (ТЗ §5):
      * амплитудные параметры: ±amp_frac (10%)
      * фазовые/позиционные (deg-unit): ±phase_deg (10°)
      * sigma_deg трактуется как «шириный» параметр: ±amp_frac·value

    Seed сдвинут от initial-DOE seed, чтобы local-boxes не коллайдились
    с LHS-точками.
    """
    out: List[Dict[str, Any]] = []
    for base_idx, base_spec in enumerate(best_specs):
        family = base_spec["family"]
        fam = FAMILIES[family]
        names = list(fam["param_names"])
        d = len(names)
        local_seed = int(seed) * (base_idx + 1) + 97
        sampler = qmc.LatinHypercube(d=d, seed=local_seed)
        unit = sampler.random(n=int(n_per_base))

        for i in range(n_per_base):
            params: Dict[str, float] = {}
            for j, name in enumerate(names):
                lo_g, hi_g = fam["ranges"][name]
                is_cyclic = bool(fam["cyclic"].get(name, False))
                v0 = float(base_spec["params"][name])
                # pick width
                if name.startswith("phi") or name.endswith("_deg") and name != "sigma_deg":
                    # фазовые / позиционные (deg, cyclic или нет)
                    half = float(phase_deg)
                elif name == "sigma_deg":
                    half = max(amp_frac * abs(v0), 1.0)
                else:
                    # амплитудные (A2, A3, Ab)
                    ref = max(abs(v0), abs(hi_g - lo_g) * 0.25)
                    half = amp_frac * ref
                x = (v0 - half) + 2.0 * half * float(unit[i, j])
                if is_cyclic:
                    params[name] = wrap_deg(x)
                else:
                    # clip в глобальные [lo_g, hi_g] если не cyclic
                    params[name] = float(max(lo_g, min(hi_g, x)))
            out.append(dict(family=family, params=params))
    return out


# ─── Ranking / selection ───────────────────────────────────────────

def filter_hard_fail(specs: List[Dict[str, Any]],
                      eps_values: Optional[List[float]] = None
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Разбить входной список на (survivors, hard_rejects).

    Каждый reject получает `reject_reason` при hard_rejects.
    """
    eps_values = eps_values or list(SCREENING_EPS)
    survivors: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    for spec in specs:
        fail, reason = hard_geometry_fail(
            spec, eps_values,
            h_min_floor=HARD_GEOMETRY["h_min_floor"],
            deltaH_amp_cap=HARD_GEOMETRY["deltaH_amp_cap"])
        if fail:
            rec = copy.deepcopy(spec)
            rec["reject_stage"] = "geometry"
            rec["reject_reason"] = reason
            rejects.append(rec)
        else:
            survivors.append(copy.deepcopy(spec))
    return survivors, rejects


def pass_rate_status(n_total: int, n_pass: int) -> str:
    """Clarification 5: pass-rate alarm bands.

    * >= 50% — "ok"
    * >= 30% — "warn"
    * <  30% — "alarm"
    """
    if n_total <= 0:
        return "alarm"
    rate = n_pass / n_total
    if rate >= 0.50:
        return "ok"
    if rate >= 0.30:
        return "warn"
    return "alarm"


def select_top_per_family(screened: List[Dict[str, Any]],
                           n_top_per_family: int,
                           score_key: str = "J_screen",
                           ) -> List[Dict[str, Any]]:
    """Взять top-N в каждой семье. Hard-fail записи пропускаются
    (соответствует contract T9)."""
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for r in screened:
        if r.get("screen_fail"):
            continue
        j = r.get(score_key)
        if j is None or (isinstance(j, float) and math.isnan(j)):
            continue
        by_family.setdefault(r["family"], []).append(r)
    out: List[Dict[str, Any]] = []
    for fam, items in by_family.items():
        items.sort(key=lambda r: r[score_key], reverse=True)
        out.extend(items[:n_top_per_family])
    return out


def select_top_global(screened: List[Dict[str, Any]],
                       n_top: int = CONFIRM_TOP_N,
                       score_key: str = "J_screen",
                       ) -> List[Dict[str, Any]]:
    """Top-N по всем семействам. Hard-fail записи никогда не попадают
    (T9)."""
    good = [r for r in screened
            if not r.get("screen_fail")
            and r.get(score_key) is not None
            and not (isinstance(r.get(score_key), float)
                     and math.isnan(r[score_key]))]
    good.sort(key=lambda r: r[score_key], reverse=True)
    return good[:int(n_top)]


__all__ = [
    "generate_initial_doe",
    "generate_local_refinement",
    "filter_hard_fail",
    "pass_rate_status",
    "select_top_per_family",
    "select_top_global",
]
