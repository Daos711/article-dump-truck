"""Pair builder — the ONLY allowed source of (smooth, textured) closures.

Ключевой инвариант ТЗ §2: для любого `ProfileExperimentSpec` возвращаются
ДВЕ closure (smooth + textured), которые гарантированно используют
**один и тот же** `bore_profile_fn`. Никаких отдельных API, принимающих
голый bore_profile_fn снаружи.

Обе closure помечены атрибутами:
  * __case__          — "smooth" или "textured"
  * __profile_hash__  — hash общего profile_spec
  * __bore_profile_fn__ — ссылка на ОДИН И ТОТ ЖЕ callable
  * __experiment_id__ — `exp_<profile_id>_<tex_id>`

Contract test T2 (scripts/../tests/test_coexp_pipeline.py)
monkeypatches bore_profile_fn через подмену атрибута и проверяет,
что обе closure зовут ИМЕННО этот объект.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .bore_profiles import make_bore_profile
from .coexp_schema import ExperimentSpec, make_experiment_spec


@dataclass
class PairBuilders:
    smooth: Callable[..., np.ndarray]
    textured: Callable[..., np.ndarray]
    bore_profile_fn: Callable[[np.ndarray], np.ndarray]
    experiment_id: str
    profile_hash: str


def _build_texture_centers(texture_spec_dict, R, L):
    """Скопировано 1:1 из magnetic_v4 run_mag_textured_compare.setup_texture
    (zone 0-90°, sf=1.5, ...). Формулы идентичны — reference texture
    не должна меняться в coexp pipeline.
    """
    zone = texture_spec_dict["zone_deg"]
    a_mm = float(texture_spec_dict["a_mm"])
    b_mm = float(texture_spec_dict["b_mm"])
    hp_um = float(texture_spec_dict["hp_um"])
    sf = float(texture_spec_dict["sf"])
    profile = str(texture_spec_dict["profile"])

    a_phi = (b_mm * 1e-3) / R
    a_Z = 2.0 * (a_mm * 1e-3) / L

    phi_s = np.deg2rad(float(zone[0]))
    phi_e = np.deg2rad(float(zone[1]))
    phi_span = phi_e - phi_s
    N_phi_tex = max(1, int(phi_span / (sf * 2.0 * a_phi)))
    N_Z_tex = max(1, int(2.0 / (sf * 2.0 * a_Z)))

    margin = a_phi * 1.1
    usable = phi_span - 2.0 * margin
    if N_phi_tex == 1:
        phi_c = np.array([phi_s + phi_span / 2.0])
    else:
        phi_c = phi_s + margin + np.linspace(0.0, usable, N_phi_tex)

    margin_Z = a_Z * 1.1
    usable_Z = 2.0 - 2.0 * margin_Z
    if N_Z_tex == 1:
        Z_c = np.array([0.0])
    else:
        Z_c = -1.0 + margin_Z + np.linspace(0.0, usable_Z, N_Z_tex)

    pg, zg = np.meshgrid(phi_c, Z_c)
    return (
        pg.ravel(), zg.ravel(), a_phi, a_Z,
        (hp_um * 1e-6),  # dimensional depth (converted to H/c by caller)
        profile,
    )


def make_H_pair_builders(
    experiment_spec: ExperimentSpec,
    R: float, L: float, c: float,
    sigma: float,
    texture_relief_fn: Optional[Callable[..., np.ndarray]] = None,
) -> PairBuilders:
    """Построить пару smooth/textured closure для одного эксперимента.

    Parameters
    ----------
    texture_relief_fn : callable or None
        Выбрасываем единственную точку I/O в solver —
        reynolds_solver.utils.create_H_with_ellipsoidal_depressions —
        наружу, чтобы тесты могли подменить её без установки PS.

    Returns
    -------
    PairBuilders
        .smooth(eps, Phi_mesh, Z_mesh) -> H
        .textured(eps, Phi_mesh, Z_mesh) -> H
        обе используют ТОТ ЖЕ bore_profile_fn; textured добавляет ТОЛЬКО
        texture relief поверх smooth H.
    """
    if texture_relief_fn is None:  # lazy import — keeps tests PS-free
        from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
        texture_relief_fn = create_H_with_ellipsoidal_depressions

    spec_dict = experiment_spec.profile.as_dict()
    bore_fn = make_bore_profile(spec_dict)
    tex_dict = experiment_spec.texture.as_dict()
    phi_c, Z_c, a_phi, a_Z, depth_m, profile = _build_texture_centers(
        tex_dict, R, L)
    depth_nondim = depth_m / c

    def smooth_fn(eps: float, Phi_mesh: np.ndarray,
                  Z_mesh: np.ndarray) -> np.ndarray:
        H0 = 1.0 + float(eps) * np.cos(Phi_mesh)
        H0 = H0 + bore_fn(Phi_mesh)
        H0 = np.sqrt(H0 ** 2 + (sigma / c) ** 2)
        return H0

    def textured_fn(eps: float, Phi_mesh: np.ndarray,
                    Z_mesh: np.ndarray) -> np.ndarray:
        H0 = smooth_fn(eps, Phi_mesh, Z_mesh)
        H = texture_relief_fn(
            H0, depth_nondim, Phi_mesh, Z_mesh,
            phi_c, Z_c, a_Z, a_phi, profile=profile)
        return H

    # Stamp metadata so contract tests can verify.
    smooth_fn.__case__ = "smooth"
    textured_fn.__case__ = "textured"
    smooth_fn.__experiment_id__ = experiment_spec.experiment_id
    textured_fn.__experiment_id__ = experiment_spec.experiment_id
    smooth_fn.__profile_hash__ = experiment_spec.profile_id
    textured_fn.__profile_hash__ = experiment_spec.profile_id
    # Same object reference for bore profile → required for T2.
    smooth_fn.__bore_profile_fn__ = bore_fn
    textured_fn.__bore_profile_fn__ = bore_fn

    return PairBuilders(
        smooth=smooth_fn, textured=textured_fn,
        bore_profile_fn=bore_fn,
        experiment_id=experiment_spec.experiment_id,
        profile_hash=experiment_spec.profile_id,
    )


def build_from_profile_spec(
        profile_spec_dict: Dict[str, Any],
        R: float, L: float, c: float, sigma: float,
        texture_spec_dict: Optional[Dict[str, Any]] = None,
        texture_relief_fn: Optional[Callable[..., np.ndarray]] = None,
) -> PairBuilders:
    """Удобная фабрика: из голого profile spec dict → PairBuilders."""
    exp = make_experiment_spec(profile_spec_dict, texture_spec_dict)
    return make_H_pair_builders(exp, R, L, c, sigma,
                                 texture_relief_fn=texture_relief_fn)


__all__ = [
    "PairBuilders",
    "make_H_pair_builders",
    "build_from_profile_spec",
]
