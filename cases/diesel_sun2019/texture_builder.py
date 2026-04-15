"""Texture placement и построение texture_relief для дизельного кейса.

Coordinates (Ausas): φ ∈ [0, 1], Z ∈ [0, B].

Key idea: phi_hmin(t) вычисляется аналитически из X(t), Y(t):
  H(φ) = 1 + X·cos(2π·φ) + Y·sin(2π·φ)  [в координатах Ausas]
  min достигается где d/dφ(X·cos + Y·sin) = 0, т.е. 2π·φ = atan2(Y, X) + π
  φ_hmin = (atan2(Y, X) + π) / (2π)  mod 1

phi_pmax по данным скалярных histories не восстановим точно — для
упрощения используем оценку phi_pmax ≈ phi_hmin - 30°/360° (оффсет
converging region, pressure peak перед минимальным зазором).
"""
import numpy as np

try:
    from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False


def compute_phi_hmin(X, Y):
    """Аналитический угол минимального зазора в координатах Ausas [0, 1]."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    phi = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    return phi % 1.0


def compute_phi_pmax_est(X, Y, lead_deg=30.0):
    """Оценка угла максимума давления (upstream от h_min).

    При вращении в +φ направлении pressure peak расположен перед
    зоной минимального зазора на lead_deg градусов.
    """
    phi_hmin = compute_phi_hmin(X, Y)
    phi_pmax = (phi_hmin - lead_deg / 360.0) % 1.0
    return phi_pmax


def analyze_loaded_arc(X, Y, weight_by="h_min_inverse"):
    """Определить loaded arc: где преимущественно находятся phi_hmin
    во время пиковых нагрузок.

    Parameters
    ----------
    X, Y : (NT,)
    weight_by : 'uniform' or 'h_min_inverse'
        Если 'h_min_inverse' — вес ~1/h_min (пиковые моменты важнее).

    Returns
    -------
    phi_center_deg : центр нагруженной дуги (градусы Ausas φ, 0-360°)
    phi_fwhm_deg   : ширина (FWHM гистограммы)
    """
    phi_hmin = compute_phi_hmin(X, Y)  # [0, 1]

    if weight_by == "h_min_inverse":
        H_min = 1 - np.sqrt(np.asarray(X) ** 2 + np.asarray(Y) ** 2)
        H_min = np.maximum(H_min, 0.01)
        weights = 1.0 / H_min
    else:
        weights = np.ones_like(phi_hmin)

    # Histogram on [0, 1] с периодичностью
    bins = np.linspace(0, 1, 37)  # 10°-bins
    hist, edges = np.histogram(phi_hmin, bins=bins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    i_peak = int(np.argmax(hist))
    phi_center = centers[i_peak]
    phi_center_deg = phi_center * 360.0

    # FWHM
    half = hist[i_peak] / 2
    above = hist > half
    # Найти непрерывный сегмент вокруг i_peak
    if above[i_peak]:
        i_lo = i_peak
        while i_lo > 0 and above[i_lo - 1]:
            i_lo -= 1
        i_hi = i_peak
        while i_hi < len(above) - 1 and above[i_hi + 1]:
            i_hi += 1
        phi_fwhm_deg = (i_hi - i_lo + 1) * 10.0
    else:
        phi_fwhm_deg = 30.0

    return phi_center_deg, phi_fwhm_deg


def suggest_texture_zones(X, Y, zone_width_deg=90.0):
    """Предложить три texture зоны на основе smooth cycle.

    Returns dict: upstream / centered / slightly_downstream.
    Координаты в градусах Ausas φ [0, 360°].
    """
    phi_center_deg, phi_fwhm_deg = analyze_loaded_arc(X, Y)

    # upstream: от центра назад на zone_width (в направлении −φ)
    upstream_start = (phi_center_deg - zone_width_deg) % 360
    upstream_end = phi_center_deg % 360

    # centered
    half = zone_width_deg / 2
    centered_start = (phi_center_deg - half) % 360
    centered_end = (phi_center_deg + half) % 360

    # downstream
    downstream_start = phi_center_deg % 360
    downstream_end = (phi_center_deg + zone_width_deg) % 360

    return {
        "loaded_arc_center_deg": float(phi_center_deg),
        "loaded_arc_fwhm_deg": float(phi_fwhm_deg),
        "upstream": {
            "phi_start_deg": float(upstream_start),
            "phi_end_deg": float(upstream_end),
        },
        "centered": {
            "phi_start_deg": float(centered_start),
            "phi_end_deg": float(centered_end),
        },
        "slightly_downstream": {
            "phi_start_deg": float(downstream_start),
            "phi_end_deg": float(downstream_end),
        },
    }


def build_diesel_texture(N_phi, N_Z, B_ausas, d_phi, d_Z,
                          phi_start_deg, phi_end_deg,
                          dimple_diameter_um, dimple_depth_um,
                          area_density, c_clearance_m, R_bearing_m,
                          L_bearing_m):
    """Построить texture_relief (N_Z, N_phi) — только вклад лунок.

    dimple_diameter_um: физический диаметр лунки в мкм
    area_density: доля площади занятая лунками (0-1)
    """
    # Перевод в безразмерные (Ausas coordinates: φ ∈ [0,1], Z ∈ [0, B])
    # Физический φ-радиан → φ_Ausas: φ_A = φ_rad / (2π)
    # Z физическая → Z_Ausas = z / (2R) (т.к. B = L/(2R))
    r_dimple_m = dimple_diameter_um * 1e-6 / 2

    # Полуоси круглой лунки в координатах Ausas
    r_phi_ausas = r_dimple_m / (2 * np.pi * R_bearing_m)
    r_Z_ausas = r_dimple_m / (2 * R_bearing_m)

    H_p = dimple_depth_um * 1e-6 / c_clearance_m  # безразм. глубина

    # Зона в Ausas φ (0..1)
    phi_s = phi_start_deg / 360.0
    phi_e = phi_end_deg / 360.0
    if phi_e < phi_s:
        zone_span = (1.0 - phi_s) + phi_e
    else:
        zone_span = phi_e - phi_s

    # Spacing из area_density
    # Каждая лунка занимает π·r² физически
    # На ячейку spacing × spacing приходится 1 лунка → density = π·r² / spacing²
    # spacing = sqrt(π·r² / density) = r · sqrt(π/density)
    spacing_m = r_dimple_m * np.sqrt(np.pi / area_density)

    # Число лунок в зоне по φ и Z
    phi_arc_m = zone_span * 2 * np.pi * R_bearing_m
    Z_total_m = L_bearing_m
    N_phi_tex = max(1, int(phi_arc_m / spacing_m))
    N_Z_tex = max(1, int(Z_total_m / spacing_m))

    # Центры лунок (с отступом от края)
    margin_phi = r_phi_ausas * 1.1
    margin_Z = r_Z_ausas * 1.1

    if N_phi_tex == 1:
        phi_centers_1d = np.array([(phi_s + zone_span / 2) % 1.0])
    else:
        usable_span = zone_span - 2 * margin_phi
        if usable_span <= 0:
            return np.zeros((N_Z, N_phi)), dict(N_tex=0, spacing_m=spacing_m)
        phi_centers_1d = (phi_s + margin_phi
                           + np.linspace(0, usable_span, N_phi_tex)) % 1.0

    if N_Z_tex == 1:
        Z_centers_1d = np.array([B_ausas / 2])
    else:
        usable_Z = B_ausas - 2 * margin_Z
        if usable_Z <= 0:
            return np.zeros((N_Z, N_phi)), dict(N_tex=0, spacing_m=spacing_m)
        Z_centers_1d = margin_Z + np.linspace(0, usable_Z, N_Z_tex)

    pg, zg = np.meshgrid(phi_centers_1d, Z_centers_1d)
    phi_c_flat = pg.ravel()
    Z_c_flat = zg.ravel()
    n_total = len(phi_c_flat)

    # Сетка центров ячеек (с ghost)
    phi_arr = np.linspace(-d_phi / 2, 1 + d_phi / 2, N_phi)
    Z_arr = np.linspace(-d_Z / 2, B_ausas + d_Z / 2, N_Z)
    Phi, Zm = np.meshgrid(phi_arr, Z_arr)

    H0 = np.zeros_like(Phi)

    if _HAS_UTILS:
        try:
            relief = create_H_with_ellipsoidal_depressions(
                H0, H_p, Phi, Zm, phi_c_flat, Z_c_flat,
                r_Z_ausas, r_phi_ausas,
                profile="smoothcap")
        except Exception:
            relief = _relief_manual(Phi, Zm, phi_c_flat, Z_c_flat,
                                     r_phi_ausas, r_Z_ausas, H_p)
    else:
        relief = _relief_manual(Phi, Zm, phi_c_flat, Z_c_flat,
                                 r_phi_ausas, r_Z_ausas, H_p)

    return relief, dict(
        N_tex=n_total,
        N_phi_tex=N_phi_tex,
        N_Z_tex=N_Z_tex,
        spacing_m=spacing_m,
        r_phi_ausas=r_phi_ausas,
        r_Z_ausas=r_Z_ausas,
        H_p=H_p,
    )


def _relief_manual(Phi, Zm, phi_c_flat, Z_c_flat,
                    r_phi, r_Z, H_p):
    """Ручной smoothcap relief если utils недоступны."""
    relief = np.zeros_like(Phi)
    for p_c, z_c in zip(phi_c_flat, Z_c_flat):
        dphi = Phi - p_c
        dphi = (dphi + 0.5) % 1.0 - 0.5   # periodic
        dz = Zm - z_c
        r2 = (dphi / r_phi) ** 2 + (dz / r_Z) ** 2
        mask = r2 < 1
        relief[mask] += H_p * (1 - r2[mask]) ** 2
    return relief
