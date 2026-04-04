"""Быстрый smoke-test: проверка импортов, сетки, закона нагрузки."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_grid():
    from models.bearing_model import setup_grid
    phi, Z, Pm, Zm, dp, dz = setup_grid(100)
    assert phi.shape == (100,)
    assert Pm.shape == (100, 100)
    assert dp > 0 and dz > 0


def test_texture_centers():
    from models.bearing_model import setup_texture
    from config import pump_params as params
    phi_c, Z_c = setup_texture(params)
    assert len(phi_c) == params.N_phi_tex * params.N_Z_tex
    assert np.all(phi_c >= 0) and np.all(phi_c <= 2 * np.pi)
    assert np.all(Z_c >= 0) and np.all(Z_c <= 1)


def test_make_H_smooth():
    from models.bearing_model import setup_grid, make_H
    from config import pump_params as params
    _, _, Pm, Zm, _, _ = setup_grid(50)
    H = make_H(0.5, Pm, Zm, params, textured=False)
    assert H.shape == (50, 50)
    assert np.min(H) > 0, "Зазор должен быть положительным"
    assert np.isclose(np.min(H), 0.5, atol=0.01)


def test_load_diesel():
    from models.diesel_quasistatic import load_diesel
    phi = np.linspace(0, 720, 360)
    F = load_diesel(phi)
    assert F.shape == (360,)
    assert np.all(F >= 1000), "Минимальная нагрузка 1000 Н"
    # Пик вблизи φ=370°
    idx_max = np.argmax(F)
    assert 350 < phi[idx_max] < 390, f"Пик при φ={phi[idx_max]}°, ожидается ~370°"


def test_configs():
    from models.pump_steady import CONFIGS as pump_cfg
    from models.diesel_quasistatic import CONFIGS as diesel_cfg
    assert len(pump_cfg) == 4
    assert len(diesel_cfg) == 4


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
