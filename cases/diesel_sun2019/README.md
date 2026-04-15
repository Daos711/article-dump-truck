# Diesel main bearing — Sun et al. (2019)

Инфраструктура для нестационарного расчёта main bearing коленвала
4-тактного дизеля с текстурой и без.

## Источник параметров

Sun, Wang, Hu, Zhu et al. "Effects of Textures on the Dynamic Characteristics
of Main Bearings in Internal Combustion Engines", *Chinese Journal of
Mechanical Engineering* **32**:23 (2019).

Геометрия и рабочий режим (см. `case_config.py`):

| Параметр | Значение |
|----------|----------|
| R (радиус journal) | 40 мм |
| L (ширина подшипника) | 27 мм |
| c (зазор) | 146 мкм |
| η (вязкость) | 0.014347 Па·с |
| n (обороты) | 2200 об/мин |
| Цилиндров | 4 |

## Convention (безразмеризация)

Используется Ausas 2008 convention:
- τ = ω·t — безразмерное время
- H = h/c — безразмерный зазор
- P = p / p_scale — безразмерное давление
- W_nd = F / F₀ — безразмерная сила
- p_scale = 6·η·ω·(R/c)²
- F₀ = 24π·η·ω·R⁴/c²
- M_nd = m·c³·ω / (24π·η·R⁴)

Один 4-тактный цикл (720° CA) = **4π** по безразмерному времени τ.

Все функции конвертации в `scaling.py`.

## Surrogate нагрузка

**Важно:** CSV с реальной нагрузкой из Sun 2019 требует ручной оцифровки
графика. Пока используется **surrogate-модель** — Gaussian-like
gas pressure pulse + центробежная инерция, спроецированные на оси
подшипника через упрощённую кинематику КШМ.

Surrogate НЕ претендует на точное воспроизведение кривой Sun 2019.
Он даёт:
- реалистичный порядок величины (десятки-сотни кН)
- правильную фазу (пик при TDC firing)
- 720°-периодичность
- зависимость амплитуды от load_pct

Параметры surrogate (tunable в `case_config.py`):
- bore, stroke, con_rod — геометрия КШМ
- p_max_MPa — пиковое давление газа (100% load)
- p_motoring_MPa — давление на такте сжатия
- m_piston_kg — масса поршневой группы

## Использование

```bash
# 1. Построить surrogate нагрузку для 2200 rpm, 100% load
python cases/diesel_sun2019/scripts/build_load_from_indicator.py \
    --n-rpm 2200 --load-pct 100 --plot

# → derived/load_main_bearing_2200rpm_100pct.csv       (размерная)
# → derived/load_main_bearing_2200rpm_100pct_nd.csv    (безразмерная)
# → derived/load_main_bearing_2200rpm_100pct_plot.png

# 2. Smooth цикл (реализация в Части 2)
python cases/diesel_sun2019/scripts/run_smooth_cycle.py
```

## Структура

```
diesel_sun2019/
├── case_config.py          # параметры подшипника и расчёта
├── scaling.py              # безразмеризация + load interpolator
├── load_data/              # (зарезервировано) CSV из оцифровки Sun 2019
├── derived/                # выходные CSV preprocessing и результаты
│   ├── load_*.csv
│   └── load_*_nd.csv
├── scripts/
│   ├── build_load_from_indicator.py  # surrogate load builder
│   └── run_smooth_cycle.py           # TBD (Часть 2)
├── results/                # выход transient расчётов
└── README.md
```

## Load interpolator

Для solver'а нужна `load_fn(t) -> (WaX, WaY)`. Использование:

```python
import numpy as np
from cases.diesel_sun2019.scaling import make_load_fn_from_crank

# Загрузить safe-CSV размерный
data = np.genfromtxt("derived/load_main_bearing_2200rpm_100pct.csv",
                      delimiter=",", skip_header=1)
crank_deg, WaX_N, WaY_N = data[:, 0], data[:, 1], data[:, 2]

# Безразмеризовать
from cases.diesel_sun2019 import scaling, case_config as cfg
omega = scaling.omega_from_rpm(cfg.n_rpm)
WaX_nd = scaling.nondim_force(WaX_N, cfg.eta, omega, cfg.R, cfg.c)
WaY_nd = scaling.nondim_force(WaY_N, cfg.eta, omega, cfg.R, cfg.c)

# Периодический интерполятор (период = 4π по τ = 720° CA)
load_fn = make_load_fn_from_crank(crank_deg, WaX_nd, WaY_nd)

# Использовать в solver:
# WaX, WaY = load_fn(t)
```

## Что следует заменить на реальные данные

1. **CSV из Sun 2019 Figure X** — оцифрованная реальная нагрузка.
   Положить в `load_data/sun2019_main_bearing_load_100pct.csv`,
   формат тот же: `crank_deg, WaX_dim_N, WaY_dim_N`.
   Переключить источник в `build_load_from_indicator.py` через флаг.

2. **FE-модель коленвала** (если будет) — daily driver вместо surrogate.

3. **Измеренные обороты и давление цилиндра** — по ним можно скалибровать
   surrogate.
