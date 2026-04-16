"""Магнитные surrogate-модели разгрузки journal bearing.

Используются ТОЛЬКО в уравнении статического равновесия ротора.
PS/Reynolds PDE не изменяется.

Классы:
  BaseMagneticForceModel       — абстрактный интерфейс
  RadialUnloadForceModel       — osesymm. restoring (main default)
  LinearSpringForceModel       — дебаг: W_mag = -k·(X,Y)
  RotatedSectorForceModel      — optional: 4 сектора повёрнутые к minimum-gap direction

Калибровка: прямо (линейно по K_mag) через проекцию W_mag на ê_resist.
"""
import numpy as np


# ─── Base ─────────────────────────────────────────────────────────

class BaseMagneticForceModel:
    """Абстрактный интерфейс. force(X, Y) → (Fx, Fy).

    Параметр K_mag (или k_lin) должен быть линейным масштабом силы,
    чтобы упрощать калибровку.
    """

    def force(self, X, Y):
        raise NotImplementedError

    @property
    def scale(self):
        """Линейный множитель силы (для рескейлинга)."""
        raise NotImplementedError

    @scale.setter
    def scale(self, value):
        raise NotImplementedError


# ─── Radial restoring (main default) ───────────────────────────────

class RadialUnloadForceModel(BaseMagneticForceModel):
    """Осесимметричная restoring-модель.

    F_mag(ε) = K_mag · [1/(H_near+reg)^n - 1/(H_far+reg)^n]
    H_near = max(1 − ε, H_floor),  H_far = 1 + ε
    Направление: -ê_r = -(X, Y)/ε  (всегда к центру).

    При ε < eps_tol → (0, 0).
    """

    def __init__(self, K_mag=0.0, n_mag=3.0, H_reg=0.05, H_floor=0.02,
                 eps_tol=1e-12):
        self.K_mag = float(K_mag)
        self.n_mag = float(n_mag)
        self.H_reg = float(H_reg)
        self.H_floor = float(H_floor)
        self.eps_tol = float(eps_tol)

    @property
    def scale(self):
        return self.K_mag

    @scale.setter
    def scale(self, value):
        self.K_mag = float(value)

    def magnitude(self, eps):
        H_near = max(1.0 - eps, self.H_floor)
        H_far = 1.0 + eps
        return self.K_mag * (
            1.0 / (H_near + self.H_reg) ** self.n_mag
            - 1.0 / (H_far + self.H_reg) ** self.n_mag
        )

    def force(self, X, Y):
        eps = float(np.sqrt(X * X + Y * Y))
        if eps < self.eps_tol or self.K_mag == 0.0:
            return 0.0, 0.0
        F = self.magnitude(eps)
        ex = X / eps
        ey = Y / eps
        return -F * ex, -F * ey


# ─── Linear spring (debug only) ───────────────────────────────────

class LinearSpringForceModel(BaseMagneticForceModel):
    """W_mag = -k_lin · (X, Y). Только для отладки."""

    def __init__(self, k_lin=0.0):
        self.k_lin = float(k_lin)

    @property
    def scale(self):
        return self.k_lin

    @scale.setter
    def scale(self, value):
        self.k_lin = float(value)

    def force(self, X, Y):
        return -self.k_lin * X, -self.k_lin * Y


# ─── Rotated sectors (optional, not for first rerun) ──────────────

class RotatedSectorForceModel(BaseMagneticForceModel):
    """4 сектора, повёрнутые относительно baseline minimum-gap direction.

    phi_0 = atan2(Y_0, X_0). Секторы: phi_0, phi_0+π/2, phi_0+π, phi_0+3π/2.
    Зазор до стенки сектора: H_m = 1 - X·cos(phi_m) - Y·sin(phi_m).
    F_m = K_mag / (max(H_m, H_floor) + H_reg)^n_mag, отталкивание к центру.
    """

    def __init__(self, phi_0_rad=0.0, K_mag=0.0, n_mag=3.0,
                 H_reg=0.05, H_floor=0.02):
        phi_base = np.asarray(phi_0_rad, dtype=float)
        self.phi_m = phi_base + np.array([0.0, 0.5 * np.pi,
                                           np.pi, 1.5 * np.pi])
        self.K_mag = float(K_mag)
        self.n_mag = float(n_mag)
        self.H_reg = float(H_reg)
        self.H_floor = float(H_floor)

    @property
    def scale(self):
        return self.K_mag

    @scale.setter
    def scale(self, value):
        self.K_mag = float(value)

    def force(self, X, Y):
        if self.K_mag == 0.0:
            return 0.0, 0.0
        Hm = 1.0 - X * np.cos(self.phi_m) - Y * np.sin(self.phi_m)
        Heff = np.maximum(Hm, self.H_floor)
        F_m = self.K_mag / (Heff + self.H_reg) ** self.n_mag
        ux = -np.cos(self.phi_m)
        uy = -np.sin(self.phi_m)
        return float(np.sum(F_m * ux)), float(np.sum(F_m * uy))


# ─── Калибровка ───────────────────────────────────────────────────

def calibrate_Kmag_from_baseline_projection(
        model_template, X_baseline, Y_baseline, W_applied_xy,
        unload_share_target):
    """Прямая калибровка: модель линейна по scale → K_mag.

    1. Установить scale=1 у модели
    2. p_0 = (W_mag(X0,Y0) · ê_resist) / ||W_applied||
    3. K_mag = unload_share_target / p_0

    Returns: копия model_template с откалиброванным scale.
    Raises ValueError если p_0 <= 0 (модель не может разгружать).
    """
    import copy
    m = copy.copy(model_template)
    m.scale = 1.0
    Wa = np.asarray(W_applied_xy, dtype=float)
    Wa_norm = float(np.linalg.norm(Wa))
    if Wa_norm < 1e-20:
        raise ValueError("applied load ~0")
    e_resist = -Wa / Wa_norm

    Fx, Fy = m.force(X_baseline, Y_baseline)
    p_0 = (Fx * e_resist[0] + Fy * e_resist[1]) / Wa_norm

    if unload_share_target <= 0.0:
        m.scale = 0.0
        return m
    if p_0 <= 0:
        raise ValueError(
            f"projection на e_resist = {p_0:.3e} ≤ 0 — модель не разгружает")
    m.scale = unload_share_target / p_0
    return m


# ─── Sanity checks ────────────────────────────────────────────────

def sanity_checks(verbose=False, seed=0):
    """Базовые проверки + новые: force-to-center.

    Returns (ok, messages).
    """
    msgs = []
    ok = True

    # 0. Radial model: F(0,0) = 0
    m = RadialUnloadForceModel(K_mag=1.0)
    Fx, Fy = m.force(0.0, 0.0)
    c = abs(Fx) < 1e-12 and abs(Fy) < 1e-12
    msgs.append(f"  [{'✓' if c else '✗'}] Radial: F(0,0) ≈ 0")
    ok = ok and c

    # 1. Radial: force-to-center для случайных точек
    rng = np.random.default_rng(seed)
    fail_pts = []
    for _ in range(20):
        eps = rng.uniform(0.05, 0.8)
        ang = rng.uniform(0, 2 * np.pi)
        X = eps * np.cos(ang)
        Y = eps * np.sin(ang)
        Fx, Fy = m.force(X, Y)
        dot = Fx * X + Fy * Y
        if dot >= 0:
            fail_pts.append((X, Y, dot))
    c = len(fail_pts) == 0
    msgs.append(f"  [{'✓' if c else '✗'}] Radial: W_mag·(X,Y)<0 "
                 f"для 20 случайных точек "
                 f"({len(fail_pts)} fails)")
    ok = ok and c

    # 2. Radial magnitude > 0 при ε>0
    F = m.magnitude(0.3)
    c = F > 0
    msgs.append(f"  [{'✓' if c else '✗'}] Radial: F_magnitude(ε=0.3) "
                 f"= {F:.3e} > 0")
    ok = ok and c

    # 3. LinearSpring: аналитическая проверка
    lm = LinearSpringForceModel(k_lin=2.0)
    Fx, Fy = lm.force(0.3, -0.4)
    c = abs(Fx + 0.6) < 1e-12 and abs(Fy - 0.8) < 1e-12
    msgs.append(f"  [{'✓' if c else '✗'}] LinearSpring: "
                 f"(-k*X, -k*Y) = ({Fx:.3f}, {Fy:.3f})")
    ok = ok and c

    # 4. Calibration check: после калибровки unload_share_actual = target
    W_applied = np.array([0.0, -100.0])  # чистый −Y
    X0, Y0 = 0.4, -0.3                    # baseline
    target = 0.10
    try:
        m_cal = calibrate_Kmag_from_baseline_projection(
            RadialUnloadForceModel(n_mag=3, H_reg=0.05, H_floor=0.02),
            X0, Y0, W_applied, target)
        Fx, Fy = m_cal.force(X0, Y0)
        e_resist = -W_applied / np.linalg.norm(W_applied)
        actual = (Fx * e_resist[0] + Fy * e_resist[1]) / np.linalg.norm(W_applied)
        c = abs(actual - target) < 1e-6
        msgs.append(f"  [{'✓' if c else '✗'}] Calibration: "
                     f"actual={actual:.6f}, target={target:.6f}, "
                     f"diff={actual-target:+.1e}")
        ok = ok and c
    except ValueError as e:
        msgs.append(f"  [✗] Calibration raised: {e}")
        ok = False

    # 5. Baseline unloading check: W_mag · ê_resist > 0
    if m_cal is not None and m_cal.scale > 0:
        Fx, Fy = m_cal.force(X0, Y0)
        proj = Fx * e_resist[0] + Fy * e_resist[1]
        c = proj > 0
        msgs.append(f"  [{'✓' if c else '✗'}] Baseline unloading: "
                     f"W_mag·ê_resist = {proj:.3e} > 0")
        ok = ok and c

    if verbose:
        print("Sanity checks magnetic_force:")
        for m_ in msgs:
            print(m_)
    return ok, msgs


if __name__ == "__main__":
    ok, _ = sanity_checks(verbose=True)
    print("\nРЕЗУЛЬТАТ:", "PASS" if ok else "FAIL")
