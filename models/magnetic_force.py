"""Surrogate модель радиальной восстанавливающей магнитной силы.

Используется только в уравнении статического равновесия ротора:
  Fx_hydro(X,Y) + Fx_mag(X,Y) = Fx_applied
  Fy_hydro(X,Y) + Fy_mag(X,Y) = Fy_applied

Физика — пассивные репульсивные магнитные секторы в корпусе. Каждый
сектор создаёт радиально-центростремительную силу, растущую при
уменьшении macro-зазора до его стенки.

КОНВЕНЦИЯ:
  (X, Y) — вектор смещения центра journal от центра bearing.
  H_m(X,Y,phi_m) = 1 - X·cos(phi_m) - Y·sin(phi_m)   — зазор до
                                                       стенки сектора
  (при X=+0.3 journal сместился в +X → зазор до стенки +X = 1-0.3=0.7.)

  ВАЖНО: в нашем bearing_model используется противоположная конвенция
  H_film = 1 + X·cos(φ). Это значит bearing-"X, Y" это тот же центр
  journal, но pressure рассчитывается по H_film (плёнка в точке φ
  bearing). Магнитный H_m и bearing H_film связаны симметрией:
  H_film(φ) = H_m(φ + π). При интеграции с равновесием знаки обоих
  вкладов в force balance совпадают.

Сила сектора m:
  H_m(X,Y) = 1 - X·cos(phi_m) - Y·sin(phi_m)   (макро-зазор, нормир. на c)
  H_eff_m  = max(H_m, H_floor)
  F_m      = K_mag / (H_eff_m + H_reg)^n_mag
  направление: +(cos(phi_m), sin(phi_m))   (отталкивает от стенки,
                                              т.е. в сторону +phi_m)

ВАЖНО:
  * Магнит НЕ входит в Reynolds / PS PDE.
  * Текстура не модулирует магнитный зазор.
  * Модель — макро-пружина, не FEM магнитостатика.
"""
import numpy as np


class MagneticForceModel:
    """Симметричные пассивные репульсивные секторы в позициях phi_m (град)."""

    def __init__(self, sector_angles_deg=(0.0, 120.0, 240.0),
                 K_mag=0.0, n_mag=3.0, H_reg=0.05, H_floor=0.02):
        self.phi_m = np.deg2rad(np.asarray(sector_angles_deg, dtype=float))
        self.K_mag = float(K_mag)
        self.n_mag = float(n_mag)
        self.H_reg = float(H_reg)
        self.H_floor = float(H_floor)

    def H_m(self, X, Y):
        """Макро-зазор до стенки каждого сектора: H_m shape = (len(phi_m),).

        H_m = 1 - X·cos(phi_m) - Y·sin(phi_m)  (journal center at (X,Y)).
        """
        return 1.0 - X * np.cos(self.phi_m) - Y * np.sin(self.phi_m)

    def force(self, X, Y):
        """Суммарная магнитная сила в точке (X, Y).

        Returns
        -------
        Fx_mag, Fy_mag : float
        """
        if self.K_mag == 0.0:
            return 0.0, 0.0
        Hm = self.H_m(X, Y)
        Heff = np.maximum(Hm, self.H_floor)
        F_m = self.K_mag / (Heff + self.H_reg) ** self.n_mag
        # Сектор отталкивает journal ОТ своей стенки к центру:
        # стенка в точке (cos phi_m, sin phi_m), unit vector от стенки
        # к центру bearing: -(cos phi_m, sin phi_m).
        ux = -np.cos(self.phi_m)
        uy = -np.sin(self.phi_m)
        Fx = float(np.sum(F_m * ux))
        Fy = float(np.sum(F_m * uy))
        return Fx, Fy

    def force_components_per_sector(self, X, Y):
        """Для диагностики: сила и H_m каждого сектора."""
        Hm = self.H_m(X, Y)
        Heff = np.maximum(Hm, self.H_floor)
        F_m = self.K_mag / (Heff + self.H_reg) ** self.n_mag
        return [
            dict(phi_m_deg=float(np.rad2deg(self.phi_m[i])),
                 H_m=float(Hm[i]), F_m=float(F_m[i]))
            for i in range(len(self.phi_m))
        ]


def calibrate_Kmag(X_baseline, Y_baseline, W_applied_xy, mag_share_target,
                   sector_angles_deg=(0.0, 120.0, 240.0),
                   n_mag=3.0, H_reg=0.05, H_floor=0.02,
                   K_lo=0.0, K_hi=None, tol=1e-6, max_iter=80):
    """Подобрать K_mag так, чтобы (W_mag · e_load) / ||W_applied|| = target.

    e_load = -W_applied / ||W_applied||  (направление разгрузки).

    Возвращает MagneticForceModel с откалиброванным K_mag.
    """
    Wa = np.asarray(W_applied_xy, dtype=float)
    Wa_norm = float(np.linalg.norm(Wa))
    if Wa_norm < 1e-20:
        raise ValueError("applied load магнитуда ~0")
    e_load = -Wa / Wa_norm

    def share_at_K(K):
        m = MagneticForceModel(sector_angles_deg=sector_angles_deg,
                                K_mag=K, n_mag=n_mag, H_reg=H_reg,
                                H_floor=H_floor)
        Fx, Fy = m.force(X_baseline, Y_baseline)
        return (Fx * e_load[0] + Fy * e_load[1]) / Wa_norm - mag_share_target

    if mag_share_target <= 0:
        return MagneticForceModel(sector_angles_deg, 0.0, n_mag, H_reg, H_floor)

    # Автоподбор K_hi
    if K_hi is None:
        K_hi = max(1.0, Wa_norm)
        for _ in range(40):
            if share_at_K(K_hi) > 0:
                break
            K_hi *= 10
        else:
            raise RuntimeError("Не удалось найти K_hi где share > target")

    # Bisection
    lo, hi = K_lo, K_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = share_at_K(mid)
        if abs(f) < tol:
            lo = hi = mid
            break
        if f > 0:
            hi = mid
        else:
            lo = mid
    K_out = 0.5 * (lo + hi)
    return MagneticForceModel(sector_angles_deg, K_out, n_mag, H_reg, H_floor)


# ─── Sanity checks (самопроверка) ─────────────────────────────────

def sanity_checks(verbose=False):
    """Обязательные проверки знака и нуля в центре.

    Возвращает (ok, messages).
    """
    msgs = []
    ok = True
    m = MagneticForceModel(K_mag=1.0)

    # 1. F(0,0) = 0
    Fx, Fy = m.force(0.0, 0.0)
    c1 = abs(Fx) < 1e-12 and abs(Fy) < 1e-12
    msgs.append(f"  [{'✓' if c1 else '✗'}] F(0,0) = "
                 f"({Fx:.3e}, {Fy:.3e}) ≈ 0")
    ok = ok and c1

    # 2. X<0, Y=0 → Fx > 0
    Fx, Fy = m.force(-0.3, 0.0)
    c2 = Fx > 1e-6
    msgs.append(f"  [{'✓' if c2 else '✗'}] X=-0.3: Fx={Fx:.3e} > 0")
    ok = ok and c2

    # 3. X>0, Y=0 → Fx < 0
    Fx, Fy = m.force(0.3, 0.0)
    c3 = Fx < -1e-6
    msgs.append(f"  [{'✓' if c3 else '✗'}] X=+0.3: Fx={Fx:.3e} < 0")
    ok = ok and c3

    # 4. Y<0, X=0 → Fy > 0
    Fx, Fy = m.force(0.0, -0.3)
    c4 = Fy > 1e-6
    msgs.append(f"  [{'✓' if c4 else '✗'}] Y=-0.3: Fy={Fy:.3e} > 0")
    ok = ok and c4

    # 5. Y>0, X=0 → Fy < 0
    Fx, Fy = m.force(0.0, 0.3)
    c5 = Fy < -1e-6
    msgs.append(f"  [{'✓' if c5 else '✗'}] Y=+0.3: Fy={Fy:.3e} < 0")
    ok = ok and c5

    if verbose:
        print("Sanity checks MagneticForceModel:")
        for m_ in msgs:
            print(m_)
    return ok, msgs


if __name__ == "__main__":
    ok, _ = sanity_checks(verbose=True)
    print("\nРЕЗУЛЬТАТ:", "PASS" if ok else "FAIL")
