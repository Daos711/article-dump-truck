"""
Валидация: GPU-солвер на грубой и тонкой сетке.
Параметры из старой модели: n=2980, eta=0.01105, h_p=10мкм, N_Z_tex=11, sqrt, без PV.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ─────────────────────────────────────────────────
# ПАРАМЕТРЫ (точно как в старой модели)
# ─────────────────────────────────────────────────
R = 0.035
c = 0.00005
L = 0.056
epsilon = 0.6
n_rpm = 2980
omega = 2 * np.pi * n_rpm / 60
U = omega * (R - c)
eta = 0.01105
h_p = 0.00001   # 10 мкм
H_p = h_p / c

pressure_scale = (6 * eta * U * R) / (c**2)

a_dim = 0.00241
b_dim = 0.002214
A_tex = 2 * a_dim / L
B_tex = b_dim / R

N_phi_tex = 8
N_Z_tex = 11        # как в старой модели — с перекрытием!
phi_start_deg = 90
phi_end_deg = 270
phi_start = np.deg2rad(phi_start_deg)
phi_end = np.deg2rad(phi_end_deg)

delta_phi_gap = (phi_end - phi_start - 2 * N_phi_tex * B_tex) / (N_phi_tex - 1)
delta_phi_center = 2 * B_tex + delta_phi_gap
phi_center_start = phi_start + B_tex
phi_c_values = phi_center_start + delta_phi_center * np.arange(N_phi_tex)

delta_Z_gap = (2 - 2 * N_Z_tex * A_tex) / (N_Z_tex - 1)
delta_Z_center = 2 * A_tex + delta_Z_gap
Z_center_start = -1 + A_tex
Z_c_values = Z_center_start + delta_Z_center * np.arange(N_Z_tex)

phi_c_grid, Z_c_grid = np.meshgrid(phi_c_values, Z_c_values)
phi_c_flat = phi_c_grid.flatten()
Z_c_flat = Z_c_grid.flatten()

load_scale = pressure_scale * (R * L) / 2


def create_H_textured(H0, Phi_mesh, Z_mesh):
    """Текстура sqrt-профиль, параметры старой модели."""
    H = H0.copy()
    for k in range(len(phi_c_flat)):
        phi_c = phi_c_flat[k]
        Z_c = Z_c_flat[k]
        delta_phi = np.arctan2(np.sin(Phi_mesh - phi_c), np.cos(Phi_mesh - phi_c))
        expr = (delta_phi / B_tex)**2 + ((Z_mesh - Z_c) / A_tex)**2
        inside = expr <= 1
        H[inside] += H_p * np.sqrt(1 - expr[inside])
    return H


def compute_W(P, Phi_mesh, phi_1D, Z_1D):
    """Несущая способность (формула старой модели)."""
    Fr = np.trapz(np.trapz(P * np.cos(Phi_mesh), phi_1D, axis=1), Z_1D)
    Ft = np.trapz(np.trapz(P * np.sin(Phi_mesh), phi_1D, axis=1), Z_1D)
    W_nd = np.sqrt(Fr**2 + Ft**2)
    return W_nd * load_scale


from reynolds_solver import solve_reynolds


# ─────────────────────────────────────────────────
# ШАГ 1: ГРУБАЯ СЕТКА 500×500 (endpoint=True)
# ─────────────────────────────────────────────────
print("=" * 70)
print("ШАГ 1: GPU на грубой сетке 500×500 (endpoint=True)")
print("=" * 70)

N = 500
phi_1D = np.linspace(0, 2 * np.pi, N)
Z_1D = np.linspace(-1, 1, N)
Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
d_phi = phi_1D[1] - phi_1D[0]
d_Z = Z_1D[1] - Z_1D[0]

H0 = 1 + epsilon * np.cos(Phi_mesh)
H_smooth = H0.copy()
H_textured = create_H_textured(H0, Phi_mesh, Z_mesh)

t0 = time.time()
P_smooth_c, _, _ = solve_reynolds(H_smooth, d_phi, d_Z, R, L,
                                   closure="laminar", cavitation="half_sommerfeld")
W_smooth_c = compute_W(P_smooth_c, Phi_mesh, phi_1D, Z_1D)

P_text_c, _, _ = solve_reynolds(H_textured, d_phi, d_Z, R, L,
                                 closure="laminar", cavitation="half_sommerfeld")
W_text_c = compute_W(P_text_c, Phi_mesh, phi_1D, Z_1D)
t1 = time.time()

print(f"  Гладкий:  W = {W_smooth_c:.1f} Н")
print(f"  Текстура: W = {W_text_c:.1f} Н")
print(f"  gain = {W_text_c/W_smooth_c:.3f}")
print(f"  Время: {t1-t0:.1f} с")


# ─────────────────────────────────────────────────
# ШАГ 2: ТОНКАЯ СЕТКА 3000×3000 (endpoint=False)
# ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ШАГ 2: GPU на тонкой сетке 3000×3000 (endpoint=False)")
print("=" * 70)

N_phi_fine = 3000
N_Z_fine = 3000
phi_1D_f = np.linspace(0, 2*np.pi, N_phi_fine, endpoint=False)
Z_1D_f = np.linspace(-1, 1, N_Z_fine)
Phi_f, Z_f = np.meshgrid(phi_1D_f, Z_1D_f)
d_phi_f = phi_1D_f[1] - phi_1D_f[0]
d_Z_f = Z_1D_f[1] - Z_1D_f[0]

H0_f = 1 + epsilon * np.cos(Phi_f)
H_smooth_f = H0_f.copy()

print("  Строим текстуру на тонкой сетке...")
H_textured_f = create_H_textured(H0_f, Phi_f, Z_f)

print("  Решаем гладкий...")
t0 = time.time()
P_smooth_f, _, _ = solve_reynolds(H_smooth_f, d_phi_f, d_Z_f, R, L,
                                   closure="laminar", cavitation="half_sommerfeld")
W_smooth_f = compute_W(P_smooth_f, Phi_f, phi_1D_f, Z_1D_f)
print(f"  Гладкий:  W = {W_smooth_f:.1f} Н  ({time.time()-t0:.0f} с)")

print("  Решаем текстуру...")
t0 = time.time()
P_text_f, _, _ = solve_reynolds(H_textured_f, d_phi_f, d_Z_f, R, L,
                                 closure="laminar", cavitation="half_sommerfeld")
W_text_f = compute_W(P_text_f, Phi_f, phi_1D_f, Z_1D_f)
print(f"  Текстура: W = {W_text_f:.1f} Н  ({time.time()-t0:.0f} с)")
print(f"  gain = {W_text_f/W_smooth_f:.3f}")


# ─────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Строим графики...")
print("=" * 70)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "validation")
os.makedirs(outdir, exist_ok=True)

Z_idx_c = np.argmin(np.abs(Z_1D - 0))
Z_idx_f = np.argmin(np.abs(Z_1D_f - 0))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (1) P(φ) грубая
ax = axes[0, 0]
ax.set_title(f"Грубая 500×500: P(φ) при Z=0  |  gain = {W_text_c/W_smooth_c:.3f}", fontsize=11)
ax.plot(phi_1D, P_smooth_c[Z_idx_c, :] * pressure_scale / 1e6, 'b-', lw=1.5, label='Гладкий')
ax.plot(phi_1D, P_text_c[Z_idx_c, :] * pressure_scale / 1e6, 'r-', lw=1.5, label='Текстура')
ax.set_xlabel('φ, рад'); ax.set_ylabel('P, МПа')
ax.legend(); ax.grid(True)

# (2) H(φ) грубая
ax = axes[0, 1]
ax.set_title("Грубая 500×500: H(φ) при Z=0", fontsize=11)
ax.plot(phi_1D, H_smooth[Z_idx_c, :], 'b-', lw=1.5, label='Гладкий')
ax.plot(phi_1D, H_textured[Z_idx_c, :], 'r-', lw=1.5, label='Текстура')
ax.set_xlabel('φ, рад'); ax.set_ylabel('H (безразм.)')
ax.legend(); ax.grid(True)

# (3) P(φ) тонкая
ax = axes[1, 0]
ax.set_title(f"Тонкая 3000×3000: P(φ) при Z=0  |  gain = {W_text_f/W_smooth_f:.3f}", fontsize=11)
ax.plot(phi_1D_f, P_smooth_f[Z_idx_f, :] * pressure_scale / 1e6, 'b-', lw=1.5, label='Гладкий')
ax.plot(phi_1D_f, P_text_f[Z_idx_f, :] * pressure_scale / 1e6, 'r-', lw=1.5, label='Текстура')
ax.set_xlabel('φ, рад'); ax.set_ylabel('P, МПа')
ax.legend(); ax.grid(True)

# (4) H(φ) тонкая
ax = axes[1, 1]
ax.set_title("Тонкая 3000×3000: H(φ) при Z=0", fontsize=11)
ax.plot(phi_1D_f, H_smooth_f[Z_idx_f, :], 'b-', lw=1.5, label='Гладкий')
ax.plot(phi_1D_f, H_textured_f[Z_idx_f, :], 'r-', lw=1.5, label='Текстура')
ax.set_xlabel('φ, рад'); ax.set_ylabel('H (безразм.)')
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "validation_overview.png"), dpi=150)
print(f"  Сохранён: validation_overview.png")

# Zoom в зону текстуры
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
phi_zoom = (np.pi/2, 3*np.pi/2)

ax = axes2[0]
ax.set_title(f"Грубая 500×500: ZOOM зона текстуры  |  gain = {W_text_c/W_smooth_c:.3f}")
mask = (phi_1D >= phi_zoom[0]) & (phi_1D <= phi_zoom[1])
ax.plot(phi_1D[mask], P_smooth_c[Z_idx_c, mask] * pressure_scale / 1e6, 'b-', lw=1, label='Гладкий')
ax.plot(phi_1D[mask], P_text_c[Z_idx_c, mask] * pressure_scale / 1e6, 'r-', lw=1, label='Текстура')
ax.set_xlabel('φ, рад'); ax.set_ylabel('P, МПа')
ax.legend(); ax.grid(True)

ax = axes2[1]
ax.set_title(f"Тонкая 3000×3000: ZOOM зона текстуры  |  gain = {W_text_f/W_smooth_f:.3f}")
mask = (phi_1D_f >= phi_zoom[0]) & (phi_1D_f <= phi_zoom[1])
ax.plot(phi_1D_f[mask], P_smooth_f[Z_idx_f, mask] * pressure_scale / 1e6, 'b-', lw=1, label='Гладкий')
ax.plot(phi_1D_f[mask], P_text_f[Z_idx_f, mask] * pressure_scale / 1e6, 'r-', lw=1, label='Текстура')
ax.set_xlabel('φ, рад'); ax.set_ylabel('P, МПа')
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "validation_zoom.png"), dpi=150)
print(f"  Сохранён: validation_zoom.png")

# СВОДКА
print("\n" + "=" * 70)
print("СВОДКА")
print("=" * 70)
print(f"Параметры: n={n_rpm}, eta={eta}, h_p={h_p*1e6:.0f}мкм, "
      f"N_Z_tex={N_Z_tex}, sqrt, без PV, ε={epsilon}")
print()
print(f"{'Сетка':<20} {'W_smooth':>10} {'W_text':>10} {'gain':>8}")
print("-" * 50)
print(f"{'500×500':<20} {W_smooth_c:>10.1f} {W_text_c:>10.1f} {W_text_c/W_smooth_c:>8.3f}")
print(f"{'3000×3000':<20} {W_smooth_f:>10.1f} {W_text_f:>10.1f} {W_text_f/W_smooth_f:>8.3f}")
print()

plt.show()